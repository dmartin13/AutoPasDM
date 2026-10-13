# AutoPasDM Architecture

## Responsibility split

The dependency direction is:

```text
Simulator / md-flexible
        |
        | particle operations and AutoPas interaction functors
        v
AutoPasDM
        |
        +-- distributed ownership / domain decomposition
        +-- migration
        +-- halo management
        +-- adaptive load balancing
        +-- communication
        |
        +--> node-level AutoPas
```

The simulator is executed once per process/rank, but it should not contain MPI,
NCCL, NVSHMEM, or other communication-backend calls. It describes the physical
model and invokes operations on the distributed particle system. AutoPasDM
owns the distributed algorithms, while AutoPas remains responsible for node-level
particle storage, traversal, and tuning.

This separation is also the basis for future distributed-level tuning. The current
implementation provides one adaptive decomposition / load-balancing strategy and one
communication backend; alternative decomposition, balancing, and transport choices
can later become additional tunable dimensions without moving distributed logic back
into the simulator.

## Public API direction

The public API describes operations, not local storage details. Examples:

```cpp
particles.applyToOwnedParticles(PositionUpdate{dt});
particles.applyToOwnedParticlesInRegion(wallMin, wallMax, WallOperation{});
auto localKineticEnergy = particles.sumOwnedParticles(0.0, KineticEnergy{});
particles.prepareInteractions();
applyApplicationBoundaryPhysics(particles);
particles.computeInteractionsPrepared(&ljFunctor);
auto globalCount = particles.getGlobalNumberOfOwnedParticles();
```

Local AutoPas iterators, halo exchange, migration, load-balancing communication, and
communication contexts are implementation details. `prepareInteractions()` exposes
only the high-level preparation phase, not the individual migration or halo
operations. This allows an application to insert physics such as a reflective wall
force before invoking one or more AutoPas functors on the prepared state.
`computeInteractions()` remains the convenience operation for the common
prepare-and-compute case.

The public API intentionally provides no direct access to the node-level AutoPas
container. Diagnostics and output consume particle traversal and local tuning
metadata through dedicated AutoPasDM operations. Spatially restricted
application kernels use `applyToOwnedParticlesInRegion()`. AutoPasDM clips
the requested region to the rank-local ownership box and delegates the traversal to
AutoPas without exposing its region iterator to the application.

## Domain decomposition

`DomainDecomposition` represents a Cartesian process grid whose topology stays fixed
for the lifetime of the distributed container. Construction starts from an equally
spaced regular grid. The process grid is generated from the number of ranks and the
selected subdivision dimensions.

Adaptive load balancing changes only the positions of the process-plane boundaries;
it does not change rank coordinates or the process-grid dimensions. Each rank stores
the complete split-position vector for x, y, and z. This is required because
`targetRank()` must remain correct after the decomposition is no longer uniform.
The rank-local ownership box is derived from these split positions and the rank's
Cartesian coordinates.

When adaptive bounds are applied, all ranks exchange their proposed local bounds and
reconstruct the complete split vectors. Shared boundaries proposed by neighboring
ranks must agree exactly. Only after the global adaptive geometry is consistent is
the node-level AutoPas box resized.

## Initial particle distribution

Particle insertion distinguishes between replicated and already distributed input.
Replicated configuration input uses `addParticlesFromRoot()`, so only one logical copy
is routed to owners. Checkpoint pieces that are already distributed across processes
use `addDistributedParticles()`, where every process contributes its local input
collection. In both cases, AutoPasDM determines rank ownership and the
simulator does not use a particle communicator.

Initialization uses an all-to-all exchange because a particle may initially belong to
any rank. This is intentionally separate from timestep migration, which exploits the
assumption that ordinary particle motion is local.

After adaptive load balancing, initial/distributed insertion still uses `targetRank()`
on the current split vectors rather than assuming equally sized subdomains.

## Runtime and communication context

`autopasdm::Runtime` owns the process communication runtime. The current backend is MPI,
but the application does not call MPI directly. Runtime duplicates a private MPI
communicator for AutoPasDM so library messages do not collide with
communication from other components using `MPI_COMM_WORLD`.

Application utilities that need collectives use backend-neutral Runtime operations.
Examples are `globalSum()`, `barrier()`, and `broadcastString()`. This keeps MPI
collectives out of the simulator and its helper components.

A future communication-backend design may separate control communication from
particle-data transport, for example MPI for process/topology control and GPU-aware
MPI, NCCL, or NVSHMEM for GPU-resident particle data. The current implementation has
not introduced this abstraction yet; MPI remains visible inside the internal
communication and load-balancing implementation.

## Particle serialization

AutoPasDM does not assume a concrete particle layout on the wire. Particle
transport is parameterized by `ParticleSerializer<Particle>`. Applications can
specialize or replace this serialization policy for their particle type while the
migration and halo algorithms remain unchanged.

Communication therefore transfers serialized byte buffers instead of depending on a
matching MPI datatype for each application particle class. The request object used by
the split-phase transport owns the serialized send and receive buffers so their
lifetime covers the outstanding MPI requests.

## Direct-neighbor transport

The direct-neighbor particle transport provides a split-phase
`beginLeftRightExchange()` / `finishLeftRightExchange()` API. The begin phase first
performs a small count handshake to determine receive-buffer sizes and then posts the
particle payload with `MPI_Irecv` / `MPI_Isend`. The payload can remain in flight until
`finishLeftRightExchange()` completes the requests and deserializes the received
particles.

The request object owns the buffers associated with outstanding operations. Its
destructor waits for unfinished requests as a safety mechanism, preventing a caller
from accidentally destroying the transport state while MPI still references the
buffers.

`exchangeLeftRight()` retains the original blocking `MPI_Sendrecv` implementation as
a reference path. Timestep migration and halo exchange use the split-phase
non-blocking transport, but currently call `finishLeftRightExchange()` immediately
after `beginLeftRightExchange()`. Thus the payload transport is non-blocking while the
algorithm currently has no communication/computation overlap. Keeping these steps
separate makes overlap a later optimization rather than a prerequisite for the
current algorithms.

## Timestep migration

Timestep migration is staged dimension by dimension in x -> y -> z. Each stage
communicates only with the preceding and succeeding face neighbor. A particle that
moves diagonally can therefore reach an edge or corner owner through multiple face
exchanges within one migration call rather than requiring direct communication with
all 26 possible 3D neighbors.

The current algorithm assumes that a particle moves by at most one subdomain in each
dimension per timestep. After all three stages, a particle that still does not belong
to the local ownership box is reported as an error. This assumption keeps ordinary
timestep migration local; initialization uses the separate all-to-all path because
it cannot make the same assumption.

Periodic global boundaries wrap particle coordinates before routing them to the
opposite side. `BoundaryType::none` intentionally drops particles that leave the
global domain. `BoundaryType::reflective` is non-periodic and treats a crossing as an
error because the application-level reflective force is expected to keep the
particle inside the box.

## Halo exchange

Halo exchange is also staged x -> y -> z. Each stage starts with the owned particles
plus halos received in previous dimensions. Forwarding those received particles in
later stages automatically generates edge and corner halos without explicit
communication with all 26 neighbors.

Periodic exchanges shift received particles into the periodic image adjacent to the
local ownership box. `none` and `reflective` boundaries suppress communication across
the corresponding global face.

The halo width passed by `AutoPasDM` is currently the interaction cutoff.
The load-balancing minimum domain width, in contrast, uses
`2 * (cutoff + Verlet skin)` to match the adaptive-grid constraint used by
md-flexible's inverted-pressure scheme.

## Adaptive load balancing

AutoPasDM currently implements the inverted-pressure regular-grid load
balancer. It keeps the Cartesian process topology fixed and proposes new local
ownership bounds from an application-provided local work measurement.

For each dimension, the local work is averaged over the process-grid plane
perpendicular to that dimension. Neighboring planes exchange their average work and
outer boundary. Every rank in one plane therefore computes the same shared boundary
with the adjacent plane. The theoretical balanced boundary is obtained with
`balanceAdjacentDomains()` and only half of the displacement is applied in one
balancing step. This damping follows md-flexible's inverted-pressure approach and
avoids overly aggressive geometry changes.

A minimum subdomain width of `2 * (cutoff + Verlet skin)` is enforced. The load
balancer itself only proposes bounds; applying them is intentionally owned by
`AutoPasDM`.

The adaptive synchronization sequence is:

```text
AutoPas::updateContainer() using the old local box
        |
        v
compute new local bounds with InvertedPressureLoadBalancer
        |
        v
collect and validate all process-plane split positions
        |
        v
AutoPas::resizeBox() to the new local box
        |
        v
merge updateContainer() emigrants and resizeBox() emigrants
        |
        v
migrate with respect to the new domain
        |
        v
rebuild halos
```

This order handles both particles that left the old AutoPas box through motion and
particles that become emigrants only because the ownership boundary moved.

The bundled md-flexible integration requests a balancing step only when
`load-balancer` is `InvertedPressure`, the simulation is active, and the configured
`load-balancing-interval` is reached. The computational work is measured on the
application side before distributed synchronization. `None` uses the static path.
md-flexible's optional ALL/TENSOR load balancer is intentionally rejected because it
requires different migration semantics and is not implemented in AutoPasDM.

## Boundary physics

Distributed topology and boundary physics are deliberately separated.
`DomainDecomposition` only decides whether a global boundary communicates, wraps, or
has no neighbor. A reflective boundary is therefore non-periodic from the distributed
layer's perspective.

The bundled single-site md-flexible example applies its original Lennard-Jones
mirror-wall model through a separate `ReflectiveBoundary` component after distributed
preparation and before the ordinary AutoPas interaction functor. Only thin
wall-adjacent regions are traversed through `applyToOwnedParticlesInRegion()`.

This keeps application-specific physical forces out of migration and halo code while
still allowing AutoPasDM to provide the spatial operation needed to apply
them efficiently.

## Particle-local and distributed reductions

Simulator components such as the thermostat express local reductions through
`sumOwnedParticles()` and distributed reductions through `globalSum()`. Output
coordination uses Runtime operations such as `broadcastString()`. These components do
not perform MPI collectives themselves. The simulator defines the physical quantity;
AutoPasDM owns how values are traversed or reduced across ranks.

## Bundled example applications

### Minimal LJ example

`examples/lj` is a deliberately small simulator that demonstrates the intended
application-facing interface. It uses AutoPas' `MoleculeLJ` particle and ordinary LJ
functor, while AutoPasDM handles distributed ownership, migration, and halo
exchange. No explicit MPI call is needed in the simulator code.

### md-flexible

The adapted md-flexible simulator lives in `examples/md-flexible`. It is an
application of AutoPasDM and may still depend directly on AutoPas for
particle types, interaction functors, and configuration options. The active simulation
path no longer constructs or stores md-flexible's legacy `RegularGridDecomposition`;
VTK domain output obtains local/global bounds and the domain rank from
AutoPasDM. AutoPasDM is the active distributed ownership and
communication layer in this example.

Particle insertion distinguishes between two input ownership models. For replicated
configuration input, `autopasdm::Runtime` tells the application which process expands the
configured objects into particles. Object generation in `MDFlexConfig` therefore does
not query MPI or the current rank. Those particles are inserted with
`addParticlesFromRoot()`. Distributed checkpoint pieces use
`addDistributedParticles()`, where every rank contributes its local input collection.

The current bundled integration supports md-flexible's `None` and `InvertedPressure`
load-balancer configurations. The optional ALL/TENSOR load balancer is not supported.
The integration is restricted to `MD_FLEXIBLE_MODE=SINGLESITE`.

## Compiled and templated components

AutoPasDM is intentionally a mixed compiled/template library.
Particle-dependent components remain in headers so arbitrary application particle
types and serializers can instantiate them. Backend-independent, non-template
implementation such as `DomainDecomposition` and non-template `Runtime` operations is
compiled into the `autopasdm` library from `src/`.

The current `Runtime` header still contains the templated reduction helper and the
private MPI communicator type. Moving the concrete communication backend fully behind
an internal interface is a later communication-layer refactoring and is deliberately
separate from the current implementation.

## Current limitations and planned extension points

- MPI is the only communication backend.
- The split-phase neighbor transport is used without actual communication/computation
  overlap.
- The count handshake in `beginLeftRightExchange()` completes before the payload
  requests are returned to the caller.
- Ordinary timestep migration supports at most one crossed subdomain per dimension
  and timestep.
- Halo exchange currently uses the cutoff as halo width.
- Adaptive load balancing keeps the Cartesian process topology fixed.
- Inverted Pressure is the only distributed load-balancing strategy implemented in
  AutoPasDM; md-flexible's ALL/TENSOR option is not supported.
- The bundled md-flexible integration currently supports the single-site particle
  model only.
- Distributed algorithm / communication-backend selection is not yet part of the
  runtime tuning loop.
- GPU-resident particle transport and GPU-oriented communication backends are future
  extension points.

These limitations are deliberately separated from the simulator-facing API. Future
work can therefore evolve communication, load balancing, overlap, and distributed
tuning without requiring simulation applications to reimplement the distributed
algorithms.
