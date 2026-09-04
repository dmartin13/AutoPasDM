# DistributedAutoPas

DistributedAutoPas is a distributed particle-container layer around AutoPas.
It keeps distributed ownership, migration, halo exchange, adaptive load balancing,
and communication out of the simulator while reusing AutoPas for node-level particle
storage, interaction traversal, and auto-tuning.

The current implementation uses a Cartesian process grid in x, y, and z. It starts
from a regular decomposition and can adapt the subdomain boundaries at runtime with
inverted-pressure load balancing while keeping the process-grid topology fixed.
Boundary conditions are selected per dimension. `periodic`, `reflective`, and `none`
are represented in the distributed topology. Reflective boundaries are non-periodic
for migration and halo exchange; the bundled single-site md-flexible example keeps
the physical Lennard-Jones mirror-wall force as application-level physics.

The simulator-facing API deliberately exposes operations on the distributed particle
system instead of the node-level AutoPas container. Existing AutoPas particle types
and interaction functors can therefore be reused while DistributedAutoPas owns the
distributed execution mechanisms.

## Repository layout

```text
DistributedAutoPas/
├── include/distributed_autopas/   # public headers and template code
├── src/                           # compiled non-template implementation
├── examples/
│   ├── minimal_mpi.cpp            # low-level runtime / integration example
│   ├── lj/                        # minimal LJ simulator using DistributedAutoPas
│   └── md-flexible/               # md-flexible integration and regression inputs
├── docs/
└── tests/
```

The expected development workspace is:

```text
<workspace>/
├── AutoPas/
└── DistributedAutoPas/
```

## Build

From the DistributedAutoPas repository:

```bash
cmake -S . -B build \
  -DAUTOPAS_SOURCE_DIR=../AutoPas \
  -DMD_FLEXIBLE_USE_MPI=ON \
  -DMD_FLEXIBLE_FUNCTOR_AUTOVEC=ON

cmake --build build -j
```

The adapted md-flexible executable is then available at:

```text
build/examples/md-flexible/md-flexible
```

The minimal LJ example is available at:

```text
build/examples/lj/lj
```

and a small md-flexible-particle integration example at:

```text
build/examples/md-flexible/distributed-lj-example
```

## Current feature set

- Cartesian domain decomposition in one, two, or three dimensions.
- Per-dimension `periodic`, `reflective`, and `none` boundary topology.
- Root-owned and already-distributed initial particle insertion.
- Staged x -> y -> z particle migration through direct face neighbors.
- Staged x -> y -> z halo exchange with propagation to edge and corner neighbors.
- Split-phase non-blocking MPI payload transport for timestep migration and halo exchange.
- Adaptive regular-grid decomposition with inverted-pressure load balancing.
- Adaptive ownership lookup using the complete process-plane split positions.
- A compact operation-based public API for particle updates, reductions, diagnostics,
  distributed synchronization, and AutoPas interaction functors.
- Integration with the bundled single-site md-flexible example, including the
  `None` and `InvertedPressure` load-balancer settings and `load-balancing-interval`.

The current implementation deliberately keeps several research directions open.
Communication/computation overlap is not used yet, MPI is the only communication
backend, and md-flexible's optional ALL/TENSOR load balancer is not supported.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details and design constraints.

## Tests

DistributedAutoPas tests use GoogleTest and are enabled by default for a standalone
checkout. CTest launches MPI tests with their required rank counts. The suite covers,
among other things:

- Cartesian and adaptive domain geometry,
- initial particle redistribution,
- staged migration and halo exchange in 3D,
- periodic, reflective, and open (`none`) topology,
- split-phase non-blocking neighbor communication,
- the public DistributedAutoPas interaction and operation API,
- inverted-pressure boundary movement and minimum-width enforcement,
- repeated adaptive load-balancing steps,
- adaptive `targetRank()` ownership, and
- an eight-rank 3D end-to-end load-balancing, migration, halo, and force check.

All tests can be run with:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### md-flexible 3D force regression

A small end-to-end regression input is available at
`examples/md-flexible/input/dap_force_compare_3d.yaml`. It compares a single-rank
reference run with an eight-rank `2 x 2 x 2` decomposition and exercises internal
face, edge, and corner interactions as well as periodic boundaries in all three
dimensions.

From `build/examples/md-flexible`:

```bash
rm -rf force_compare_3d/np1 force_compare_3d/np8
mkdir -p force_compare_3d/np1 force_compare_3d/np8

./md-flexible \
  --yaml-filename input/dap_force_compare_3d.yaml \
  --vtk-output-folder force_compare_3d/np1

mpirun -np 8 ./md-flexible \
  --yaml-filename input/dap_force_compare_3d.yaml \
  --vtk-output-folder force_compare_3d/np8

python3 scripts/compare_vtk_forces.py \
  force_compare_3d/np1 \
  force_compare_3d/np8
```

The comparison should report the same owned-particle IDs and forces within the
configured tolerance.

### md-flexible load-balancing regression

`examples/md-flexible/input/dap_load_balancing_compare.yaml` and
`scripts/run_load_balancing_regression.py` compare the bundled DistributedAutoPas
integration against an MPI-enabled original md-flexible build. Both decompositions
must adapt and the final global particle state is compared independently of the exact
owner rank, because runtime-based work measurements can produce different adaptive
split positions.

For example:

```bash
python3 scripts/run_load_balancing_regression.py \
  --reference /path/to/AutoPas/build/examples/md-flexible/md-flexible \
  --distributed ./md-flexible \
  --config input/dap_load_balancing_compare.yaml \
  --ranks 8
```

### md-flexible reflective-boundary regression

The single-site md-flexible example contains a reflective-wall regression input at
`examples/md-flexible/input/dap_reflective_compare.yaml`. It places particles near
all six global faces and two opposite corners. With sigma = epsilon = 1 and a wall
distance of 0.5, the expected force component from each reflective wall is exactly
24. The same setup can be compared between one rank and an eight-rank `2 x 2 x 2`
decomposition.

From `build/examples/md-flexible`:

```bash
rm -rf reflective_compare/np1 reflective_compare/np8
mkdir -p reflective_compare/np1 reflective_compare/np8

./md-flexible \
  --yaml-filename input/dap_reflective_compare.yaml \
  --vtk-output-folder reflective_compare/np1

mpirun -np 8 ./md-flexible \
  --yaml-filename input/dap_reflective_compare.yaml \
  --vtk-output-folder reflective_compare/np8

python3 scripts/compare_vtk_forces.py \
  reflective_compare/np1 \
  reflective_compare/np8

python3 scripts/check_reflective_forces.py reflective_compare/np8
```

The first script checks decomposition independence. The second checks the forces
against the analytical values of the reflective regression setup.

The test build can be disabled with `-DDISTRIBUTED_AUTOPAS_BUILD_TESTS=OFF`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current architecture,
design constraints, and remaining limitations.
