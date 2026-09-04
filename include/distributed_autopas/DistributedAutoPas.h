#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autopas/AutoPas.h"
#include "autopas/options/IteratorBehavior.h"
#include "autopas/utils/WrapOpenMP.h"
#include "distributed_autopas/BoundaryType.h"
#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/HaloExchange.h"
#include "distributed_autopas/ParticleMigration.h"
#include "distributed_autopas/ParticleSerialization.h"
#include "distributed_autopas/PerformanceCounters.h"
#include "distributed_autopas/Runtime.h"
#include "distributed_autopas/load_balancing/InvertedPressureLoadBalancer.h"

namespace dap {

/**
 * Distributed particle container built around a node-local AutoPas instance.
 *
 * The public interface is intentionally expressed in terms of operations on a
 * distributed particle system rather than in terms of MPI or local AutoPas
 * iterators. The simulator describes what should happen to owned particles and
 * DistributedAutoPas decides how the local storage and communication are handled.
 *
 * @tparam Particle AutoPas-compatible particle type stored in the distributed container.
 * @tparam Serializer Serialization policy used to communicate the particle type.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class DistributedAutoPas {
 public:
  /**
   * Construct a distributed container with the legacy one-dimensional x decomposition.
   *
   * The x dimension is periodic, while y and z are not subdivided and use non-periodic boundaries.
   *
   * @param runtime Process runtime used for distributed communication.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param cutoff Interaction cutoff used to size the halo region.
   */
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff)
      : DistributedAutoPas(
            runtime, globalMin, globalMax, cutoff, std::array<bool, 3>{true, false, false},
            std::array<BoundaryType, 3>{BoundaryType::periodic, BoundaryType::none, BoundaryType::none}) {}

  /**
   * Construct a distributed container with an automatically generated Cartesian process grid.
   *
   * All global boundaries are periodic.
   *
   * @param runtime Process runtime used for distributed communication.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param cutoff Interaction cutoff used to size the halo region.
   * @param subdivideDimensions Selects which Cartesian dimensions may be subdivided.
   */
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff, const std::array<bool, 3> &subdivideDimensions)
      : DistributedAutoPas(
            runtime, globalMin, globalMax, cutoff, subdivideDimensions,
            std::array<BoundaryType, 3>{BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic}) {}

  /**
   * Construct a distributed container with an automatically generated Cartesian process grid and explicit boundaries.
   *
   * @param runtime Process runtime used for distributed communication.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param cutoff Interaction cutoff used to size the halo region.
   * @param subdivideDimensions Selects which Cartesian dimensions may be subdivided.
   * @param boundaryTypes Boundary condition applied at the global boundary of each dimension.
   */
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff, const std::array<bool, 3> &subdivideDimensions,
                     const std::array<BoundaryType, 3> &boundaryTypes)
      : DistributedAutoPas(runtime, globalMin, globalMax, cutoff, subdivideDimensions, boundaryTypes, [](auto &) {}) {}

  /**
   * Construct and configure the node-local AutoPas instance before initialization.
   *
   * This overload keeps the legacy x-periodic decomposition for backwards compatibility. The configurator is invoked
   * after box and cutoff setup and before AutoPas::init().
   *
   * @tparam Configurator Callable accepting a reference to the node-local AutoPas instance.
   * @param runtime Process runtime used for distributed communication.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param cutoff Interaction cutoff used to size the halo region.
   * @param configurator Callback used to configure the node-local AutoPas instance.
   */
  template <class Configurator>
  requires std::invocable < Configurator, autopas::AutoPas<Particle>
  & > DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin,
                         const std::array<double, 3> &globalMax, double cutoff, Configurator &&configurator)
      : DistributedAutoPas(runtime, globalMin, globalMax, cutoff, std::array<bool, 3>{true, false, false},
                           std::array<BoundaryType, 3>{BoundaryType::periodic, BoundaryType::none, BoundaryType::none},
                           std::forward<Configurator>(configurator)) {}

  /**
   * Construct and configure a distributed container with an automatically generated Cartesian process grid.
   *
   * All global boundaries are periodic. The configurator is invoked after box and cutoff setup and before
   * AutoPas::init().
   *
   * @tparam Configurator Callable accepting a reference to the node-local AutoPas instance.
   * @param runtime Process runtime used for distributed communication.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param cutoff Interaction cutoff used to size the halo region.
   * @param subdivideDimensions Selects which Cartesian dimensions may be subdivided.
   * @param configurator Callback used to configure the node-local AutoPas instance.
   */
  template <class Configurator>
  requires std::invocable < Configurator, autopas::AutoPas<Particle>
  & > DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin,
                         const std::array<double, 3> &globalMax, double cutoff,
                         const std::array<bool, 3> &subdivideDimensions, Configurator &&configurator)
      : DistributedAutoPas(
            runtime, globalMin, globalMax, cutoff, subdivideDimensions,
            std::array<BoundaryType, 3>{BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic},
            std::forward<Configurator>(configurator)) {}

  /**
   * Construct and configure a distributed container with a Cartesian process grid
   * and explicit per-dimension boundary conditions.
   *
   * periodic, reflective, and none are accepted as distributed boundary topology.
   * Reflective boundaries are non-periodic for migration and halo exchange. The
   * physical wall force itself is intentionally application-defined and is not
   * applied by DistributedAutoPas. The configurator is invoked after box and cutoff setup and before AutoPas::init().
   *
   * @tparam Configurator Callable accepting a reference to the node-local AutoPas instance.
   * @param runtime Process runtime used for distributed communication.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param cutoff Interaction cutoff used to size the halo region.
   * @param subdivideDimensions Selects which Cartesian dimensions may be subdivided.
   * @param boundaryTypes Boundary condition applied at the global boundary of each dimension.
   * @param configurator Callback used to configure the node-local AutoPas instance.
   */
  template <class Configurator>
  requires std::invocable < Configurator, autopas::AutoPas<Particle>
  & > DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin,
                         const std::array<double, 3> &globalMax, double cutoff,
                         const std::array<bool, 3> &subdivideDimensions,
                         const std::array<BoundaryType, 3> &boundaryTypes, Configurator &&configurator)
      : _runtime(runtime),
  _domain(runtime.rank(), runtime.size(), globalMin, globalMax, subdivideDimensions, boundaryTypes),
  _particleMigration(runtime.communicator() DAP_DETAILED_COUNTERS_ARGUMENT(&_performanceCounters)),
  _haloExchange(runtime.communicator() DAP_DETAILED_COUNTERS_ARGUMENT(&_performanceCounters)), _cutoff(cutoff) {
    _autoPas.setBoxMin(_domain.localMin());
    _autoPas.setBoxMax(_domain.localMax());
    _autoPas.setCutoff(_cutoff);

    std::forward<Configurator>(configurator)(_autoPas);

    _autoPas.init();
  }

  /**
   * Add one particle directly to the node-local AutoPas container.
   *
   * The particle must already belong to this rank's local ownership box. Use addDistributedParticles() or
   * addParticlesFromRoot() when ownership is not known by the caller.
   *
   * @param particle Particle owned by this rank.
   */
  void addParticle(const Particle &particle) { _autoPas.addParticle(particle); }

  /**
   * Add a particle collection that conceptually originates on one root rank.
   *
   * All ranks must call this collective operation with the same root rank. Only the collection passed on rootRank is
   * considered. Collections present on the other ranks are ignored. This is useful for replicated application input
   * such as deterministic particles generated from a shared configuration file.
   *
   * @tparam Collection Particle collection type that supports range-based iteration.
   * @param particles Particle collection considered on rootRank.
   * @param rootRank Rank whose particle collection is redistributed.
   */
  template <class Collection>
  void addParticlesFromRoot(const Collection &particles, int rootRank = 0) {
    if (rootRank < 0 or rootRank >= _domain.numRanks()) {
      throw std::runtime_error("DistributedAutoPas: invalid root rank for particle insertion.");
    }

    if (_runtime.rank() == rootRank) {
      addDistributedParticles(particles);
    } else {
      const std::vector<Particle> noParticles;
      addDistributedParticles(noParticles);
    }
  }

  /**
   * Redistribute application-provided particles according to the current domain decomposition and add locally owned
   * particles to the node-local AutoPas container.
   *
   * Every rank contributes its local input collection and must call this collective operation. Particles are sent
   * directly to their owning rank, so the application does not need a particle communicator or access to the
   * distributed communication context. This operation is intended for initialization and checkpoint loading, not for
   * timestep migration. Particles outside the global simulation box are ignored.
   *
   * @tparam Collection Particle collection type that supports range-based iteration.
   * @param particles Particles contributed by this rank.
   */
  template <class Collection>
  void addDistributedParticles(const Collection &particles) {
    std::vector<std::vector<Particle>> particlesByDestination(static_cast<std::size_t>(_domain.numRanks()));

    for (const auto &particle : particles) {
      const auto &position = particle.getR();
      const auto &globalMin = _domain.globalMin();
      const auto &globalMax = _domain.globalMax();

      const bool insideGlobalDomain = position[0] >= globalMin[0] and position[0] < globalMax[0] and
                                      position[1] >= globalMin[1] and position[1] < globalMax[1] and
                                      position[2] >= globalMin[2] and position[2] < globalMax[2];
      if (not insideGlobalDomain) {
        // TODO: add a warning here
        continue;
      }

      particlesByDestination[static_cast<std::size_t>(_domain.targetRank(position))].push_back(particle);
    }

    auto localParticles =
        exchangeParticlesByRank<Particle, Serializer>(_runtime.communicator(), particlesByDestination);
    _autoPas.addParticles(localParticles);
  }

  /**
   * Execute a particle-local kernel for every particle owned by this process.
   *
   * The kernel must be safe to execute concurrently for different particles. Today
   * this is implemented with the node-local AutoPas iterator and OpenMP. Keeping the
   * operation at this level allows a future implementation to dispatch the same
   * operation to a GPU without exposing container iterators to the simulator.
   *
   * @tparam Kernel Callable accepting a mutable particle reference.
   * @param kernel Particle-local operation executed for every owned particle.
   */
  template <class Kernel>
  void applyToOwnedParticles(Kernel &&kernel) {
    auto &&kernelRef = kernel;
    AUTOPAS_OPENMP(parallel)
    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      kernelRef(*iter);
    }
  }

  /**
   * Execute a particle-local kernel for owned particles inside a spatial region.
   *
   * The requested region is intersected with this rank's local ownership box before
   * it is forwarded to AutoPas. This keeps applications independent of the local
   * container while allowing AutoPas to use its optimized region iterator.
   *
   * @tparam Kernel Callable accepting a mutable particle reference.
   * @param regionMin Lower corner of the requested region.
   * @param regionMax Upper corner of the requested region.
   * @param kernel Particle-local operation.
   */
  template <class Kernel>
  void applyToOwnedParticlesInRegion(const std::array<double, 3> &regionMin, const std::array<double, 3> &regionMax,
                                     Kernel &&kernel) {
    std::array<double, 3> clippedMin{};
    std::array<double, 3> clippedMax{};

    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      if (regionMin[dimension] > regionMax[dimension]) {
        throw std::invalid_argument("DistributedAutoPas: region minimum must not exceed region maximum.");
      }

      clippedMin[dimension] = std::max(regionMin[dimension], _domain.localMin()[dimension]);
      clippedMax[dimension] = std::min(regionMax[dimension], _domain.localMax()[dimension]);

      if (clippedMin[dimension] >= clippedMax[dimension]) {
        return;
      }
    }

    auto &&kernelRef = kernel;
    AUTOPAS_OPENMP(parallel)
    for (auto iter = _autoPas.getRegionIterator(clippedMin, clippedMax, autopas::IteratorBehavior::owned);
         iter.isValid(); ++iter) {
      kernelRef(*iter);
    }
  }

  /**
   * Sum a particle-local quantity over all particles owned by this process.
   *
   * This is a local reduction. Distributed reductions are intentionally kept
   * separate via globalSum(), so callers can combine several local quantities
   * before triggering communication. The operation hides the node-local AutoPas
   * iterator and can later be implemented by a device reduction.
   *
   * @tparam Value Arithmetic reduction type.
   * @tparam Transform Callable mapping a particle to one contribution of type Value.
   * @param initialValue Initial value of the local sum.
   * @param transform Particle-local transformation whose results are added to the sum.
   */
  template <class Value, class Transform>
  [[nodiscard]] Value sumOwnedParticles(Value initialValue, Transform &&transform) const {
    auto result = initialValue;
    auto &&transformRef = transform;
    AUTOPAS_OPENMP(parallel reduction(+ : result))
    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      result += transformRef(*iter);
    }
    return result;
  }

  /**
   * Read-only traversal intended for diagnostics and output. Unlike
   * applyToOwnedParticles(), this is deliberately sequential to preserve output order.
   *
   * @tparam Visitor Callable accepting a const particle reference.
   * @param visitor Operation executed sequentially for every owned particle.
   */
  template <class Visitor>
  void forEachOwnedParticle(Visitor &&visitor) const {
    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      visitor(*iter);
    }
  }

  /**
   * Prepare the distributed particle state for one or more interaction calculations.
   *
   * This updates the node-local container, migrates particles that changed ownership,
   * and rebuilds the halo particles. Applications that need to apply additional
   * physics between communication and the ordinary AutoPas interaction kernel can
   * call this method explicitly.
   */
  void prepareInteractions() { synchronizeParticles(); }

  /**
   * Prepare the distributed particle state and update the local ownership bounds using
   * inverted-pressure load balancing.
   *
   * The work value is the application-provided computational load measured for this rank.
   * All ranks must call this overload together. The minimum subdomain width is derived from
   * the node-local AutoPas halo width in the same way as md-flexible:
   * 2 * (cutoff + Verlet skin).
   *
   * @param localWork Computational work measured on this MPI rank.
   */
  void prepareInteractions(double localWork) { synchronizeParticlesWithLoadBalancing(localWork); }

  /**
   * Prepare the distributed particle state while applying updated local ownership bounds.
   *
   * This overload is the integration point for adaptive regular-grid load balancing. The
   * node-local AutoPas container is updated using the old box first, then resized to the
   * new box before all resulting emigrants are migrated with respect to the updated domain.
   * Neighboring ranks must provide matching shared boundaries.
   *
   * @param localMin Updated lower corner of this rank's ownership box.
   * @param localMax Updated upper corner of this rank's ownership box.
   */
  void prepareInteractions(const std::array<double, 3> &localMin, const std::array<double, 3> &localMax) {
    synchronizeParticles(localMin, localMax);
  }

  /**
   * Execute an AutoPas functor on a particle state prepared by prepareInteractions().
   *
   * No migration or halo exchange is triggered here. This is useful when several
   * interaction functors operate on the same prepared distributed state or when
   * application-level physics, such as a reflective wall force, must be applied
   * after communication and before the ordinary particle interactions. The boolean result has the same semantics as
   * AutoPas::computeInteractions() and indicates whether the call was a tuning iteration.
   *
   * @tparam Functor AutoPas interaction functor type.
   * @param functor Interaction functor executed on the prepared node-local AutoPas state.
   */
  template <class Functor>
  bool computeInteractionsPrepared(Functor *functor) {
    return _autoPas.computeInteractions(functor);
  }

  /**
   * Synchronize the distributed particle state and execute one ordinary AutoPas functor.
   *
   * This convenience operation keeps the original single-call API. Applications that
   * need an explicit preparation phase should use prepareInteractions() followed by
   * computeInteractionsPrepared(). The boolean result has the same semantics as AutoPas::computeInteractions() and
   * indicates whether the interaction call was a tuning iteration.
   *
   * @tparam Functor AutoPas interaction functor type.
   * @param functor Interaction functor executed after distributed synchronization.
   */
  template <class Functor>
  bool computeInteractions(Functor *functor) {
    prepareInteractions();
    return computeInteractionsPrepared(functor);
  }

  /** Get the number of particles owned by this rank. */
  [[nodiscard]] std::size_t getLocalNumberOfOwnedParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::owned);
  }

  /** Get the number of halo particles stored by this rank. */
  [[nodiscard]] std::size_t getLocalNumberOfHaloParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::halo);
  }

  /** Get the number of owned and halo particles stored by this rank. */
  [[nodiscard]] std::size_t getLocalNumberOfParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::ownedOrHalo);
  }

  /** Get the global number of owned particles summed over all ranks. */
  [[nodiscard]] std::size_t getGlobalNumberOfOwnedParticles() const {
    return globalSum(getLocalNumberOfOwnedParticles());
  }

  /** Get the rank in the DistributedAutoPas runtime. */
  [[nodiscard]] int rank() const { return _runtime.rank(); }

  /** Get the total number of ranks in the DistributedAutoPas runtime. */
  [[nodiscard]] int numberOfRanks() const { return _runtime.size(); }

  /** Check whether this process is rank zero. */
  [[nodiscard]] bool isRoot() const { return _runtime.isRoot(); }

  /** Synchronize all ranks in the DistributedAutoPas runtime. */
  void barrier() const { _runtime.barrier(); }

  /**
   * Sum one scalar value over all ranks.
   *
   * @tparam T Supported arithmetic type.
   * @param localValue Value contributed by this rank.
   */
  template <class T>
  [[nodiscard]] T globalSum(T localValue) const {
    return _runtime.globalSum(localValue);
  }

  /** Get the lower corner of this rank's local ownership box. */
  [[nodiscard]] const std::array<double, 3> &localBoxMin() const { return _domain.localMin(); }

  /** Get the upper corner of this rank's local ownership box. */
  [[nodiscard]] const std::array<double, 3> &localBoxMax() const { return _domain.localMax(); }

  /** Get the lower corner of the global simulation box. */
  [[nodiscard]] const std::array<double, 3> &globalBoxMin() const { return _domain.globalMin(); }

  /** Get the upper corner of the global simulation box. */
  [[nodiscard]] const std::array<double, 3> &globalBoxMax() const { return _domain.globalMax(); }

  /** Get the boundary conditions of the global simulation box. */
  [[nodiscard]] const std::array<BoundaryType, 3> &boundaryTypes() const { return _domain.boundaryTypes(); }

  // Local AutoPas metadata that is still used by md-flexible for tuning statistics.
  // These methods deliberately expose values, not the local container itself.

  /** Check whether the node-local AutoPas tuning search space contains at most one configuration. */
  [[nodiscard]] bool localSearchSpaceIsTrivial() const { return _autoPas.searchSpaceIsTrivial(); }

  /** Get the mean rebuild frequency observed by the node-local AutoPas instance. */
  [[nodiscard]] double getMeanLocalRebuildFrequency() { return _autoPas.getMeanRebuildFrequency(); }

  /** Get the Verlet skin configured for the node-local AutoPas instance. */
  [[nodiscard]] auto getVerletSkin() { return _autoPas.getVerletSkin(); }

  /** Get the Verlet rebuild frequency configured for the node-local AutoPas instance. */
  [[nodiscard]] auto getVerletRebuildFrequency() { return _autoPas.getVerletRebuildFrequency(); }

  /**
   * Return the currently active node-local AutoPas configurations for diagnostics.
   *
   * This exposes tuning metadata without exposing the local AutoPas container itself.
   */
  [[nodiscard]] auto getCurrentLocalConfigurations() const { return _autoPas.getCurrentConfigs(); }

  /**
   * Get a local snapshot of the optional DistributedAutoPas performance counters.
   *
   * All values are zero when the library was built without DISTRIBUTED_AUTOPAS_ENABLE_TIMERS.
   */
  [[nodiscard]] performance::PerformanceCountersSnapshot getPerformanceCountersSnapshot() const {
    return _performanceCounters.snapshot();
  }

  /** Reset all locally accumulated DistributedAutoPas performance counter values. */
  void resetPerformanceCounters() { _performanceCounters.reset(); }

  /** Finalize the node-local AutoPas instance before the communication runtime is destroyed. */
  void finalize() { _autoPas.finalize(); }

 private:
  /**
   * Update the node-local container, migrate emigrants, and rebuild halo particles without changing the domain bounds.
   */
  void synchronizeParticles() {
    DAP_SCOPED_TIMER(_performanceCounters.synchronization);
    DAP_TIMER_START(_performanceCounters.updateContainer);
    auto emigrants = _autoPas.updateContainer();
    DAP_TIMER_STOP(_performanceCounters.updateContainer);
    migrateAndExchangeHalos(std::move(emigrants));
  }

  /**
   * Synchronize particles while applying externally proposed local ownership bounds.
   *
   * @param localMin Proposed lower corner of this rank's ownership box.
   * @param localMax Proposed upper corner of this rank's ownership box.
   */
  void synchronizeParticles(const std::array<double, 3> &localMin, const std::array<double, 3> &localMax) {
    DAP_SCOPED_TIMER(_performanceCounters.synchronization);
    DAP_TIMER_START(_performanceCounters.updateContainer);
    auto emigrants = _autoPas.updateContainer();
    DAP_TIMER_STOP(_performanceCounters.updateContainer);
    resizeDomainAndMigrate(std::move(emigrants), localMin, localMax);
  }

  /**
   * Synchronize particles and update the local domain through inverted-pressure load balancing.
   *
   * @param localWork Computational work measured on this rank.
   */
  void synchronizeParticlesWithLoadBalancing(double localWork) {
    DAP_SCOPED_TIMER(_performanceCounters.synchronization);
    DAP_TIMER_START(_performanceCounters.updateContainer);
    auto emigrants = _autoPas.updateContainer();
    DAP_TIMER_STOP(_performanceCounters.updateContainer);

    // There is no distributed geometry to balance with a single rank. Avoid creating
    // unnecessary plane communicators and keep the ordinary synchronization path.
    if (_runtime.size() == 1) {
      migrateAndExchangeHalos(std::move(emigrants));
      return;
    }

    DAP_TIMER_START(_performanceCounters.loadBalancing);
    if (not _invertedPressureLoadBalancer.has_value()) {
      _invertedPressureLoadBalancer.emplace(_runtime.communicator(), _domain);
    }

    const double minWidth = 2. * (_cutoff + _autoPas.getVerletSkin());
    const auto balancedBox = _invertedPressureLoadBalancer->balance(localWork, _domain, minWidth);
    DAP_TIMER_STOP(_performanceCounters.loadBalancing);
    resizeDomainAndMigrate(std::move(emigrants), balancedBox.min, balancedBox.max);
  }

  /**
   * Resize the complete adaptive decomposition and migrate all particles invalidated by the geometry change.
   *
   * @param emigrants Particles already removed by AutoPas::updateContainer() for the previous ownership box.
   * @param localMin Proposed lower corner of this rank's new ownership box.
   * @param localMax Proposed upper corner of this rank's new ownership box.
   */
  void resizeDomainAndMigrate(std::vector<Particle> emigrants, const std::array<double, 3> &localMin,
                              const std::array<double, 3> &localMax) {
    // Synchronize the complete Cartesian split positions before changing AutoPas.
    // targetRank() needs all process-plane boundaries once the grid is no longer uniform.
    auto resizedDomain = _domain;
    DAP_TIMER_START(_performanceCounters.boundarySynchronization);
    resizedDomain.setSubdomainBoundaries(collectSubdomainBoundaries(localMin, localMax));
    DAP_TIMER_STOP(_performanceCounters.boundarySynchronization);

    DAP_TIMER_START(_performanceCounters.resizeBox);
    auto additionalEmigrants = _autoPas.resizeBox(resizedDomain.localMin(), resizedDomain.localMax());
    DAP_TIMER_STOP(_performanceCounters.resizeBox);
    emigrants.reserve(emigrants.size() + additionalEmigrants.size());
    emigrants.insert(emigrants.end(), std::make_move_iterator(additionalEmigrants.begin()),
                     std::make_move_iterator(additionalEmigrants.end()));

    _domain = std::move(resizedDomain);
    migrateAndExchangeHalos(std::move(emigrants));
  }

  /**
   * Collect local ownership bounds from all ranks and reconstruct the global split positions in every dimension.
   *
   * Shared boundaries proposed by adjacent ranks must agree exactly.
   *
   * @param localMin Proposed lower corner of this rank's ownership box.
   * @param localMax Proposed upper corner of this rank's ownership box.
   */
  [[nodiscard]] std::array<std::vector<double>, 3> collectSubdomainBoundaries(
      const std::array<double, 3> &localMin, const std::array<double, 3> &localMax) const {
    const std::array<double, 6> localBounds{localMin[0], localMin[1], localMin[2],
                                            localMax[0], localMax[1], localMax[2]};
    std::vector<double> gatheredBounds(static_cast<std::size_t>(_domain.numRanks()) * localBounds.size());

    if (MPI_Allgather(localBounds.data(), static_cast<int>(localBounds.size()), MPI_DOUBLE, gatheredBounds.data(),
                      static_cast<int>(localBounds.size()), MPI_DOUBLE, _runtime.communicator()) != MPI_SUCCESS) {
      throw std::runtime_error("DistributedAutoPas: failed to synchronize adaptive subdomain boundaries.");
    }

    std::array<std::vector<double>, 3> boundaries;
    std::array<std::vector<bool>, 3> initialized;
    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      const auto numberOfBoundaries = static_cast<std::size_t>(_domain.processGrid()[dimension] + 1);
      boundaries[dimension].resize(numberOfBoundaries);
      initialized[dimension].assign(numberOfBoundaries, false);
    }

    const auto setBoundary = [&](std::size_t dimension, std::size_t boundaryIndex, double value) {
      if (initialized[dimension][boundaryIndex] and boundaries[dimension][boundaryIndex] != value) {
        throw std::runtime_error(
            "DistributedAutoPas: neighboring ranks proposed inconsistent adaptive subdomain boundaries.");
      }
      boundaries[dimension][boundaryIndex] = value;
      initialized[dimension][boundaryIndex] = true;
    };

    for (int rank = 0; rank < _domain.numRanks(); ++rank) {
      const auto coordinates = _domain.rankToCoordinates(rank);
      const auto offset = static_cast<std::size_t>(rank) * localBounds.size();

      for (std::size_t dimension = 0; dimension < 3; ++dimension) {
        const auto lowerBoundary = static_cast<std::size_t>(coordinates[dimension]);
        const auto upperBoundary = lowerBoundary + 1;
        setBoundary(dimension, lowerBoundary, gatheredBounds[offset + dimension]);
        setBoundary(dimension, upperBoundary, gatheredBounds[offset + 3 + dimension]);
      }
    }

    return boundaries;
  }

  /**
   * Complete particle migration and rebuild halo particles for the current domain decomposition.
   *
   * @param emigrants Particles that no longer belong to the local ownership box.
   */
  void migrateAndExchangeHalos(std::vector<Particle> emigrants) {
    DAP_TIMER_START(_performanceCounters.migration);
    auto immigrants = _particleMigration.migrate(emigrants, _domain);
    DAP_TIMER_STOP(_performanceCounters.migration);

    DAP_TIMER_START(_performanceCounters.addMigratedParticles);
    _autoPas.addParticles(immigrants);
    DAP_TIMER_STOP(_performanceCounters.addMigratedParticles);

    DAP_TIMER_START(_performanceCounters.ownedParticleCollection);
    std::vector<Particle> ownedParticles;
    ownedParticles.reserve(getLocalNumberOfOwnedParticles());

    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      ownedParticles.push_back(*iter);
    }
    DAP_TIMER_STOP(_performanceCounters.ownedParticleCollection);

    DAP_TIMER_START(_performanceCounters.haloExchange);
    auto haloParticles = _haloExchange.exchange(ownedParticles, _domain, _cutoff);
    DAP_TIMER_STOP(_performanceCounters.haloExchange);

    DAP_TIMER_START(_performanceCounters.addHaloParticles);
    _autoPas.addHaloParticles(haloParticles);
    DAP_TIMER_STOP(_performanceCounters.addHaloParticles);
  }

  Runtime &_runtime;
  DomainDecomposition _domain;
  [[no_unique_address]] performance::PerformanceCounters _performanceCounters;
  ParticleMigration<Particle, Serializer> _particleMigration;
  HaloExchange<Particle, Serializer> _haloExchange;
  std::optional<load_balancing::InvertedPressureLoadBalancer> _invertedPressureLoadBalancer;
  autopas::AutoPas<Particle> _autoPas;
  double _cutoff;
};

}  // namespace dap
