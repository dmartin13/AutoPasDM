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
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class DistributedAutoPas {
 public:
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff)
      : DistributedAutoPas(
            runtime, globalMin, globalMax, cutoff, std::array<bool, 3>{true, false, false},
            std::array<BoundaryType, 3>{BoundaryType::periodic, BoundaryType::none, BoundaryType::none}) {}

  /** Construct a distributed container with a Cartesian process grid. */
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff, const std::array<bool, 3> &subdivideDimensions)
      : DistributedAutoPas(
            runtime, globalMin, globalMax, cutoff, subdivideDimensions,
            std::array<BoundaryType, 3>{BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic}) {}

  /** Construct a distributed container with explicit per-dimension boundary conditions. */
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff, const std::array<bool, 3> &subdivideDimensions,
                     const std::array<BoundaryType, 3> &boundaryTypes)
      : DistributedAutoPas(runtime, globalMin, globalMax, cutoff, subdivideDimensions, boundaryTypes, [](auto &) {}) {}

  /**
   * Construct and configure the node-local AutoPas instance before initialization.
   * This overload keeps the legacy x-periodic decomposition for backwards compatibility.
   */
  template <class Configurator>
  requires std::invocable < Configurator, autopas::AutoPas<Particle>
  & > DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin,
                         const std::array<double, 3> &globalMax, double cutoff, Configurator &&configurator)
      : DistributedAutoPas(runtime, globalMin, globalMax, cutoff, std::array<bool, 3>{true, false, false},
                           std::array<BoundaryType, 3>{BoundaryType::periodic, BoundaryType::none, BoundaryType::none},
                           std::forward<Configurator>(configurator)) {}

  /** Construct and configure a distributed container with a Cartesian process grid. */
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
   * applied by DistributedAutoPas.
   */
  template <class Configurator>
  requires std::invocable < Configurator, autopas::AutoPas<Particle>
  & > DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin,
                         const std::array<double, 3> &globalMax, double cutoff,
                         const std::array<bool, 3> &subdivideDimensions,
                         const std::array<BoundaryType, 3> &boundaryTypes, Configurator &&configurator)
      : _runtime(runtime),
  _domain(runtime.rank(), runtime.size(), globalMin, globalMax, subdivideDimensions, boundaryTypes),
  _particleMigration(runtime.communicator()), _haloExchange(runtime.communicator()), _cutoff(cutoff) {
    _autoPas.setBoxMin(_domain.localMin());
    _autoPas.setBoxMax(_domain.localMax());
    _autoPas.setCutoff(_cutoff);

    std::forward<Configurator>(configurator)(_autoPas);

    _autoPas.init();
  }

  void addParticle(const Particle &particle) { _autoPas.addParticle(particle); }

  /**
   * Redistribute application-provided particles according to the current domain decomposition
   * and add the particles owned by this rank to the local AutoPas container.
   *
   * Every rank contributes its local input collection and must call this collective operation.
   * Particles are sent directly to their owning rank, so the application does not need a particle
   * communicator or access to the distributed communication context. This operation is intended
   * for initialization and checkpoint loading, not for timestep migration.
   *
   * Particles outside the global simulation box are ignored. The application can detect such
   * input through its global particle-count sanity check.
   */
  /**
   * Add a particle collection that conceptually originates on one root rank.
   *
   * All ranks must call this collective operation with the same root rank. Only the
   * collection passed on rootRank is considered. Collections present on the other
   * ranks are ignored. This is useful for replicated application input such as
   * deterministic particles generated from a shared configuration file.
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
   * after communication and before the ordinary particle interactions.
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
   * computeInteractionsPrepared().
   */
  template <class Functor>
  bool computeInteractions(Functor *functor) {
    prepareInteractions();
    return computeInteractionsPrepared(functor);
  }

  [[nodiscard]] std::size_t getLocalNumberOfOwnedParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::owned);
  }

  [[nodiscard]] std::size_t getLocalNumberOfHaloParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::halo);
  }

  [[nodiscard]] std::size_t getLocalNumberOfParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::ownedOrHalo);
  }

  [[nodiscard]] std::size_t getGlobalNumberOfOwnedParticles() const {
    return globalSum(getLocalNumberOfOwnedParticles());
  }

  [[nodiscard]] int rank() const { return _runtime.rank(); }
  [[nodiscard]] int numberOfRanks() const { return _runtime.size(); }
  [[nodiscard]] bool isRoot() const { return _runtime.isRoot(); }

  void barrier() const { _runtime.barrier(); }

  template <class T>
  [[nodiscard]] T globalSum(T localValue) const {
    return _runtime.globalSum(localValue);
  }

  [[nodiscard]] const std::array<double, 3> &localBoxMin() const { return _domain.localMin(); }
  [[nodiscard]] const std::array<double, 3> &localBoxMax() const { return _domain.localMax(); }
  [[nodiscard]] const std::array<double, 3> &globalBoxMin() const { return _domain.globalMin(); }
  [[nodiscard]] const std::array<double, 3> &globalBoxMax() const { return _domain.globalMax(); }
  [[nodiscard]] const std::array<BoundaryType, 3> &boundaryTypes() const { return _domain.boundaryTypes(); }

  // Maybe this can be removed/restructured later
  // Local AutoPas metadata that is still used by md-flexible for tuning statistics.
  // These methods deliberately expose values, not the local container itself.
  [[nodiscard]] bool localSearchSpaceIsTrivial() const { return _autoPas.searchSpaceIsTrivial(); }
  [[nodiscard]] double getMeanLocalRebuildFrequency() { return _autoPas.getMeanRebuildFrequency(); }
  [[nodiscard]] auto getVerletSkin() { return _autoPas.getVerletSkin(); }
  [[nodiscard]] auto getVerletRebuildFrequency() { return _autoPas.getVerletRebuildFrequency(); }

  /**
   * Return the currently active node-local AutoPas configurations for diagnostics.
   *
   * This exposes tuning metadata without exposing the local AutoPas container itself.
   */
  [[nodiscard]] auto getCurrentLocalConfigurations() const { return _autoPas.getCurrentConfigs(); }

  void finalize() { _autoPas.finalize(); }

 private:
  void synchronizeParticles() {
    auto emigrants = _autoPas.updateContainer();
    migrateAndExchangeHalos(std::move(emigrants));
  }

  void synchronizeParticles(const std::array<double, 3> &localMin, const std::array<double, 3> &localMax) {
    auto emigrants = _autoPas.updateContainer();
    resizeDomainAndMigrate(std::move(emigrants), localMin, localMax);
  }

  void synchronizeParticlesWithLoadBalancing(double localWork) {
    auto emigrants = _autoPas.updateContainer();

    // There is no distributed geometry to balance with a single rank. Avoid creating
    // unnecessary plane communicators and keep the ordinary synchronization path.
    if (_runtime.size() == 1) {
      migrateAndExchangeHalos(std::move(emigrants));
      return;
    }

    if (not _invertedPressureLoadBalancer.has_value()) {
      _invertedPressureLoadBalancer.emplace(_runtime.communicator(), _domain);
    }

    const double minWidth = 2. * (_cutoff + _autoPas.getVerletSkin());
    const auto balancedBox = _invertedPressureLoadBalancer->balance(localWork, _domain, minWidth);
    resizeDomainAndMigrate(std::move(emigrants), balancedBox.min, balancedBox.max);
  }

  void resizeDomainAndMigrate(std::vector<Particle> emigrants, const std::array<double, 3> &localMin,
                              const std::array<double, 3> &localMax) {
    // Synchronize the complete Cartesian split positions before changing AutoPas.
    // targetRank() needs all process-plane boundaries once the grid is no longer uniform.
    auto resizedDomain = _domain;
    resizedDomain.setSubdomainBoundaries(collectSubdomainBoundaries(localMin, localMax));

    auto additionalEmigrants = _autoPas.resizeBox(resizedDomain.localMin(), resizedDomain.localMax());
    emigrants.reserve(emigrants.size() + additionalEmigrants.size());
    emigrants.insert(emigrants.end(), std::make_move_iterator(additionalEmigrants.begin()),
                     std::make_move_iterator(additionalEmigrants.end()));

    _domain = std::move(resizedDomain);
    migrateAndExchangeHalos(std::move(emigrants));
  }

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

  void migrateAndExchangeHalos(std::vector<Particle> emigrants) {
    auto immigrants = _particleMigration.migrate(emigrants, _domain);
    _autoPas.addParticles(immigrants);

    std::vector<Particle> ownedParticles;
    ownedParticles.reserve(getLocalNumberOfOwnedParticles());

    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      ownedParticles.push_back(*iter);
    }

    auto haloParticles = _haloExchange.exchange(ownedParticles, _domain, _cutoff);
    _autoPas.addHaloParticles(haloParticles);
  }

  Runtime &_runtime;
  DomainDecomposition _domain;
  ParticleMigration<Particle, Serializer> _particleMigration;
  HaloExchange<Particle, Serializer> _haloExchange;
  std::optional<load_balancing::InvertedPressureLoadBalancer> _invertedPressureLoadBalancer;
  autopas::AutoPas<Particle> _autoPas;
  double _cutoff;
};

}  // namespace dap
