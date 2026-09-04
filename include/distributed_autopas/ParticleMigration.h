#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/ParticleSerialization.h"
#include "distributed_autopas/PerformanceCounters.h"

namespace dap {

/**
 * Migrate particles between direct neighbors of a Cartesian process grid.
 *
 * Migration is performed dimension by dimension. Particles can move by at most
 * one subdomain in each dimension per timestep, while diagonal moves are routed
 * through multiple face-neighbor exchanges within the same migration call.
 *
 * Periodic boundaries wrap particles to the opposite side of the global box.
 * BoundaryType::none discards particles that leave the global box.
 * BoundaryType::reflective is non-periodic and expects the application-level
 * reflective force to keep particles inside the global box. Crossing a reflective
 * boundary is treated as an error instead of silently deleting or wrapping the particle.
 *
 * @tparam Particle Particle type migrated between ranks.
 * @tparam Serializer Serialization policy used for the particle type.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class ParticleMigration {
 public:
  /**
   * Construct a particle migrator for one communication context.
   *
   * @param comm MPI communicator used for migration.
   */
  explicit ParticleMigration(MPI_Comm comm) : _comm(comm) {}

#ifdef DAP_ENABLE_DETAILED_TIMERS
  /**
   * Construct a particle migrator that records detailed performance counters.
   *
   * @param comm MPI communicator used for migration.
   * @param performanceCounters Performance counter sink receiving migration and communication measurements.
   */
  ParticleMigration(MPI_Comm comm, performance::PerformanceCounters *performanceCounters)
      : _comm(comm), _performanceCounters(performanceCounters) {}
#endif

  /**
   * Migrate emigrants to their owning rank using staged direct-neighbor communication.
   *
   * @param emigrants Particles that left the previous local AutoPas ownership box.
   * @param domain Current Cartesian domain decomposition used to route the particles.
   */
  [[nodiscard]] std::vector<Particle> migrate(const std::vector<Particle> &emigrants,
                                              const DomainDecomposition &domain) const {
    auto remainingParticles = emigrants;
    bool crossedReflectiveBoundary = false;

    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      const auto boundaryType = domain.boundaryType(static_cast<int>(dimension));
      DAP_DETAILED_TIMER_START(_performanceCounters, migrationClassification);

      if (domain.processGrid()[dimension] == 1) {
        std::vector<Particle> particlesInsideBoundary;
        particlesInsideBoundary.reserve(remainingParticles.size());

        for (auto particle : remainingParticles) {
          auto position = particle.getR();
          const bool outsideGlobalBoundary = position[dimension] < domain.globalMin()[dimension] or
                                             position[dimension] >= domain.globalMax()[dimension];

          if (boundaryType == BoundaryType::periodic) {
            domain.applyPeriodicBoundary(position, static_cast<int>(dimension));
            particle.setR(position);
            particlesInsideBoundary.push_back(std::move(particle));
          } else if (not outsideGlobalBoundary) {
            particlesInsideBoundary.push_back(std::move(particle));
          } else if (boundaryType == BoundaryType::reflective) {
            crossedReflectiveBoundary = true;
          }
          // BoundaryType::none deliberately drops particles outside the global box.
        }

        remainingParticles = std::move(particlesInsideBoundary);
        DAP_DETAILED_TIMER_STOP(_performanceCounters, migrationClassification);
        continue;
      }

      std::vector<Particle> sendPreceding;
      std::vector<Particle> sendSucceeding;
      std::vector<Particle> stayForNextDimension;

      sendPreceding.reserve(remainingParticles.size() / 3);
      sendSucceeding.reserve(remainingParticles.size() / 3);
      stayForNextDimension.reserve(remainingParticles.size());

      for (auto particle : remainingParticles) {
        auto position = particle.getR();

        if (position[dimension] < domain.localMin()[dimension]) {
          const bool crossesGlobalBoundary = domain.coordinates()[dimension] == 0;

          if (crossesGlobalBoundary) {
            if (boundaryType == BoundaryType::periodic) {
              domain.applyPeriodicBoundary(position, static_cast<int>(dimension));
              particle.setR(position);
            } else if (boundaryType == BoundaryType::reflective) {
              crossedReflectiveBoundary = true;
              continue;
            } else {
              // BoundaryType::none deliberately drops particles outside the global box.
              continue;
            }
          }

          sendPreceding.push_back(std::move(particle));
        } else if (position[dimension] >= domain.localMax()[dimension]) {
          const bool crossesGlobalBoundary = domain.coordinates()[dimension] == domain.processGrid()[dimension] - 1;

          if (crossesGlobalBoundary) {
            if (boundaryType == BoundaryType::periodic) {
              domain.applyPeriodicBoundary(position, static_cast<int>(dimension));
              particle.setR(position);
            } else if (boundaryType == BoundaryType::reflective) {
              crossedReflectiveBoundary = true;
              continue;
            } else {
              // BoundaryType::none deliberately drops particles outside the global box.
              continue;
            }
          }

          sendSucceeding.push_back(std::move(particle));
        } else {
          stayForNextDimension.push_back(std::move(particle));
        }
      }

      DAP_DETAILED_TIMER_STOP(_performanceCounters, migrationClassification);
      DAP_DETAILED_ADD(_performanceCounters, migrationParticlesSent, sendPreceding.size() + sendSucceeding.size());

      auto exchangeRequest = beginLeftRightExchange<Particle, Serializer>(
          _comm, domain.precedingNeighbor(static_cast<int>(dimension)),
          domain.succeedingNeighbor(static_cast<int>(dimension)), sendPreceding, sendSucceeding,
          100 + static_cast<int>(dimension) * 10 DAP_DETAILED_COUNTERS_ARGUMENT(_performanceCounters));
      const auto exchange =
          finishLeftRightExchange(exchangeRequest DAP_DETAILED_COUNTERS_ARGUMENT(_performanceCounters));
      DAP_DETAILED_ADD(_performanceCounters, migrationParticlesReceived,
                       exchange.recvFromLeft.size() + exchange.recvFromRight.size());

      stayForNextDimension.reserve(stayForNextDimension.size() + exchange.recvFromLeft.size() +
                                   exchange.recvFromRight.size());
      stayForNextDimension.insert(stayForNextDimension.end(), exchange.recvFromLeft.begin(),
                                  exchange.recvFromLeft.end());
      stayForNextDimension.insert(stayForNextDimension.end(), exchange.recvFromRight.begin(),
                                  exchange.recvFromRight.end());

      remainingParticles = std::move(stayForNextDimension);
    }

    // Error checks happen only after all staged neighbor exchanges have completed.
    // This keeps error handling off the communication path without adding a global
    // collective to every timestep.
    if (crossedReflectiveBoundary) {
      throw std::runtime_error(
          "Particle crossed a reflective global boundary. The reflective force should keep particles inside the "
          "simulation box; consider decreasing the timestep.");
    }

    for (const auto &particle : remainingParticles) {
      if (not domain.isInsideLocalDomain(particle.getR())) {
        throw std::runtime_error("Particle moved more than one subdomain in one timestep.");
      }
    }

    return remainingParticles;
  }

 private:
  MPI_Comm _comm{MPI_COMM_WORLD};
#ifdef DAP_ENABLE_DETAILED_TIMERS
  performance::PerformanceCounters *_performanceCounters{nullptr};
#endif
};

}  // namespace dap
