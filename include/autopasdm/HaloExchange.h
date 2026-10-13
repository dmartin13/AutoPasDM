#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "autopasdm/Communication.h"
#include "autopasdm/DomainDecomposition.h"
#include "autopasdm/ParticleSerialization.h"
#include "autopasdm/PerformanceCounters.h"

namespace autopasdm {

/**
 * Exchange halo particles for a Cartesian process grid.
 *
 * The exchange is staged dimension by dimension (x -> y -> z). Particles
 * received in an earlier dimension participate in later stages. This propagates
 * face halos to edge and corner neighbors without communicating directly with
 * all 26 possible neighbors.
 *
 * Periodic global boundaries create shifted periodic images. BoundaryType::none
 * and BoundaryType::reflective are non-periodic communication boundaries and
 * therefore create no halo across the global box boundary. Reflective wall
 * forces are intentionally handled outside the distributed communication layer.
 *
 * @tparam Particle Particle type exchanged between ranks.
 * @tparam Serializer Serialization policy used for the particle type.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class HaloExchange {
 public:
  /**
   * Construct a halo exchanger for one communication context.
   *
   * @param comm MPI communicator used for halo communication.
   */
  explicit HaloExchange(MPI_Comm comm) : _comm(comm) {}

#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
  /**
   * Construct a halo exchanger that records detailed performance counters.
   *
   * @param comm MPI communicator used for halo communication.
   * @param performanceCounters Performance counter sink receiving halo and communication measurements.
   */
  HaloExchange(MPI_Comm comm, performance::PerformanceCounters *performanceCounters)
      : _comm(comm), _performanceCounters(performanceCounters) {}
#endif

  /**
   * Build all halo particles required by this rank for the current ownership geometry.
   *
   * @param ownedParticles Particles currently owned by this rank.
   * @param domain Current Cartesian domain decomposition.
   * @param haloWidth Width of the halo region.
   */
  [[nodiscard]] std::vector<Particle> exchange(const std::vector<Particle> &ownedParticles,
                                               const DomainDecomposition &domain, double haloWidth) const {
    AUTOPASDM_DETAILED_TIMER_START(_performanceCounters, haloInitialCopy);
    std::vector<Particle> candidates = ownedParticles;
    AUTOPASDM_DETAILED_TIMER_STOP(_performanceCounters, haloInitialCopy);
    std::vector<Particle> haloParticles;

    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      const int precedingNeighbor = domain.precedingNeighbor(static_cast<int>(dimension));
      const int succeedingNeighbor = domain.succeedingNeighbor(static_cast<int>(dimension));

      if (precedingNeighbor == DomainDecomposition::noNeighbor and
          succeedingNeighbor == DomainDecomposition::noNeighbor) {
        continue;
      }

      std::vector<Particle> sendPreceding;
      std::vector<Particle> sendSucceeding;

      AUTOPASDM_DETAILED_TIMER_START(_performanceCounters, haloSelection);
      for (const auto &particle : candidates) {
        const auto &position = particle.getR();

        if (precedingNeighbor != DomainDecomposition::noNeighbor and
            position[dimension] < domain.localMin()[dimension] + haloWidth) {
          sendPreceding.push_back(particle);
        }

        if (succeedingNeighbor != DomainDecomposition::noNeighbor and
            position[dimension] >= domain.localMax()[dimension] - haloWidth) {
          sendSucceeding.push_back(particle);
        }
      }

      AUTOPASDM_DETAILED_TIMER_STOP(_performanceCounters, haloSelection);
      AUTOPASDM_DETAILED_ADD(_performanceCounters, haloParticlesSent, sendPreceding.size() + sendSucceeding.size());

      LeftRightExchange<Particle> exchange;

      if (domain.processGrid()[dimension] == 1) {
        // For a periodic dimension that is not decomposed, both neighbors are this rank.
        // Reproduce the same left/right exchange locally instead of serializing the
        // particles and sending a potentially large payload through MPI to ourselves.
        // sendSucceeding arrives from the preceding side and sendPreceding from the
        // succeeding side, matching beginLeftRightExchange() semantics.
        exchange.recvFromLeft = std::move(sendSucceeding);
        exchange.recvFromRight = std::move(sendPreceding);
      } else {
        auto exchangeRequest = beginLeftRightExchange<Particle, Serializer>(
            _comm, precedingNeighbor, succeedingNeighbor, sendPreceding, sendSucceeding,
            300 + static_cast<int>(dimension) * 10 AUTOPASDM_DETAILED_COUNTERS_ARGUMENT(_performanceCounters));
        exchange = finishLeftRightExchange(exchangeRequest AUTOPASDM_DETAILED_COUNTERS_ARGUMENT(_performanceCounters));
      }

      AUTOPASDM_DETAILED_ADD(_performanceCounters, haloParticlesReceived,
                             exchange.recvFromLeft.size() + exchange.recvFromRight.size());

      AUTOPASDM_DETAILED_TIMER_START(_performanceCounters, haloPeriodicShift);
      shiftPeriodicImages(exchange.recvFromLeft, domain, dimension, true);
      shiftPeriodicImages(exchange.recvFromRight, domain, dimension, false);
      AUTOPASDM_DETAILED_TIMER_STOP(_performanceCounters, haloPeriodicShift);

      AUTOPASDM_DETAILED_TIMER_START(_performanceCounters, haloBufferInsertion);
      candidates.reserve(candidates.size() + exchange.recvFromLeft.size() + exchange.recvFromRight.size());
      haloParticles.reserve(haloParticles.size() + exchange.recvFromLeft.size() + exchange.recvFromRight.size());

      haloParticles.insert(haloParticles.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
      haloParticles.insert(haloParticles.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());

      candidates.insert(candidates.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
      candidates.insert(candidates.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());
      AUTOPASDM_DETAILED_TIMER_STOP(_performanceCounters, haloBufferInsertion);
    }

    return haloParticles;
  }

 private:
  /**
   * Shift received particles to the periodic image adjacent to the local domain if an exchange crossed the global box.
   *
   * @param particles Received particles to shift in place.
   * @param domain Current Cartesian domain decomposition.
   * @param dimension Cartesian dimension of the current halo stage.
   * @param receivedFromPreceding Whether the particles were received from the preceding neighbor.
   */
  static void shiftPeriodicImages(std::vector<Particle> &particles, const DomainDecomposition &domain,
                                  std::size_t dimension, bool receivedFromPreceding) {
    if (domain.boundaryType(static_cast<int>(dimension)) != BoundaryType::periodic) {
      return;
    }

    const bool crossesPeriodicBoundary = receivedFromPreceding
                                             ? domain.coordinates()[dimension] == 0
                                             : domain.coordinates()[dimension] == domain.processGrid()[dimension] - 1;

    if (not crossesPeriodicBoundary) {
      return;
    }

    const double globalLength = domain.globalMax()[dimension] - domain.globalMin()[dimension];
    const double shift = receivedFromPreceding ? -globalLength : globalLength;

    for (auto &particle : particles) {
      auto position = particle.getR();
      position[dimension] += shift;
      particle.setR(position);
    }
  }

  MPI_Comm _comm{MPI_COMM_WORLD};
#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
  performance::PerformanceCounters *_performanceCounters{nullptr};
#endif
};

}  // namespace autopasdm
