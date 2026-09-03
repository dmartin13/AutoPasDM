#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

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

  /**
   * Build all halo particles required by this rank for the current ownership geometry.
   *
   * @param ownedParticles Particles currently owned by this rank.
   * @param domain Current Cartesian domain decomposition.
   * @param haloWidth Width of the halo region.
   */
  [[nodiscard]] std::vector<Particle> exchange(const std::vector<Particle> &ownedParticles,
                                               const DomainDecomposition &domain, double haloWidth) const {
    std::vector<Particle> candidates = ownedParticles;
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

      auto exchangeRequest =
          beginLeftRightExchange<Particle, Serializer>(_comm, precedingNeighbor, succeedingNeighbor, sendPreceding,
                                                       sendSucceeding, 300 + static_cast<int>(dimension) * 10);
      auto exchange = finishLeftRightExchange(exchangeRequest);

      shiftPeriodicImages(exchange.recvFromLeft, domain, dimension, true);
      shiftPeriodicImages(exchange.recvFromRight, domain, dimension, false);

      candidates.reserve(candidates.size() + exchange.recvFromLeft.size() + exchange.recvFromRight.size());
      haloParticles.reserve(haloParticles.size() + exchange.recvFromLeft.size() + exchange.recvFromRight.size());

      haloParticles.insert(haloParticles.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
      haloParticles.insert(haloParticles.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());

      candidates.insert(candidates.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
      candidates.insert(candidates.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());
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
};

}  // namespace dap
