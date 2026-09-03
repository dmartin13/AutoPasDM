#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <stdexcept>

#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/load_balancing/InvertedPressure.h"

namespace dap::load_balancing {

/**
 * Local ownership bounds proposed by a load balancer.
 */
struct LocalBox {
  /** Lower corner of the proposed local ownership box. */
  std::array<double, 3> min{};

  /** Upper corner of the proposed local ownership box. */
  std::array<double, 3> max{};
};

/**
 * Inverted-pressure load balancer for a fixed Cartesian process topology.
 *
 * The measured work of every rank is first averaged over the process-grid plane
 * perpendicular to the dimension currently being balanced. Adjacent planes then
 * exchange their average work and outer boundary. This ensures that all ranks in
 * one plane calculate the same shared boundary position.
 *
 * As in md-flexible, only half of the distance to the theoretically balanced
 * position is applied in one balancing step. This damps large geometry changes and
 * keeps the local boxes valid when both sides move in the same update.
 *
 * This class only proposes new local bounds. Applying them to DomainDecomposition
 * and AutoPas is deliberately handled by DistributedAutoPas.
 */
class InvertedPressureLoadBalancer {
 public:
  /**
   * Construct the load balancer and create one communicator for each process-grid plane.
   *
   * @param communicator MPI communicator containing all ranks of the domain decomposition.
   * @param domain Domain decomposition whose fixed process topology is balanced by this object.
   */
  InvertedPressureLoadBalancer(MPI_Comm communicator, const DomainDecomposition &domain)
      : _communicator(communicator), _processGrid(domain.processGrid()), _coordinates(domain.coordinates()) {
    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      const int color = _coordinates[dimension];
      const auto firstOtherDimension = (dimension + 1) % 3;
      const auto secondOtherDimension = (dimension + 2) % 3;
      const int key =
          _coordinates[firstOtherDimension] * _processGrid[secondOtherDimension] + _coordinates[secondOtherDimension];

      if (MPI_Comm_split(_communicator, color, key, &_planeCommunicators[dimension]) != MPI_SUCCESS) {
        throw std::runtime_error("DistributedAutoPas: failed to create process-plane communicator.");
      }
    }
  }

  InvertedPressureLoadBalancer(const InvertedPressureLoadBalancer &) = delete;
  InvertedPressureLoadBalancer &operator=(const InvertedPressureLoadBalancer &) = delete;
  InvertedPressureLoadBalancer(InvertedPressureLoadBalancer &&) = delete;
  InvertedPressureLoadBalancer &operator=(InvertedPressureLoadBalancer &&) = delete;

  /**
   * Free the process-plane communicators unless MPI has already been finalized.
   */
  ~InvertedPressureLoadBalancer() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
      return;
    }

    for (auto &communicator : _planeCommunicators) {
      if (communicator != MPI_COMM_NULL) {
        MPI_Comm_free(&communicator);
      }
    }
  }

  /**
   * Calculate updated local ownership bounds from one local work measurement.
   *
   * The input domain is not modified.
   *
   * @param localWork Work measured on this MPI rank.
   * @param domain Current domain decomposition.
   * @param minWidth Minimum allowed subdomain width in every balanced dimension.
   */
  [[nodiscard]] LocalBox balance(double localWork, const DomainDecomposition &domain, double minWidth) const {
    if (localWork < 0.) {
      throw std::invalid_argument("DistributedAutoPas: load-balancing work must not be negative.");
    }
    if (minWidth <= 0.) {
      throw std::invalid_argument("DistributedAutoPas: minimum load-balancing domain width must be positive.");
    }
    if (domain.processGrid() != _processGrid or domain.coordinates() != _coordinates) {
      throw std::invalid_argument(
          "DistributedAutoPas: load balancer must be used with the process topology it was created for.");
    }

    const auto oldLocalMin = domain.localMin();
    const auto oldLocalMax = domain.localMax();
    LocalBox balancedBox{oldLocalMin, oldLocalMax};

    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      if (_processGrid[dimension] == 1) {
        continue;
      }

      double planeWorkSum = 0.;
      if (MPI_Allreduce(&localWork, &planeWorkSum, 1, MPI_DOUBLE, MPI_SUM, _planeCommunicators[dimension]) !=
          MPI_SUCCESS) {
        throw std::runtime_error("DistributedAutoPas: failed to reduce work in process plane.");
      }

      int ranksInPlane = 0;
      MPI_Comm_size(_planeCommunicators[dimension], &ranksInPlane);
      const double planeWork = planeWorkSum / static_cast<double>(ranksInPlane);

      const bool hasPrecedingPlane = _coordinates[dimension] > 0;
      const bool hasSucceedingPlane = _coordinates[dimension] < _processGrid[dimension] - 1;

      std::array<double, 2> infoFromPreceding{};
      std::array<double, 2> infoFromSucceeding{};
      const std::array<double, 2> infoForPreceding{planeWork, oldLocalMax[dimension]};
      const std::array<double, 2> infoForSucceeding{planeWork, oldLocalMin[dimension]};

      std::array<MPI_Request, 4> requests{};
      int requestCount = 0;
      const int tag = 400 + static_cast<int>(dimension);

      if (hasPrecedingPlane) {
        MPI_Irecv(infoFromPreceding.data(), static_cast<int>(infoFromPreceding.size()), MPI_DOUBLE,
                  domain.precedingNeighbor(static_cast<int>(dimension)), tag, _communicator, &requests[requestCount++]);
      }
      if (hasSucceedingPlane) {
        MPI_Irecv(infoFromSucceeding.data(), static_cast<int>(infoFromSucceeding.size()), MPI_DOUBLE,
                  domain.succeedingNeighbor(static_cast<int>(dimension)), tag, _communicator,
                  &requests[requestCount++]);
      }
      if (hasPrecedingPlane) {
        MPI_Isend(infoForPreceding.data(), static_cast<int>(infoForPreceding.size()), MPI_DOUBLE,
                  domain.precedingNeighbor(static_cast<int>(dimension)), tag, _communicator, &requests[requestCount++]);
      }
      if (hasSucceedingPlane) {
        MPI_Isend(infoForSucceeding.data(), static_cast<int>(infoForSucceeding.size()), MPI_DOUBLE,
                  domain.succeedingNeighbor(static_cast<int>(dimension)), tag, _communicator,
                  &requests[requestCount++]);
      }

      if (requestCount > 0 and MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error("DistributedAutoPas: failed to exchange neighboring process-plane work.");
      }

      if (hasPrecedingPlane) {
        const double precedingPlaneWork = infoFromPreceding[0];
        const double precedingPlaneMin = infoFromPreceding[1];
        if (precedingPlaneWork + planeWork > 0.) {
          const double targetBoundary = balanceAdjacentDomains(precedingPlaneWork, planeWork, precedingPlaneMin,
                                                               oldLocalMax[dimension], minWidth);
          balancedBox.min[dimension] = oldLocalMin[dimension] + (targetBoundary - oldLocalMin[dimension]) / 2.;
        }
      }

      if (hasSucceedingPlane) {
        const double succeedingPlaneWork = infoFromSucceeding[0];
        const double succeedingPlaneMax = infoFromSucceeding[1];
        if (planeWork + succeedingPlaneWork > 0.) {
          const double targetBoundary = balanceAdjacentDomains(planeWork, succeedingPlaneWork, oldLocalMin[dimension],
                                                               succeedingPlaneMax, minWidth);
          balancedBox.max[dimension] = oldLocalMax[dimension] + (targetBoundary - oldLocalMax[dimension]) / 2.;
        }
      }
    }

    return balancedBox;
  }

 private:
  MPI_Comm _communicator{MPI_COMM_NULL};
  std::array<MPI_Comm, 3> _planeCommunicators{MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL};
  std::array<int, 3> _processGrid{1, 1, 1};
  std::array<int, 3> _coordinates{0, 0, 0};
};

}  // namespace dap::load_balancing
