#pragma once

#include <array>
#include <vector>

#include "distributed_autopas/BoundaryType.h"

namespace dap {

/**
 * Cartesian regular-grid domain decomposition.
 *
 * The decomposition starts with equally sized subdomains. The local box can later
 * be resized by a load balancer while the Cartesian process topology stays fixed.
 * The decomposition itself is independent of the communication backend. Runtime
 * provides the process index and number of processes during construction.
 */
class DomainDecomposition {
 public:
  /** Sentinel used when no neighbor exists at a non-periodic global boundary. */
  static constexpr int noNeighbor = -1;

  /**
   * Construct the legacy one-dimensional decomposition along x.
   *
   * The x dimension is periodic, while y and z are not subdivided and use non-periodic boundaries.
   *
   * @param rank Rank of the local process.
   * @param numRanks Total number of processes.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   */
  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax);

  /**
   * Construct a Cartesian decomposition with automatically generated process-grid dimensions.
   *
   * All global boundaries are periodic.
   *
   * @param rank Rank of the local process.
   * @param numRanks Total number of processes.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param subdivideDimensions Selects which Cartesian dimensions may be subdivided.
   */
  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      const std::array<bool, 3> &subdivideDimensions);

  /**
   * Construct a Cartesian decomposition with an explicit process grid.
   *
   * All global boundaries are periodic.
   *
   * @param rank Rank of the local process.
   * @param numRanks Total number of processes.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param processGrid Number of process-grid cells in x, y, and z.
   */
  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      std::array<int, 3> processGrid);

  /**
   * Construct a Cartesian decomposition with automatically generated process-grid dimensions and explicit boundaries.
   *
   * @param rank Rank of the local process.
   * @param numRanks Total number of processes.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param subdivideDimensions Selects which Cartesian dimensions may be subdivided.
   * @param boundaryTypes Boundary condition applied at the global boundary of each dimension.
   */
  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      const std::array<bool, 3> &subdivideDimensions, const std::array<BoundaryType, 3> &boundaryTypes);

  /**
   * Construct a Cartesian decomposition with an explicit process grid and explicit boundaries.
   *
   * @param rank Rank of the local process.
   * @param numRanks Total number of processes.
   * @param globalMin Lower corner of the global simulation box.
   * @param globalMax Upper corner of the global simulation box.
   * @param processGrid Number of process-grid cells in x, y, and z.
   * @param boundaryTypes Boundary condition applied at the global boundary of each dimension.
   */
  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      std::array<int, 3> processGrid, const std::array<BoundaryType, 3> &boundaryTypes);

  /**
   * Generate a Cartesian process grid by factorizing the number of ranks over the selected dimensions.
   *
   * Prime factors of the rank count are distributed over the dimensions selected for subdivision.
   *
   * @param numRanks Total number of processes.
   * @param subdivideDimensions Selects which Cartesian dimensions may be subdivided.
   */
  [[nodiscard]] static std::array<int, 3> generateProcessGrid(int numRanks,
                                                              const std::array<bool, 3> &subdivideDimensions);

  /**
   * Wrap coordinates in every periodic dimension into the global simulation box.
   *
   * @param pos Position to wrap in place.
   */
  void applyPeriodicBoundary(std::array<double, 3> &pos) const;

  /**
   * Wrap one coordinate if the selected dimension is periodic.
   *
   * @param pos Position to wrap in place.
   * @param dimension Cartesian dimension to process.
   */
  void applyPeriodicBoundary(std::array<double, 3> &pos, int dimension) const;

  /**
   * Check whether a position belongs to this rank's ownership box.
   *
   * @param pos Position to test.
   */
  [[nodiscard]] bool isInsideLocalDomain(const std::array<double, 3> &pos) const;

  /**
   * Check whether a position lies in this rank's halo region.
   *
   * The halo is only extended towards dimensions in which a communication neighbor exists.
   *
   * @param pos Position to test.
   * @param haloWidth Width of the halo region around the local ownership box.
   */
  [[nodiscard]] bool isInsideHaloRegion(const std::array<double, 3> &pos, double haloWidth) const;

  /**
   * Determine the rank that owns a position according to the current adaptive split positions.
   *
   * @param pos Position inside or on the boundary of the global simulation box.
   */
  [[nodiscard]] int targetRank(const std::array<double, 3> &pos) const;

  /**
   * Replace the global split positions of the Cartesian process grid.
   *
   * For dimension d, boundaries[d] must contain processGrid[d] + 1 strictly
   * increasing entries spanning the complete global box. The local box is derived
   * from the entries matching this rank's Cartesian coordinates.
   *
   * @param boundaries Split positions for x, y, and z.
   */
  void setSubdomainBoundaries(std::array<std::vector<double>, 3> boundaries);

  /**
   * Convert a linear rank to Cartesian process-grid coordinates.
   *
   * @param rank Rank to convert.
   */
  [[nodiscard]] std::array<int, 3> rankToCoordinates(int rank) const;

  /**
   * Convert Cartesian process-grid coordinates to the corresponding linear rank.
   *
   * @param coordinates Process-grid coordinates to convert.
   */
  [[nodiscard]] int coordinatesToRank(const std::array<int, 3> &coordinates) const;

  /**
   * Return the neighbor rank for a Cartesian grid offset.
   *
   * Periodic dimensions wrap around the process grid. At a global boundary with
   * a non-periodic boundary type (`none` or `reflective`), no neighbor exists and
   * noNeighbor is returned.
   *
   * @param offset Cartesian offset relative to this rank. Each component must describe the requested neighboring cell.
   */
  [[nodiscard]] int neighborRank(const std::array<int, 3> &offset) const;

  /**
   * Get the direct neighbor preceding this rank in one Cartesian dimension.
   *
   * @param dimension Cartesian dimension.
   */
  [[nodiscard]] int precedingNeighbor(int dimension) const;

  /**
   * Get the direct neighbor succeeding this rank in one Cartesian dimension.
   *
   * @param dimension Cartesian dimension.
   */
  [[nodiscard]] int succeedingNeighbor(int dimension) const;

  /** Get the direct neighbor in negative x direction. */
  [[nodiscard]] int leftNeighbor() const;

  /** Get the direct neighbor in positive x direction. */
  [[nodiscard]] int rightNeighbor() const;

  /** Get the rank of this process. */
  [[nodiscard]] int rank() const;

  /** Get the total number of processes in the decomposition. */
  [[nodiscard]] int numRanks() const;

  /** Get the number of process-grid cells in x, y, and z. */
  [[nodiscard]] const std::array<int, 3> &processGrid() const;

  /** Get the Cartesian process-grid coordinates of this rank. */
  [[nodiscard]] const std::array<int, 3> &coordinates() const;

  /** Get the global boundary conditions in x, y, and z. */
  [[nodiscard]] const std::array<BoundaryType, 3> &boundaryTypes() const;

  /**
   * Get the boundary condition of one Cartesian dimension.
   *
   * @param dimension Cartesian dimension.
   */
  [[nodiscard]] BoundaryType boundaryType(int dimension) const;

  /** Get the lower corner of the global simulation box. */
  [[nodiscard]] const std::array<double, 3> &globalMin() const;

  /** Get the upper corner of the global simulation box. */
  [[nodiscard]] const std::array<double, 3> &globalMax() const;

  /** Get the lower corner of this rank's local ownership box. */
  [[nodiscard]] const std::array<double, 3> &localMin() const;

  /** Get the upper corner of this rank's local ownership box. */
  [[nodiscard]] const std::array<double, 3> &localMax() const;

 private:
  int _rank{0};
  int _numRanks{1};

  std::array<int, 3> _processGrid{1, 1, 1};
  std::array<int, 3> _coordinates{0, 0, 0};
  std::array<BoundaryType, 3> _boundaryTypes{BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic};

  std::array<double, 3> _globalMin{};
  std::array<double, 3> _globalMax{};

  std::array<double, 3> _localMin{};
  std::array<double, 3> _localMax{};

  std::array<std::vector<double>, 3> _subdomainBoundaries{};
};

}  // namespace dap
