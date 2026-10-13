#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

#include "autopasdm/DomainDecomposition.h"

namespace {

constexpr std::array<double, 3> globalMin{0., 1., 2.};
constexpr std::array<double, 3> globalMax{10., 11., 12.};

TEST(DomainDecompositionTest, SplitsGlobalDomainAlongX) {
  constexpr int numRanks = 4;
  constexpr double subdomainLength = 2.5;

  for (int rank = 0; rank < numRanks; ++rank) {
    const autopasdm::DomainDecomposition domain(rank, numRanks, globalMin, globalMax);

    EXPECT_EQ(domain.rank(), rank);
    EXPECT_EQ(domain.numRanks(), numRanks);
    EXPECT_EQ(domain.processGrid(), (std::array<int, 3>{4, 1, 1}));
    EXPECT_EQ(domain.coordinates(), (std::array<int, 3>{rank, 0, 0}));
    EXPECT_EQ(domain.globalMin(), globalMin);
    EXPECT_EQ(domain.globalMax(), globalMax);

    EXPECT_DOUBLE_EQ(domain.localMin()[0], rank * subdomainLength);
    EXPECT_DOUBLE_EQ(domain.localMax()[0], (rank + 1) * subdomainLength);
    EXPECT_DOUBLE_EQ(domain.localMin()[1], globalMin[1]);
    EXPECT_DOUBLE_EQ(domain.localMax()[1], globalMax[1]);
    EXPECT_DOUBLE_EQ(domain.localMin()[2], globalMin[2]);
    EXPECT_DOUBLE_EQ(domain.localMax()[2], globalMax[2]);
  }
}

TEST(DomainDecompositionTest, GeneratesProcessGridFromSubdivisionMask) {
  EXPECT_EQ(autopasdm::DomainDecomposition::generateProcessGrid(4, {true, false, false}),
            (std::array<int, 3>{4, 1, 1}));
  EXPECT_EQ(autopasdm::DomainDecomposition::generateProcessGrid(4, {true, true, false}), (std::array<int, 3>{2, 2, 1}));
  EXPECT_EQ(autopasdm::DomainDecomposition::generateProcessGrid(8, {true, true, true}), (std::array<int, 3>{2, 2, 2}));
  EXPECT_EQ(autopasdm::DomainDecomposition::generateProcessGrid(12, {true, true, true}), (std::array<int, 3>{2, 2, 3}));
  EXPECT_EQ(autopasdm::DomainDecomposition::generateProcessGrid(6, {false, true, true}), (std::array<int, 3>{1, 2, 3}));
}

TEST(DomainDecompositionTest, RejectsInvalidRankAndGridInformation) {
  EXPECT_THROW((autopasdm::DomainDecomposition{0, 0, globalMin, globalMax}), std::invalid_argument);
  EXPECT_THROW((autopasdm::DomainDecomposition{-1, 4, globalMin, globalMax}), std::invalid_argument);
  EXPECT_THROW((autopasdm::DomainDecomposition{4, 4, globalMin, globalMax}), std::invalid_argument);

  EXPECT_THROW((autopasdm::DomainDecomposition{0, 4, globalMin, globalMax, std::array<int, 3>{2, 1, 1}}),
               std::invalid_argument);
  EXPECT_THROW((autopasdm::DomainDecomposition{0, 4, globalMin, globalMax, std::array<int, 3>{2, 0, 2}}),
               std::invalid_argument);
  EXPECT_THROW((void)autopasdm::DomainDecomposition::generateProcessGrid(4, {false, false, false}),
               std::invalid_argument);
}

TEST(DomainDecompositionTest, SplitsGlobalDomainInThreeDimensions) {
  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{8., 6., 4.};
  constexpr std::array<int, 3> processGrid{2, 3, 2};

  const autopasdm::DomainDecomposition domain(9, 12, boxMin, boxMax, processGrid);

  EXPECT_EQ(domain.processGrid(), processGrid);
  EXPECT_EQ(domain.coordinates(), (std::array<int, 3>{1, 1, 1}));
  EXPECT_EQ(domain.localMin(), (std::array<double, 3>{4., 2., 2.}));
  EXPECT_EQ(domain.localMax(), (std::array<double, 3>{8., 4., 4.}));
}

TEST(DomainDecompositionTest, ConvertsBetweenRanksAndGridCoordinates) {
  constexpr std::array<int, 3> processGrid{2, 3, 2};
  const autopasdm::DomainDecomposition domain(0, 12, globalMin, globalMax, processGrid);

  for (int rank = 0; rank < 12; ++rank) {
    EXPECT_EQ(domain.coordinatesToRank(domain.rankToCoordinates(rank)), rank);
  }

  EXPECT_EQ(domain.rankToCoordinates(0), (std::array<int, 3>{0, 0, 0}));
  EXPECT_EQ(domain.rankToCoordinates(9), (std::array<int, 3>{1, 1, 1}));
  EXPECT_EQ(domain.rankToCoordinates(11), (std::array<int, 3>{1, 2, 1}));
  EXPECT_EQ(domain.coordinatesToRank({1, 2, 1}), 11);
}

TEST(DomainDecompositionTest, DeterminesTargetRankInThreeDimensions) {
  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{8., 6., 4.};
  constexpr std::array<int, 3> processGrid{2, 3, 2};
  const autopasdm::DomainDecomposition domain(0, 12, boxMin, boxMax, processGrid);

  EXPECT_EQ(domain.targetRank({0.1, 0.1, 0.1}), 0);
  EXPECT_EQ(domain.targetRank({3.999, 1.999, 1.999}), 0);
  EXPECT_EQ(domain.targetRank({4., 2., 2.}), 9);
  EXPECT_EQ(domain.targetRank({7.999, 5.999, 3.999}), 11);
}

TEST(DomainDecompositionTest, DeterminesTargetRankFromAdaptiveSubdomainBoundaries) {
  autopasdm::DomainDecomposition domain(2, 4, globalMin, globalMax);

  domain.setSubdomainBoundaries({
      std::vector<double>{0., 1.25, 4., 8.5, 10.},
      std::vector<double>{1., 11.},
      std::vector<double>{2., 12.},
  });

  EXPECT_EQ(domain.localMin(), (std::array<double, 3>{4., 1., 2.}));
  EXPECT_EQ(domain.localMax(), (std::array<double, 3>{8.5, 11., 12.}));

  EXPECT_EQ(domain.targetRank({0.5, 5., 5.}), 0);
  EXPECT_EQ(domain.targetRank({1.25, 5., 5.}), 1);
  EXPECT_EQ(domain.targetRank({3.999, 5., 5.}), 1);
  EXPECT_EQ(domain.targetRank({4., 5., 5.}), 2);
  EXPECT_EQ(domain.targetRank({8.5, 5., 5.}), 3);
}

TEST(DomainDecompositionTest, DeterminesTargetRankFromAdaptiveThreeDimensionalBoundaries) {
  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{8., 6., 4.};
  constexpr std::array<int, 3> processGrid{2, 3, 2};
  autopasdm::DomainDecomposition domain(0, 12, boxMin, boxMax, processGrid);

  domain.setSubdomainBoundaries({
      std::vector<double>{0., 1., 8.},
      std::vector<double>{0., 1., 5., 6.},
      std::vector<double>{0., 3.5, 4.},
  });

  EXPECT_EQ(domain.targetRank({0.5, 4., 3.75}), 3);
  EXPECT_EQ(domain.targetRank({1., 5., 3.}), 10);
}

TEST(DomainDecompositionTest, RejectsInvalidSubdomainBoundaries) {
  autopasdm::DomainDecomposition domain(1, 4, globalMin, globalMax);

  EXPECT_THROW(domain.setSubdomainBoundaries({
                   std::vector<double>{0., 2., 10.},
                   std::vector<double>{1., 11.},
                   std::vector<double>{2., 12.},
               }),
               std::invalid_argument);
  EXPECT_THROW(domain.setSubdomainBoundaries({
                   std::vector<double>{0., 2.5, 2.5, 7.5, 10.},
                   std::vector<double>{1., 11.},
                   std::vector<double>{2., 12.},
               }),
               std::invalid_argument);
  EXPECT_THROW(domain.setSubdomainBoundaries({
                   std::vector<double>{0.1, 2.5, 5., 7.5, 10.},
                   std::vector<double>{1., 11.},
                   std::vector<double>{2., 12.},
               }),
               std::invalid_argument);
}

TEST(DomainDecompositionTest, FindsPeriodicFaceEdgeAndCornerNeighbors) {
  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{3., 3., 3.};
  constexpr std::array<int, 3> processGrid{3, 3, 3};

  const autopasdm::DomainDecomposition center(13, 27, boxMin, boxMax, processGrid);
  EXPECT_EQ(center.coordinates(), (std::array<int, 3>{1, 1, 1}));
  EXPECT_EQ(center.neighborRank({1, 0, 0}), 22);
  EXPECT_EQ(center.neighborRank({1, -1, 0}), 19);
  EXPECT_EQ(center.neighborRank({-1, 1, 1}), 8);

  const autopasdm::DomainDecomposition corner(0, 27, boxMin, boxMax, processGrid);
  EXPECT_EQ(corner.neighborRank({-1, -1, -1}), 26);
  EXPECT_EQ(corner.precedingNeighbor(0), 18);
  EXPECT_EQ(corner.succeedingNeighbor(1), 3);
  EXPECT_EQ(corner.succeedingNeighbor(2), 1);
}

TEST(DomainDecompositionTest, UsesPeriodicNeighbors) {
  const autopasdm::DomainDecomposition firstRank(0, 4, globalMin, globalMax);
  EXPECT_EQ(firstRank.leftNeighbor(), 3);
  EXPECT_EQ(firstRank.rightNeighbor(), 1);

  const autopasdm::DomainDecomposition middleRank(2, 4, globalMin, globalMax);
  EXPECT_EQ(middleRank.leftNeighbor(), 1);
  EXPECT_EQ(middleRank.rightNeighbor(), 3);

  const autopasdm::DomainDecomposition lastRank(3, 4, globalMin, globalMax);
  EXPECT_EQ(lastRank.leftNeighbor(), 2);
  EXPECT_EQ(lastRank.rightNeighbor(), 0);
}

TEST(DomainDecompositionTest, LocalDomainUsesHalfOpenBounds) {
  const autopasdm::DomainDecomposition domain(1, 4, globalMin, globalMax);

  EXPECT_TRUE(domain.isInsideLocalDomain({2.5, 1., 2.}));
  EXPECT_TRUE(domain.isInsideLocalDomain({4.999, 10.999, 11.999}));

  EXPECT_FALSE(domain.isInsideLocalDomain({2.499, 5., 5.}));
  EXPECT_FALSE(domain.isInsideLocalDomain({5., 5., 5.}));
  EXPECT_FALSE(domain.isInsideLocalDomain({3., 11., 5.}));
  EXPECT_FALSE(domain.isInsideLocalDomain({3., 5., 12.}));
}

TEST(DomainDecompositionTest, DetectsHaloRegionOutsideLocalDomain) {
  const autopasdm::DomainDecomposition domain(1, 4, globalMin, globalMax);
  constexpr double haloWidth = 0.5;

  EXPECT_TRUE(domain.isInsideHaloRegion({2.25, 5., 5.}, haloWidth));
  EXPECT_TRUE(domain.isInsideHaloRegion({5.25, 5., 5.}, haloWidth));

  EXPECT_FALSE(domain.isInsideHaloRegion({2.5, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({4.999, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({1.999, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({5.5, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({2.25, 0.999, 5.}, haloWidth));
}

TEST(DomainDecompositionTest, DetectsFaceEdgeAndCornerHaloRegionsInThreeDimensions) {
  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{4., 4., 4.};
  constexpr std::array<int, 3> processGrid{2, 2, 2};
  const autopasdm::DomainDecomposition domain(0, 8, boxMin, boxMax, processGrid);
  constexpr double haloWidth = 0.25;

  EXPECT_TRUE(domain.isInsideHaloRegion({2.1, 1., 1.}, haloWidth));
  EXPECT_TRUE(domain.isInsideHaloRegion({2.1, 2.1, 1.}, haloWidth));
  EXPECT_TRUE(domain.isInsideHaloRegion({2.1, 2.1, 2.1}, haloWidth));

  EXPECT_FALSE(domain.isInsideHaloRegion({1.9, 1.9, 1.9}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({2.3, 1., 1.}, haloWidth));
}

TEST(DomainDecompositionTest, DeterminesTargetRankFromXPosition) {
  const autopasdm::DomainDecomposition domain(0, 4, globalMin, globalMax);

  EXPECT_EQ(domain.targetRank({0., 5., 5.}), 0);
  EXPECT_EQ(domain.targetRank({2.499, 5., 5.}), 0);
  EXPECT_EQ(domain.targetRank({2.5, 5., 5.}), 1);
  EXPECT_EQ(domain.targetRank({5., 5., 5.}), 2);
  EXPECT_EQ(domain.targetRank({7.5, 5., 5.}), 3);
  EXPECT_EQ(domain.targetRank({9.999, 5., 5.}), 3);
}

TEST(DomainDecompositionTest, AppliesPeriodicBoundaryInXOnly) {
  const autopasdm::DomainDecomposition domain(0, 4, globalMin, globalMax);

  std::array<double, 3> below{-20.25, 4., 7.};
  domain.applyPeriodicBoundary(below);
  EXPECT_DOUBLE_EQ(below[0], 9.75);
  EXPECT_DOUBLE_EQ(below[1], 4.);
  EXPECT_DOUBLE_EQ(below[2], 7.);

  std::array<double, 3> above{30.25, 4., 7.};
  domain.applyPeriodicBoundary(above);
  EXPECT_DOUBLE_EQ(above[0], 0.25);
  EXPECT_DOUBLE_EQ(above[1], 4.);
  EXPECT_DOUBLE_EQ(above[2], 7.);

  std::array<double, 3> upperBoundary{10., 4., 7.};
  domain.applyPeriodicBoundary(upperBoundary);
  EXPECT_DOUBLE_EQ(upperBoundary[0], 0.);
}

TEST(DomainDecompositionTest, UsesNoneBoundaryWithoutPeriodicWraparoundNeighbor) {
  constexpr std::array<autopasdm::BoundaryType, 3> boundaries{
      autopasdm::BoundaryType::periodic, autopasdm::BoundaryType::none, autopasdm::BoundaryType::none};
  const autopasdm::DomainDecomposition corner(0, 8, {0., 0., 0.}, {2., 2., 2.}, std::array<int, 3>{2, 2, 2},
                                              boundaries);

  EXPECT_EQ(corner.precedingNeighbor(0), 4);
  EXPECT_EQ(corner.precedingNeighbor(1), autopasdm::DomainDecomposition::noNeighbor);
  EXPECT_EQ(corner.precedingNeighbor(2), autopasdm::DomainDecomposition::noNeighbor);
  EXPECT_EQ(corner.succeedingNeighbor(1), 2);
  EXPECT_EQ(corner.succeedingNeighbor(2), 1);
  EXPECT_EQ(corner.neighborRank({-1, -1, 0}), autopasdm::DomainDecomposition::noNeighbor);
}

TEST(DomainDecompositionTest, WrapsOnlyPeriodicDimensions) {
  constexpr std::array<autopasdm::BoundaryType, 3> boundaries{
      autopasdm::BoundaryType::periodic, autopasdm::BoundaryType::none, autopasdm::BoundaryType::periodic};
  const autopasdm::DomainDecomposition domain(0, 1, {0., 0., 0.}, {2., 2., 2.}, std::array<int, 3>{1, 1, 1},
                                              boundaries);

  std::array<double, 3> position{-0.25, -0.25, 2.25};
  domain.applyPeriodicBoundary(position);

  EXPECT_DOUBLE_EQ(position[0], 1.75);
  EXPECT_DOUBLE_EQ(position[1], -0.25);
  EXPECT_DOUBLE_EQ(position[2], 0.25);
}

TEST(DomainDecompositionTest, TreatsReflectiveBoundaryAsNonPeriodicTopology) {
  constexpr std::array<autopasdm::BoundaryType, 3> boundaries{
      autopasdm::BoundaryType::periodic, autopasdm::BoundaryType::reflective, autopasdm::BoundaryType::none};
  const autopasdm::DomainDecomposition domain(0, 8, {0., 0., 0.}, {2., 2., 2.}, std::array<int, 3>{2, 2, 2},
                                              boundaries);

  EXPECT_EQ(domain.precedingNeighbor(0), 4);
  EXPECT_EQ(domain.precedingNeighbor(1), autopasdm::DomainDecomposition::noNeighbor);
  EXPECT_EQ(domain.succeedingNeighbor(1), 2);
  EXPECT_EQ(domain.boundaryType(1), autopasdm::BoundaryType::reflective);

  std::array<double, 3> position{0.5, -0.25, 0.5};
  domain.applyPeriodicBoundary(position);
  EXPECT_DOUBLE_EQ(position[1], -0.25);
}

}  // namespace
