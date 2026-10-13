#include <gtest/gtest.h>
#include <mpi.h>

#include <array>

#include "autopasdm/DomainDecomposition.h"
#include "autopasdm/load_balancing/InvertedPressureLoadBalancer.h"

namespace {

TEST(InvertedPressureLoadBalancerTest, AveragesWorkAcrossPlanesAndMovesSharedBoundaryConsistently) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ASSERT_EQ(size, 8);

  autopasdm::DomainDecomposition domain(rank, size, {0., 0., 0.}, {10., 10., 10.},
                                        std::array<bool, 3>{true, true, true});
  autopasdm::load_balancing::InvertedPressureLoadBalancer loadBalancer(MPI_COMM_WORLD, domain);

  const auto coordinates = domain.coordinates();
  const double xPlaneBaseWork = coordinates[0] == 0 ? 4. : 2.;
  const double yzVariation = coordinates[1] == coordinates[2] ? 1. : -1.;
  const double localWork = xPlaneBaseWork + yzVariation;

  const auto balancedBox = loadBalancer.balance(localWork, domain, 1.);

  constexpr double expectedXBoundary = 25. / 6.;
  if (coordinates[0] == 0) {
    EXPECT_DOUBLE_EQ(balancedBox.min[0], 0.);
    EXPECT_NEAR(balancedBox.max[0], expectedXBoundary, 1e-12);
  } else {
    EXPECT_NEAR(balancedBox.min[0], expectedXBoundary, 1e-12);
    EXPECT_DOUBLE_EQ(balancedBox.max[0], 10.);
  }

  for (std::size_t dimension = 1; dimension < 3; ++dimension) {
    const double expectedMin = coordinates[dimension] == 0 ? 0. : 5.;
    const double expectedMax = coordinates[dimension] == 0 ? 5. : 10.;
    EXPECT_NEAR(balancedBox.min[dimension], expectedMin, 1e-12);
    EXPECT_NEAR(balancedBox.max[dimension], expectedMax, 1e-12);
  }
}

TEST(InvertedPressureLoadBalancerTest, EnforcesMinimumWidthBeforeApplyingDampedShift) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ASSERT_EQ(size, 2);

  autopasdm::DomainDecomposition domain(rank, size, {0., 0., 0.}, {10., 10., 10.},
                                        std::array<bool, 3>{true, false, false});
  autopasdm::load_balancing::InvertedPressureLoadBalancer loadBalancer(MPI_COMM_WORLD, domain);

  const double localWork = domain.coordinates()[0] == 0 ? 99. : 1.;
  const auto balancedBox = loadBalancer.balance(localWork, domain, 2.);

  if (domain.coordinates()[0] == 0) {
    EXPECT_DOUBLE_EQ(balancedBox.min[0], 0.);
    EXPECT_DOUBLE_EQ(balancedBox.max[0], 3.5);
  } else {
    EXPECT_DOUBLE_EQ(balancedBox.min[0], 3.5);
    EXPECT_DOUBLE_EQ(balancedBox.max[0], 10.);
  }
}

}  // namespace
