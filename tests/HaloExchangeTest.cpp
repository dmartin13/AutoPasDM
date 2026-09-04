#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <vector>

#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/HaloExchange.h"
#include "distributed_autopas/Particle.h"

namespace {

using Particle = dap::Particle;

constexpr std::array<double, 3> globalMin{0., 0., 0.};
constexpr std::array<double, 3> globalMax{4., 1., 1.};
constexpr double haloWidth = 0.2;

Particle makeParticle(unsigned long id, double x) {
  Particle particle;
  particle.setID(id);
  particle.setR({x, 0.5, 0.5});
  return particle;
}

dap::DomainDecomposition makeDomain(int &rank, int &numberOfRanks) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
  return dap::DomainDecomposition(rank, numberOfRanks, globalMin, globalMax);
}

struct TestContext {
  int rank{0};
  int numberOfRanks{0};
  dap::DomainDecomposition domain;
  dap::HaloExchange<Particle> haloExchange;

  TestContext() : domain(makeDomain(rank, numberOfRanks)), haloExchange(MPI_COMM_WORLD) {}
};

TEST(HaloExchangeTest, ExchangesLeftAndRightBoundaryParticles) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  const auto leftParticleId = static_cast<unsigned long>(1000 + 10 * context.rank);
  const auto rightParticleId = leftParticleId + 1;

  const std::vector<Particle> ownedParticles{
      makeParticle(leftParticleId, context.domain.localMin()[0] + 0.1),
      makeParticle(rightParticleId, context.domain.localMax()[0] - 0.1),
  };

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  ASSERT_EQ(halos.size(), 2);

  const int leftSource = context.domain.leftNeighbor();
  const int rightSource = context.domain.rightNeighbor();

  EXPECT_EQ(halos[0].getID(), static_cast<unsigned long>(1000 + 10 * leftSource + 1));
  EXPECT_EQ(halos[1].getID(), static_cast<unsigned long>(1000 + 10 * rightSource));

  double expectedLeftX = static_cast<double>(leftSource) + 0.9;
  if (context.rank == 0) {
    expectedLeftX -= globalMax[0] - globalMin[0];
  }

  double expectedRightX = static_cast<double>(rightSource) + 0.1;
  if (context.rank == context.numberOfRanks - 1) {
    expectedRightX += globalMax[0] - globalMin[0];
  }

  EXPECT_NEAR(halos[0].getR()[0], expectedLeftX, 1e-12);
  EXPECT_NEAR(halos[1].getR()[0], expectedRightX, 1e-12);
}

TEST(HaloExchangeTest, IgnoresParticlesOutsideHaloWidth) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  const std::vector<Particle> ownedParticles{
      makeParticle(2000 + context.rank, context.domain.localMin()[0] + 0.5),
  };

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  EXPECT_TRUE(halos.empty());
}

TEST(HaloExchangeTest, ShiftsPeriodicLeftHaloBelowGlobalDomain) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  std::vector<Particle> ownedParticles;
  if (context.rank == context.numberOfRanks - 1) {
    ownedParticles.push_back(makeParticle(3000, globalMax[0] - 0.1));
  }

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  if (context.rank == 0) {
    ASSERT_EQ(halos.size(), 1);
    EXPECT_EQ(halos.front().getID(), 3000);
    EXPECT_NEAR(halos.front().getR()[0], globalMin[0] - 0.1, 1e-12);
    EXPECT_LT(halos.front().getR()[0], globalMin[0]);
  } else {
    EXPECT_TRUE(halos.empty());
  }
}

TEST(HaloExchangeTest, ShiftsPeriodicRightHaloAboveGlobalDomain) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  std::vector<Particle> ownedParticles;
  if (context.rank == 0) {
    ownedParticles.push_back(makeParticle(4000, globalMin[0] + 0.1));
  }

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  if (context.rank == context.numberOfRanks - 1) {
    ASSERT_EQ(halos.size(), 1);
    EXPECT_EQ(halos.front().getID(), 4000);
    EXPECT_NEAR(halos.front().getR()[0], globalMax[0] + 0.1, 1e-12);
    EXPECT_GT(halos.front().getR()[0], globalMax[0]);
  } else {
    EXPECT_TRUE(halos.empty());
  }
}

TEST(HaloExchangeTest, PropagatesFaceEdgeAndCornerHalosInThreeDimensions) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
  ASSERT_EQ(numberOfRanks, 8);

  const std::array<double, 3> min{0., 0., 0.};
  const std::array<double, 3> max{2., 2., 2.};
  const dap::DomainDecomposition domain(rank, numberOfRanks, min, max, std::array<int, 3>{2, 2, 2});
  const dap::HaloExchange<Particle> haloExchange(MPI_COMM_WORLD);

  std::vector<Particle> ownedParticles;
  if (rank == 0) {
    Particle particle;
    particle.setID(5000);
    particle.setR({0.9, 0.9, 0.9});
    ownedParticles.push_back(particle);
  }

  const auto halos = haloExchange.exchange(ownedParticles, domain, haloWidth);

  if (rank == 0) {
    EXPECT_TRUE(halos.empty());
    return;
  }

  ASSERT_EQ(halos.size(), 1);
  EXPECT_EQ(halos.front().getID(), 5000);
  EXPECT_EQ(halos.front().getR(), (std::array<double, 3>{0.9, 0.9, 0.9}));
}

TEST(HaloExchangeTest, ShiftsPeriodicCornerHaloAcrossAllDimensions) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
  ASSERT_EQ(numberOfRanks, 8);

  const std::array<double, 3> min{0., 0., 0.};
  const std::array<double, 3> max{2., 2., 2.};
  const dap::DomainDecomposition domain(rank, numberOfRanks, min, max, std::array<int, 3>{2, 2, 2});
  const dap::HaloExchange<Particle> haloExchange(MPI_COMM_WORLD);

  std::vector<Particle> ownedParticles;
  if (rank == 7) {
    Particle particle;
    particle.setID(6000);
    particle.setR({1.9, 1.9, 1.9});
    ownedParticles.push_back(particle);
  }

  const auto halos = haloExchange.exchange(ownedParticles, domain, haloWidth);

  if (rank == 0) {
    ASSERT_EQ(halos.size(), 1);
    EXPECT_EQ(halos.front().getID(), 6000);
    EXPECT_NEAR(halos.front().getR()[0], -0.1, 1e-12);
    EXPECT_NEAR(halos.front().getR()[1], -0.1, 1e-12);
    EXPECT_NEAR(halos.front().getR()[2], -0.1, 1e-12);
  }
}


TEST(HaloExchangeTest, CreatesPeriodicHalosLocallyForUndecomposedDimension) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
  ASSERT_EQ(numberOfRanks, 4);

  const std::array<double, 3> min{0., 0., 0.};
  const std::array<double, 3> max{4., 1., 1.};
  const dap::DomainDecomposition domain(rank, numberOfRanks, min, max, std::array<int, 3>{4, 1, 1});
  const dap::HaloExchange<Particle> haloExchange(MPI_COMM_WORLD);

  Particle lowerParticle;
  lowerParticle.setID(static_cast<unsigned long>(9000 + rank * 10));
  lowerParticle.setR({domain.localMin()[0] + 0.5, 0.1, 0.5});

  Particle upperParticle;
  upperParticle.setID(static_cast<unsigned long>(9001 + rank * 10));
  upperParticle.setR({domain.localMin()[0] + 0.5, 0.9, 0.5});

  const auto halos = haloExchange.exchange({lowerParticle, upperParticle}, domain, haloWidth);

  ASSERT_EQ(halos.size(), 2);
  EXPECT_EQ(halos[0].getID(), upperParticle.getID());
  EXPECT_NEAR(halos[0].getR()[1], -0.1, 1e-12);
  EXPECT_EQ(halos[1].getID(), lowerParticle.getID());
  EXPECT_NEAR(halos[1].getR()[1], 1.1, 1e-12);
}

TEST(HaloExchangeTest, NoneBoundarySuppressesGlobalHaloButKeepsInternalHalo) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
  ASSERT_EQ(numberOfRanks, 8);

  constexpr std::array<dap::BoundaryType, 3> boundaries{dap::BoundaryType::none, dap::BoundaryType::none,
                                                        dap::BoundaryType::none};
  const dap::DomainDecomposition domain(rank, numberOfRanks, {0., 0., 0.}, {2., 2., 2.}, std::array<int, 3>{2, 2, 2},
                                        boundaries);
  const dap::HaloExchange<Particle> haloExchange(MPI_COMM_WORLD);

  std::vector<Particle> ownedParticles;
  if (rank == 0) {
    ownedParticles.push_back(makeParticle(7000, 0.1));
    ownedParticles.push_back(makeParticle(7001, 0.9));
  }

  const auto halos = haloExchange.exchange(ownedParticles, domain, haloWidth);

  if (rank == 4) {
    ASSERT_EQ(halos.size(), 1);
    EXPECT_EQ(halos.front().getID(), 7001);
    EXPECT_NEAR(halos.front().getR()[0], 0.9, 1e-12);
  } else {
    EXPECT_TRUE(halos.empty());
  }
}

TEST(HaloExchangeTest, ReflectiveBoundarySuppressesGlobalHaloButKeepsInternalHalo) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
  ASSERT_EQ(numberOfRanks, 4);

  constexpr std::array<dap::BoundaryType, 3> boundaries{dap::BoundaryType::reflective, dap::BoundaryType::none,
                                                        dap::BoundaryType::none};
  const dap::DomainDecomposition domain(rank, numberOfRanks, {0., 0., 0.}, {4., 1., 1.}, std::array<int, 3>{4, 1, 1},
                                        boundaries);
  const dap::HaloExchange<Particle> haloExchange(MPI_COMM_WORLD);

  std::vector<Particle> ownedParticles;
  if (rank == 0) {
    ownedParticles.push_back(makeParticle(8000, 0.1));
    ownedParticles.push_back(makeParticle(8001, 0.9));
  }

  const auto halos = haloExchange.exchange(ownedParticles, domain, haloWidth);

  if (rank == 1) {
    ASSERT_EQ(halos.size(), 1);
    EXPECT_EQ(halos.front().getID(), 8001);
    EXPECT_NEAR(halos.front().getR()[0], 0.9, 1e-12);
  } else {
    EXPECT_TRUE(halos.empty());
  }
}

}  // namespace
