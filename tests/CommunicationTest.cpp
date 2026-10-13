#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "autopasdm/Communication.h"
#include "autopasdm/Particle.h"

namespace {

using Particle = autopasdm::Particle;

Particle makeParticle(unsigned long id, double marker) {
  Particle particle;
  particle.setID(id);
  particle.setR({marker, marker + 0.1, marker + 0.2});
  particle.setF({-marker, -marker - 0.1, -marker - 0.2});
  return particle;
}

void expectParticleEquals(const Particle &particle, unsigned long expectedId, double expectedMarker) {
  EXPECT_EQ(particle.getID(), expectedId);
  EXPECT_EQ(particle.getR(), (std::array<double, 3>{expectedMarker, expectedMarker + 0.1, expectedMarker + 0.2}));
  EXPECT_EQ(particle.getF(), (std::array<double, 3>{-expectedMarker, -expectedMarker - 0.1, -expectedMarker - 0.2}));
}

TEST(CommunicationTest, ExchangesParticlesWithLeftAndRightNeighbors) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  ASSERT_EQ(numberOfRanks, 4);

  const int left = (rank - 1 + numberOfRanks) % numberOfRanks;
  const int right = (rank + 1) % numberOfRanks;

  const double leftMarker = 10. + rank;
  const double rightMarker = 20. + rank;
  const std::vector<Particle> sendLeft{makeParticle(100 + rank, leftMarker)};
  const std::vector<Particle> sendRight{makeParticle(200 + rank, rightMarker)};

  const auto received = autopasdm::exchangeLeftRight(MPI_COMM_WORLD, left, right, sendLeft, sendRight, 1000);

  ASSERT_EQ(received.recvFromLeft.size(), 1);
  ASSERT_EQ(received.recvFromRight.size(), 1);

  expectParticleEquals(received.recvFromLeft.front(), 200 + left, 20. + left);
  expectParticleEquals(received.recvFromRight.front(), 100 + right, 10. + right);
}

TEST(CommunicationTest, SupportsSplitBeginFinishNeighborExchange) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  ASSERT_EQ(numberOfRanks, 4);

  const int left = (rank - 1 + numberOfRanks) % numberOfRanks;
  const int right = (rank + 1) % numberOfRanks;

  const std::vector<Particle> sendLeft{makeParticle(500 + rank, 50. + rank)};
  const std::vector<Particle> sendRight{makeParticle(600 + rank, 60. + rank)};

  auto request = autopasdm::beginLeftRightExchange(MPI_COMM_WORLD, left, right, sendLeft, sendRight, 1500);
  EXPECT_TRUE(request.active());

  // The split API deliberately leaves the payload exchange in flight here so
  // useful work can be inserted between begin and finish in later algorithms.
  const auto localWork = rank * rank;
  EXPECT_GE(localWork, 0);

  const auto received = autopasdm::finishLeftRightExchange(request);
  EXPECT_FALSE(request.active());

  ASSERT_EQ(received.recvFromLeft.size(), 1);
  ASSERT_EQ(received.recvFromRight.size(), 1);
  expectParticleEquals(received.recvFromLeft.front(), 600 + left, 60. + left);
  expectParticleEquals(received.recvFromRight.front(), 500 + right, 50. + right);

  EXPECT_THROW((void)autopasdm::finishLeftRightExchange(request), std::logic_error);
}

TEST(CommunicationTest, SplitExchangeSupportsMissingNeighbors) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  ASSERT_EQ(numberOfRanks, 4);

  const int left = rank == 0 ? -1 : rank - 1;
  const int right = rank == numberOfRanks - 1 ? -1 : rank + 1;

  const std::vector<Particle> sendLeft{makeParticle(700 + rank, 70. + rank)};
  const std::vector<Particle> sendRight{makeParticle(800 + rank, 80. + rank)};

  auto request = autopasdm::beginLeftRightExchange(MPI_COMM_WORLD, left, right, sendLeft, sendRight, 1750);
  const auto received = autopasdm::finishLeftRightExchange(request);

  const std::size_t expectedFromLeft = left < 0 ? 0 : 1;
  const std::size_t expectedFromRight = right < 0 ? 0 : 1;
  ASSERT_EQ(received.recvFromLeft.size(), expectedFromLeft);
  ASSERT_EQ(received.recvFromRight.size(), expectedFromRight);

  if (left >= 0) {
    expectParticleEquals(received.recvFromLeft.front(), 800 + left, 80. + left);
  }
  if (right >= 0) {
    expectParticleEquals(received.recvFromRight.front(), 700 + right, 70. + right);
  }
}

TEST(CommunicationTest, SupportsEmptyNeighborBuffers) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  ASSERT_EQ(numberOfRanks, 4);

  const int left = (rank - 1 + numberOfRanks) % numberOfRanks;
  const int right = (rank + 1) % numberOfRanks;

  std::vector<Particle> sendLeft;
  std::vector<Particle> sendRight;
  if (rank % 2 == 0) {
    sendLeft.push_back(makeParticle(300 + rank, 30. + rank));
  } else {
    sendRight.push_back(makeParticle(400 + rank, 40. + rank));
  }

  const auto received = autopasdm::exchangeLeftRight(MPI_COMM_WORLD, left, right, sendLeft, sendRight, 2000);

  const std::size_t expectedFromLeft = left % 2 == 1 ? 1 : 0;
  const std::size_t expectedFromRight = right % 2 == 0 ? 1 : 0;

  ASSERT_EQ(received.recvFromLeft.size(), expectedFromLeft);
  ASSERT_EQ(received.recvFromRight.size(), expectedFromRight);

  if (expectedFromLeft == 1) {
    expectParticleEquals(received.recvFromLeft.front(), 400 + left, 40. + left);
  }
  if (expectedFromRight == 1) {
    expectParticleEquals(received.recvFromRight.front(), 300 + right, 30. + right);
  }
}

TEST(CommunicationTest, ExchangesParticlesWithArbitraryDestinationRanks) {
  int rank = 0;
  int numberOfRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  ASSERT_EQ(numberOfRanks, 4);

  std::vector<std::vector<Particle>> particlesByDestination(numberOfRanks);
  for (int destination = 0; destination < numberOfRanks; ++destination) {
    const auto id = static_cast<unsigned long>(rank * 100 + destination);
    const double marker = rank * 10. + destination;
    particlesByDestination[destination].push_back(makeParticle(id, marker));
  }

  auto received = autopasdm::exchangeParticlesByRank(MPI_COMM_WORLD, particlesByDestination);
  ASSERT_EQ(received.size(), static_cast<std::size_t>(numberOfRanks));

  std::sort(received.begin(), received.end(), [](const auto &a, const auto &b) { return a.getID() < b.getID(); });

  for (int source = 0; source < numberOfRanks; ++source) {
    const auto expectedId = static_cast<unsigned long>(source * 100 + rank);
    const double expectedMarker = source * 10. + rank;
    expectParticleEquals(received[source], expectedId, expectedMarker);
  }
}

TEST(CommunicationTest, RejectsWrongNumberOfDestinationBuffers) {
  int numberOfRanks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  ASSERT_EQ(numberOfRanks, 4);

  std::vector<std::vector<Particle>> particlesByDestination(numberOfRanks - 1);
  EXPECT_THROW(autopasdm::exchangeParticlesByRank(MPI_COMM_WORLD, particlesByDestination), std::invalid_argument);
}

}  // namespace
