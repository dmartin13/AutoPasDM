#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "autopasdm/BoundaryType.h"
#include "src/ReflectiveBoundary.h"

namespace {

struct TestParticle {
  std::array<double, 3> r{};
  std::array<double, 3> f{};
  std::size_t typeId{};

  [[nodiscard]] const std::array<double, 3> &getR() const { return r; }
  [[nodiscard]] const std::array<double, 3> &getF() const { return f; }
  [[nodiscard]] std::size_t getTypeId() const { return typeId; }

  void addF(const std::array<double, 3> &force) {
    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      f[dimension] += force[dimension];
    }
  }
};

class TestParticlePropertiesLibrary {
 public:
  explicit TestParticlePropertiesLibrary(std::vector<double> sigmas) : _sigmas(std::move(sigmas)) {}

  [[nodiscard]] double getSigma(std::size_t typeId) const { return _sigmas.at(typeId); }

  [[nodiscard]] double getMixingSigmaSquared(std::size_t firstType, std::size_t secondType) const {
    EXPECT_EQ(firstType, secondType);
    const auto sigma = getSigma(firstType);
    return sigma * sigma;
  }

  [[nodiscard]] double getMixing24Epsilon(std::size_t firstType, std::size_t secondType) const {
    EXPECT_EQ(firstType, secondType);
    return 24.;
  }

 private:
  std::vector<double> _sigmas;
};

class TestContainer {
 public:
  std::array<double, 3> globalMin{0., 0., 0.};
  std::array<double, 3> globalMax{8., 8., 8.};
  std::array<double, 3> localMin{0., 0., 0.};
  std::array<double, 3> localMax{8., 8., 8.};
  std::array<autopasdm::BoundaryType, 3> boundaries{autopasdm::BoundaryType::none, autopasdm::BoundaryType::none,
                                                    autopasdm::BoundaryType::none};
  std::vector<TestParticle> particles;
  std::size_t particlesVisitedByRegionOperations{};

  [[nodiscard]] const auto &boundaryTypes() const { return boundaries; }
  [[nodiscard]] const auto &globalBoxMin() const { return globalMin; }
  [[nodiscard]] const auto &globalBoxMax() const { return globalMax; }
  [[nodiscard]] const auto &localBoxMin() const { return localMin; }
  [[nodiscard]] const auto &localBoxMax() const { return localMax; }

  template <class Kernel>
  void applyToOwnedParticlesInRegion(const std::array<double, 3> &regionMin, const std::array<double, 3> &regionMax,
                                     Kernel &&kernel) {
    for (auto &particle : particles) {
      const auto &position = particle.getR();
      const bool insideRegion = position[0] >= regionMin[0] and position[0] < regionMax[0] and
                                position[1] >= regionMin[1] and position[1] < regionMax[1] and
                                position[2] >= regionMin[2] and position[2] < regionMax[2];
      if (insideRegion) {
        ++particlesVisitedByRegionOperations;
        kernel(particle);
      }
    }
  }
};

TEST(ReflectiveBoundaryTest, AppliesRepulsiveForceAtLowerAndUpperWalls) {
  TestContainer container;
  container.boundaries = {autopasdm::BoundaryType::reflective, autopasdm::BoundaryType::reflective,
                          autopasdm::BoundaryType::none};
  container.particles = {{{0.5, 4., 4.}, {}, 0}, {{4., 7.5, 4.}, {}, 0}};
  TestParticlePropertiesLibrary particlePropertiesLibrary({1.});

  ReflectiveBoundary::apply(container, particlePropertiesLibrary, 1.);

  EXPECT_NEAR(container.particles[0].getF()[0], 24., 1e-12);
  EXPECT_NEAR(container.particles[0].getF()[1], 0., 1e-12);
  EXPECT_NEAR(container.particles[0].getF()[2], 0., 1e-12);

  EXPECT_NEAR(container.particles[1].getF()[0], 0., 1e-12);
  EXPECT_NEAR(container.particles[1].getF()[1], -24., 1e-12);
  EXPECT_NEAR(container.particles[1].getF()[2], 0., 1e-12);
}

TEST(ReflectiveBoundaryTest, AppliesIndependentForcesAtReflectiveCorner) {
  TestContainer container;
  container.boundaries = {autopasdm::BoundaryType::reflective, autopasdm::BoundaryType::reflective,
                          autopasdm::BoundaryType::reflective};
  container.particles = {{{0.5, 0.5, 0.5}, {}, 0}};
  TestParticlePropertiesLibrary particlePropertiesLibrary({1.});

  ReflectiveBoundary::apply(container, particlePropertiesLibrary, 1.);

  EXPECT_NEAR(container.particles[0].getF()[0], 24., 1e-12);
  EXPECT_NEAR(container.particles[0].getF()[1], 24., 1e-12);
  EXPECT_NEAR(container.particles[0].getF()[2], 24., 1e-12);
}

TEST(ReflectiveBoundaryTest, OnlyActsInsideRepulsiveLennardJonesRange) {
  TestContainer container;
  container.boundaries = {autopasdm::BoundaryType::reflective, autopasdm::BoundaryType::none,
                          autopasdm::BoundaryType::none};
  container.particles = {{{0.55, 4., 4.}, {}, 0}, {{0.55, 5., 5.}, {}, 1}};
  TestParticlePropertiesLibrary particlePropertiesLibrary({1., 0.5});

  ReflectiveBoundary::apply(container, particlePropertiesLibrary, 1.);

  EXPECT_GT(container.particles[0].getF()[0], 0.);
  EXPECT_DOUBLE_EQ(container.particles[1].getF()[0], 0.);
}

TEST(ReflectiveBoundaryTest, IgnoresNoneAndPeriodicBoundaries) {
  TestContainer container;
  container.boundaries = {autopasdm::BoundaryType::periodic, autopasdm::BoundaryType::none,
                          autopasdm::BoundaryType::none};
  container.particles = {{{0.5, 4., 4.}, {}, 0}};
  TestParticlePropertiesLibrary particlePropertiesLibrary({1.});

  ReflectiveBoundary::apply(container, particlePropertiesLibrary, 1.);

  EXPECT_EQ(container.particles[0].getF(), (std::array<double, 3>{0., 0., 0.}));
}

TEST(ReflectiveBoundaryTest, OnlyRunsOnRanksTouchingTheGlobalReflectiveWall) {
  TestContainer container;
  container.boundaries = {autopasdm::BoundaryType::reflective, autopasdm::BoundaryType::none,
                          autopasdm::BoundaryType::none};
  container.localMin = {2., 0., 0.};
  container.localMax = {4., 8., 8.};
  container.particles = {{{2.1, 4., 4.}, {}, 0}};
  TestParticlePropertiesLibrary particlePropertiesLibrary({1.});

  ReflectiveBoundary::apply(container, particlePropertiesLibrary, 1.);

  EXPECT_EQ(container.particles[0].getF(), (std::array<double, 3>{0., 0., 0.}));
  EXPECT_EQ(container.particlesVisitedByRegionOperations, 0);
}

TEST(ReflectiveBoundaryTest, RestrictsTraversalToReflectiveWallRegions) {
  TestContainer container;
  container.boundaries = {autopasdm::BoundaryType::reflective, autopasdm::BoundaryType::none,
                          autopasdm::BoundaryType::none};
  container.particles = {{{0.5, 4., 4.}, {}, 0}, {{4., 4., 4.}, {}, 0}, {{7.5, 4., 4.}, {}, 0}};
  TestParticlePropertiesLibrary particlePropertiesLibrary({1.});

  ReflectiveBoundary::apply(container, particlePropertiesLibrary, 1.);

  EXPECT_EQ(container.particlesVisitedByRegionOperations, 2);
  EXPECT_NEAR(container.particles[0].getF()[0], 24., 1e-12);
  EXPECT_EQ(container.particles[1].getF(), (std::array<double, 3>{0., 0., 0.}));
  EXPECT_NEAR(container.particles[2].getF()[0], -24., 1e-12);
}

}  // namespace
