#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "autopasdm/integration/StormerVerlet.h"

namespace {

struct TestParticle {
  double mass{1.0};
  std::array<double, 3> position{0.0, 0.0, 0.0};
  std::array<double, 3> velocity{0.0, 0.0, 0.0};
  std::array<double, 3> force{0.0, 0.0, 0.0};
};

struct TestParticleSystem {
  std::vector<TestParticle> particles;

  template <class Function>
  void applyToOwnedParticles(Function &&function) {
    for (auto &particle : particles) {
      function(particle);
    }
  }
};

struct TestParticleOperations {
  void resetForce(TestParticle &particle) const { particle.force = {0.0, 0.0, 0.0}; }

  double getMass(const TestParticle &particle) const { return particle.mass; }

  auto getPosition(const TestParticle &particle) const { return particle.position; }

  void setPosition(TestParticle &particle, const std::array<double, 3> &position) const {
    particle.position = position;
  }

  auto getVelocity(const TestParticle &particle) const { return particle.velocity; }

  void setVelocity(TestParticle &particle, const std::array<double, 3> &velocity) const {
    particle.velocity = velocity;
  }

  auto getForce(const TestParticle &particle) const { return particle.force; }
};

struct ConstantForceEvaluator {
  std::array<double, 3> force{0.0, 0.0, 0.0};
  bool result{false};
  int calls{0};
  std::vector<std::array<double, 3>> forcesBeforeEvaluation;

  bool operator()(TestParticleSystem &particles) {
    ++calls;

    particles.applyToOwnedParticles([&](auto &particle) {
      // Record the force before replacing it, so the test can check that it was reset.
      forcesBeforeEvaluation.push_back(particle.force);
      particle.force = force;
    });

    return result;
  }
};

struct PositionDependentForceEvaluator {
  int calls{0};

  bool operator()(TestParticleSystem &particles) {
    ++calls;
    particles.applyToOwnedParticles([](auto &particle) { particle.force = {particle.position[0], 0.0, 0.0}; });
    return false;
  }
};

TEST(StormerVerletTest, InitializeResetsAndComputesInitialForce) {
  TestParticleSystem particles{{TestParticle{
      .mass = 1.0,
      .position = {1.0, 0.0, 0.0},
      .velocity = {0.0, 0.0, 0.0},
      .force = {7.0, 0.0, 0.0},
  }}};
  TestParticleOperations operations;
  ConstantForceEvaluator forceEvaluator{
      .force = {2.0, 0.0, 0.0},
      .result = true,
  };
  autopasdm::timeintegration::StormerVerlet integrator;

  const bool result = integrator.initialize(particles, operations, forceEvaluator);

  EXPECT_TRUE(result);
  EXPECT_EQ(forceEvaluator.calls, 1);
  EXPECT_EQ(forceEvaluator.forcesBeforeEvaluation[0], (std::array<double, 3>{0.0, 0.0, 0.0}));
  EXPECT_EQ(particles.particles[0].force, (std::array<double, 3>{2.0, 0.0, 0.0}));
}

TEST(StormerVerletTest, StepUpdatesPositionAndVelocityForConstantForce) {
  TestParticleSystem particles{{TestParticle{
      .mass = 2.0,
      .position = {1.0, 0.0, 0.0},
      .velocity = {3.0, 0.0, 0.0},
      .force = {0.0, 0.0, 0.0},
  }}};
  TestParticleOperations operations;
  ConstantForceEvaluator forceEvaluator{
      .force = {2.0, 0.0, 0.0},
      .result = true,
  };
  autopasdm::timeintegration::StormerVerlet integrator;

  (void)integrator.initialize(particles, operations, forceEvaluator);
  const bool result = integrator.step(particles, 0.1, operations, forceEvaluator);

  EXPECT_TRUE(result);
  EXPECT_EQ(forceEvaluator.calls, 2);
  EXPECT_NEAR(particles.particles[0].position[0], 1.305, 1e-12);
  EXPECT_NEAR(particles.particles[0].velocity[0], 3.1, 1e-12);
}

TEST(StormerVerletTest, RecomputesForceAtUpdatedPosition) {
  TestParticleSystem particles{{TestParticle{
      .mass = 1.0,
      .position = {1.0, 0.0, 0.0},
      .velocity = {0.0, 0.0, 0.0},
      .force = {0.0, 0.0, 0.0},
  }}};
  TestParticleOperations operations;
  PositionDependentForceEvaluator forceEvaluator;
  autopasdm::timeintegration::StormerVerlet integrator;

  (void)integrator.initialize(particles, operations, forceEvaluator);
  (void)integrator.step(particles, 0.2, operations, forceEvaluator);

  EXPECT_EQ(forceEvaluator.calls, 2);
  EXPECT_NEAR(particles.particles[0].position[0], 1.02, 1e-12);
  EXPECT_NEAR(particles.particles[0].force[0], 1.02, 1e-12);
  EXPECT_NEAR(particles.particles[0].velocity[0], 0.202, 1e-12);
}

}  // namespace