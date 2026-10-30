#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>

#include "autopas/utils/ArrayMath.h"
#include "autopasdm/integration/TimeIntegrator.h"

namespace autopasdm::timeintegration {

struct StormerVerlet {
  template <class ParticleSystem, class ParticleOperations, class ForceEvaluator>
  decltype(auto) initialize(ParticleSystem &particles, ParticleOperations &operations,
                            ForceEvaluator &forceEvaluator) const {
    // Clear the forces before computing the initial forces.
    particles.applyToOwnedParticles([&](auto &particle) { operations.resetForce(particle); });

    return std::invoke(forceEvaluator, particles);
  }

  template <class ParticleSystem, class ParticleOperations, class ForceEvaluator>
  auto step(ParticleSystem &particles, double dt, ParticleOperations &operations,
            ForceEvaluator &forceEvaluator) const {
    // Apply the first half of the velocity update, then update the position.
    particles.applyToOwnedParticles([&](auto &particle) {
      const double mass = operations.getMass(particle);
      auto velocity = operations.getVelocity(particle);
      const auto force = operations.getForce(particle);

      const auto scaledForce = autopas::utils::ArrayMath::mulScalar(force, dt / (2.0 * mass));
      velocity = autopas::utils::ArrayMath::add(velocity, scaledForce);
      operations.setVelocity(particle, velocity);

      auto position = operations.getPosition(particle);
      position = autopas::utils::ArrayMath::add(position, autopas::utils::ArrayMath::mulScalar(velocity, dt));
      operations.setPosition(particle, position);
    });

    // Compute the forces at the updated positions.
    particles.applyToOwnedParticles([&](auto &particle) { operations.resetForce(particle); });
    using ForceResult = std::invoke_result_t<ForceEvaluator &, ParticleSystem &>;

    if constexpr (std::is_void_v<ForceResult>) {
      std::invoke(forceEvaluator, particles);
      updateSecondHalfVelocity(particles, dt, operations);
    } else {
      auto result = std::invoke(forceEvaluator, particles);
      updateSecondHalfVelocity(particles, dt, operations);
      return result;
    }
  }

 private:
  template <class ParticleSystem, class ParticleOperations>
  static void updateSecondHalfVelocity(ParticleSystem &particles, double dt, ParticleOperations &operations) {
    // Complete the velocity update using the newly computed forces.
    particles.applyToOwnedParticles([&](auto &particle) {
      const double mass = operations.getMass(particle);
      auto velocity = operations.getVelocity(particle);
      const auto force = operations.getForce(particle);

      const auto scaledForce = autopas::utils::ArrayMath::mulScalar(force, dt / (2.0 * mass));
      velocity = autopas::utils::ArrayMath::add(velocity, scaledForce);

      operations.setVelocity(particle, velocity);
    });
  }
};

}  // namespace autopasdm::timeintegration