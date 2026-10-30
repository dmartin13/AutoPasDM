#pragma once

#include <functional>
#include <type_traits>

#include "autopasdm/integration/TimeIntegrator.h"

namespace autopasdm::timeintegration {

struct StormerVerlet {
  template <class ParticleSystem, class ParticleOperations, class ForceEvaluator>
  decltype(auto) initialize(ParticleSystem &particles, ParticleOperations &operations,
                            ForceEvaluator &forceEvaluator) const {
    particles.applyToOwnedParticles([&](auto &particle) { operations.resetForce(particle); });

    return std::invoke(forceEvaluator, particles);
  }

  template <class ParticleSystem, class ParticleOperations, class ForceEvaluator>
  auto step(ParticleSystem &particles, double dt, ParticleOperations &operations,
            ForceEvaluator &forceEvaluator) const {
    particles.applyToOwnedParticles([&](auto &particle) { operations.updatePosition(particle, dt); });

    auto updateVelocities = [&] {
      particles.applyToOwnedParticles([&](auto &particle) { operations.updateVelocity(particle, dt); });
    };

    using ForceResult = std::invoke_result_t<ForceEvaluator &, ParticleSystem &>;

    if constexpr (std::is_void_v<ForceResult>) {
      std::invoke(forceEvaluator, particles);
      updateVelocities();
    } else {
      auto result = std::invoke(forceEvaluator, particles);
      updateVelocities();
      return result;
    }
  }
};

}  // namespace autopasdm::timeintegration