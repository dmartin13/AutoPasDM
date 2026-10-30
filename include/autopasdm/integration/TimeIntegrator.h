#pragma once

#include <concepts>
#include <functional>

namespace autopasdm::integration {

template <class ForceEvaluator, class ParticleSystem>
concept ForceEvaluatorFor = requires(ForceEvaluator &forceEvaluator, ParticleSystem &particles) {
  std::invoke(forceEvaluator, particles);
};

template <class Integrator, class ParticleSystem, class ParticleOperations, class ForceEvaluator>
concept TimeIntegrator = ForceEvaluatorFor<ForceEvaluator, ParticleSystem> &&
    requires(Integrator &integrator, ParticleSystem &particles, ParticleOperations &operations,
             ForceEvaluator &forceEvaluator, double dt) {
  integrator.initialize(particles, operations, forceEvaluator);
  integrator.step(particles, dt, operations, forceEvaluator);
};

}  // namespace autopasdm::integration