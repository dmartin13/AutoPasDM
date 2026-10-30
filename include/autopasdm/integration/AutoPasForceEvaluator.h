#pragma once

namespace autopasdm::timeintegration {

/**
 * Evaluates interactions with one AutoPas functor.
 *
 * The particle system's computeInteractions() method performs the distributed
 * preparation and returns whether this was a tuning iteration.
 */
template <class Functor>
class AutoPasForceEvaluator {
 public:
  explicit AutoPasForceEvaluator(Functor &functor) : _functor(functor) {}

  template <class ParticleSystem>
  bool operator()(ParticleSystem &particles) const {
    return particles.computeInteractions(&_functor);
  }

 private:
  Functor &_functor;
};

}  // namespace autopasdm::timeintegration