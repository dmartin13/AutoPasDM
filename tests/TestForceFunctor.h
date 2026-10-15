#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "autopas/baseFunctors/PairwiseFunctor.h"

namespace autopasdm::testing {

/**
 * Small deterministic AoS-only force functor for AutoPasDM tests.
 *
 * For every pair within the cutoff, particle A receives (r_B - r_A) and,
 * with Newton3 enabled, particle B receives the opposite force. Using a
 * deliberately simple force law keeps the expected result of the distributed
 * interaction test easy to calculate exactly and avoids a dependency on an
 * application-specific molecular-dynamics functor.
 */
template <class Particle>
class TestForceFunctor : public autopas::PairwiseFunctor<Particle, TestForceFunctor<Particle>> {
 public:
  using SoAArraysType = typename Particle::SoAArraysType;
  using SoAView = autopas::SoAView<SoAArraysType>;
  using NeighborList = std::span<const std::size_t>;

  explicit TestForceFunctor(double cutoff)
      : autopas::PairwiseFunctor<Particle, TestForceFunctor<Particle>>(cutoff), _cutoffSquared(cutoff * cutoff) {}

  void AoSFunctor(Particle &particleA, Particle &particleB, bool newton3) override {
    const auto &rA = particleA.getR();
    const auto &rB = particleB.getR();

    std::array<double, 3> displacement{};
    double distanceSquared = 0.0;
    for (std::size_t d = 0; d < displacement.size(); ++d) {
      displacement[d] = rB[d] - rA[d];
      distanceSquared += displacement[d] * displacement[d];
    }

    if (distanceSquared >= _cutoffSquared) {
      return;
    }

    auto forceA = particleA.getF();
    for (std::size_t d = 0; d < displacement.size(); ++d) {
      forceA[d] += displacement[d];
    }
    particleA.setF(forceA);

    if (newton3) {
      auto forceB = particleB.getF();
      for (std::size_t d = 0; d < displacement.size(); ++d) {
        forceB[d] -= displacement[d];
      }
      particleB.setF(forceB);
    }
  }

  void SoAFunctorSingle(SoAView /*soa*/, bool /*newton3*/) override {
    throw std::runtime_error("TestForceFunctor only supports AoS.");
  }

  void SoAFunctorPair(SoAView /*soa1*/, SoAView /*soa2*/, bool /*newton3*/) override {
    throw std::runtime_error("TestForceFunctor only supports AoS.");
  }

  void SoAFunctorVerlet(SoAView /*soa*/, std::size_t /*indexFirst*/, NeighborList /*neighborList*/,
                        bool /*newton3*/) override {
    throw std::runtime_error("TestForceFunctor only supports AoS.");
  }

  [[nodiscard]] bool allowsNewton3() override { return true; }
  [[nodiscard]] bool allowsNonNewton3() override { return false; }
  [[nodiscard]] bool isRelevantForTuning() override { return true; }
  [[nodiscard]] std::string getName() override { return "TestForceFunctor"; }

 private:
  double _cutoffSquared;
};

}  // namespace autopasdm::testing
