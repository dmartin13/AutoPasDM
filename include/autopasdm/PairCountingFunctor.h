#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autopas/baseFunctors/PairwiseFunctor.h"

namespace autopasdm {

/**
 * Small AoS-only test functor that counts interacting particle pairs.
 *
 * This is an ordinary AutoPas functor. It is templated only so the same interaction test also verifies that
 * AutoPasDM is no longer tied to one concrete particle type.
 *
 * @tparam Particle AutoPas-compatible particle type used by the test.
 */
template <class Particle>
class PairCountingFunctor : public autopas::PairwiseFunctor<Particle, PairCountingFunctor<Particle>> {
 public:
  using SoAArraysType = typename Particle::SoAArraysType;
  using SoAView = autopas::SoAView<SoAArraysType>;
  using NeighborList = std::vector<size_t, autopas::AlignedAllocator<size_t>>;

  /**
   * Construct a pair counter for one local ownership box.
   *
   * @param cutoff Interaction cutoff used when counting pairs.
   * @param localMin Lower corner of the local ownership box.
   * @param localMax Upper corner of the local ownership box.
   */
  PairCountingFunctor(double cutoff, std::array<double, 3> localMin, std::array<double, 3> localMax)
      : autopas::PairwiseFunctor<Particle, PairCountingFunctor<Particle>>(cutoff),
        _cutoffSquared(cutoff * cutoff),
        _localMin(localMin),
        _localMax(localMax) {}

  /** Reset all counters before an AutoPas traversal. */
  void initTraversal() override {
    _numPairs = 0;
    _numOwnedOwnedPairs = 0;
    _numOwnedHaloPairs = 0;
    _numHaloHaloPairs = 0;
    _ownedHaloPairs.clear();
  }

  /**
   * Count one AoS particle pair when it lies within the cutoff.
   *
   * @param particleA First particle of the pair.
   * @param particleB Second particle of the pair.
   * @param newton3 Whether the traversal uses Newton's third law. The value does not change the counting logic.
   */
  void AoSFunctor(Particle &particleA, Particle &particleB, bool /*newton3*/) override {
    const auto &rA = particleA.getR();
    const auto &rB = particleB.getR();

    const double dx = rA[0] - rB[0];
    const double dy = rA[1] - rB[1];
    const double dz = rA[2] - rB[2];
    const double distanceSquared = dx * dx + dy * dy + dz * dz;

    if (distanceSquared >= _cutoffSquared) {
      return;
    }

    ++_numPairs;

    const bool aOwned = isInsideLocalDomain(rA);
    const bool bOwned = isInsideLocalDomain(rB);

    if (aOwned and bOwned) {
      ++_numOwnedOwnedPairs;
    } else if (aOwned != bOwned) {
      ++_numOwnedHaloPairs;
      _ownedHaloPairs.emplace_back(particleA.getID(), particleB.getID());
    } else {
      ++_numHaloHaloPairs;
    }
  }

  /**
   * Reject unsupported single-cell SoA traversal.
   *
   * @param soa SoA view supplied by AutoPas.
   * @param newton3 Whether the traversal uses Newton's third law.
   */
  void SoAFunctorSingle(SoAView /*soa*/, bool /*newton3*/) override {
    throw std::runtime_error("PairCountingFunctor only supports AoS.");
  }

  /**
   * Reject unsupported pair-cell SoA traversal.
   *
   * @param soa1 First SoA view supplied by AutoPas.
   * @param soa2 Second SoA view supplied by AutoPas.
   * @param newton3 Whether the traversal uses Newton's third law.
   */
  void SoAFunctorPair(SoAView /*soa1*/, SoAView /*soa2*/, bool /*newton3*/) override {
    throw std::runtime_error("PairCountingFunctor only supports AoS.");
  }

  /**
   * Reject unsupported Verlet-list SoA traversal.
   *
   * @param soa SoA view supplied by AutoPas.
   * @param indexFirst Index of the first particle in the Verlet interaction.
   * @param neighborList Neighbor indices supplied by AutoPas.
   * @param newton3 Whether the traversal uses Newton's third law.
   */
  void SoAFunctorVerlet(SoAView /*soa*/, size_t /*indexFirst*/, const NeighborList & /*neighborList*/,
                        bool /*newton3*/) override {
    throw std::runtime_error("PairCountingFunctor only supports AoS.");
  }

  /** Indicate that the functor supports Newton's third law. */
  [[nodiscard]] bool allowsNewton3() override { return true; }

  /** Indicate that the functor does not support non-Newton3 traversals. */
  [[nodiscard]] bool allowsNonNewton3() override { return false; }

  /** Indicate that the functor may participate in AutoPas tuning. */
  [[nodiscard]] bool isRelevantForTuning() override { return true; }

  /** Get the functor name used by AutoPas diagnostics. */
  [[nodiscard]] std::string getName() override { return "PairCountingFunctor"; }

  /** Get the total number of interacting pairs counted in the last traversal. */
  [[nodiscard]] size_t numPairs() const { return _numPairs; }

  /** Get the number of owned-owned pairs counted in the last traversal. */
  [[nodiscard]] size_t numOwnedOwnedPairs() const { return _numOwnedOwnedPairs; }

  /** Get the number of owned-halo pairs counted in the last traversal. */
  [[nodiscard]] size_t numOwnedHaloPairs() const { return _numOwnedHaloPairs; }

  /** Get the number of halo-halo pairs counted in the last traversal. */
  [[nodiscard]] size_t numHaloHaloPairs() const { return _numHaloHaloPairs; }
  /** Get the particle ID pairs classified as owned-halo interactions in the last traversal. */
  [[nodiscard]] const std::vector<std::pair<unsigned long, unsigned long>> &ownedHaloPairs() const {
    return _ownedHaloPairs;
  }

 private:
  /**
   * Check whether a particle position belongs to the local ownership box used by the test functor.
   *
   * @param position Position to classify.
   */
  [[nodiscard]] bool isInsideLocalDomain(const std::array<double, 3> &position) const {
    return position[0] >= _localMin[0] and position[0] < _localMax[0] and position[1] >= _localMin[1] and
           position[1] < _localMax[1] and position[2] >= _localMin[2] and position[2] < _localMax[2];
  }

  double _cutoffSquared;
  std::array<double, 3> _localMin;
  std::array<double, 3> _localMax;
  size_t _numPairs{0};
  size_t _numOwnedOwnedPairs{0};
  size_t _numOwnedHaloPairs{0};
  size_t _numHaloHaloPairs{0};

  std::vector<std::pair<unsigned long, unsigned long>> _ownedHaloPairs;
};

}  // namespace autopasdm
