#pragma once

#include <array>

#include "autopas/particles/ParticleDefinitions.h"
#include "autopasdm/ParticleSerialization.h"

namespace autopasdm {

/**
 * Particle type used by the minimal example.
 *
 * AutoPasDM itself is no longer tied to this type. Applications can instantiate
 * AutoPasDM with their own AutoPas-compatible particle type.
 */
using Particle = autopas::ParticleBaseFP64;

/**
 * Serializer for the minimal example particle.
 */
template <>
struct ParticleSerializer<Particle> {
  /** Serialized state required to reconstruct the minimal example particle on another rank. */
  struct Message {
    std::array<double, 3> r{};
    std::array<double, 3> f{};
    unsigned long id{0};
  };

  /**
   * Serialize one minimal-example particle.
   *
   * @param particle Particle to serialize.
   */
  static Message pack(const Particle &particle) { return Message{particle.getR(), particle.getF(), particle.getID()}; }

  /**
   * Reconstruct one minimal-example particle from its serialized state.
   *
   * @param message Serialized particle state.
   */
  static Particle unpack(const Message &message) {
    Particle particle;
    particle.setR(message.r);
    particle.setF(message.f);
    particle.setID(message.id);
    return particle;
  }
};

}  // namespace autopasdm
