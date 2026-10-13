#pragma once

#include <array>
#include <cstddef>

#include "autopasdm/ParticleSerialization.h"
#include "molecularDynamicsLibrary/MoleculeLJ.h"

namespace autopasdm {

template <>
struct ParticleSerializer<mdLib::MoleculeLJ> {
  /** Serialized state required to reconstruct a MoleculeLJ on another rank. */
  struct Message {
    std::array<double, 3> r{};
    std::array<double, 3> v{};
    std::array<double, 3> f{};
    unsigned long id{0};
    std::size_t typeId{0};
  };

  /**
   * Serialize one MoleculeLJ particle for distributed communication.
   *
   * @param particle Particle to serialize.
   */
  static Message pack(const mdLib::MoleculeLJ &particle) {
    return Message{particle.getR(), particle.getV(), particle.getF(), particle.getID(), particle.getTypeId()};
  }

  /**
   * Reconstruct one MoleculeLJ particle from its serialized state.
   *
   * @param message Serialized particle state.
   */
  static mdLib::MoleculeLJ unpack(const Message &message) {
    mdLib::MoleculeLJ particle(message.r, message.v, message.id, message.typeId);
    particle.setF(message.f);
    particle.setOwnershipState(autopas::OwnershipState::owned);
    return particle;
  }
};

}  // namespace autopasdm
