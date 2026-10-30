#pragma once

#include <array>
#include <cstddef>

#include "src/TypeDefinitions.h"

struct MDFlexibleParticleOperations {
  const ParticlePropertiesLibraryType &particlePropertiesLibrary;
  std::array<double, 3> globalForce;

  template <class Particle>
  void resetForce(Particle &particle) const {
    particle.setF(globalForce);
  }

  template <class Particle>
  void updatePosition(Particle &particle, double dt) const {
    const auto mass = particlePropertiesLibrary.getMolMass(particle.getTypeId());
    const auto force = particle.getF();

    particle.setOldF(force);
    particle.setF(globalForce);

    auto displacement = particle.getV();
    for (std::size_t d = 0; d < 3; ++d) {
      displacement[d] = dt * displacement[d] + (dt * dt / (2. * mass)) * force[d];
    }
    particle.addR(displacement);
  }

  template <class Particle>
  void updateVelocity(Particle &particle, double dt) const {
    const auto mass = particlePropertiesLibrary.getMolMass(particle.getTypeId());
    const auto oldForce = particle.getOldF();

    auto deltaVelocity = particle.getF();
    for (std::size_t d = 0; d < 3; ++d) {
      deltaVelocity[d] = (dt / (2. * mass)) * (deltaVelocity[d] + oldForce[d]);
    }
    particle.addV(deltaVelocity);
  }
};