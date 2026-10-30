#pragma once

#include <array>

#include "src/TypeDefinitions.h"

struct MDFlexibleParticleOperations {
  const ParticlePropertiesLibraryType &particlePropertiesLibrary;
  std::array<double, 3> globalForce;

  template <class Particle>
  void resetForce(Particle &particle) const {
    // Set the force to the configured external force before pair forces are added.
    particle.setF(globalForce);
  }

  template <class Particle>
  double getMass(const Particle &particle) const {
    return particlePropertiesLibrary.getMolMass(particle.getTypeId());
  }

  template <class Particle>
  auto getPosition(const Particle &particle) const {
    return particle.getR();
  }

  template <class Particle, class Vector>
  void setPosition(Particle &particle, const Vector &position) const {
    particle.setR(position);
  }

  template <class Particle>
  auto getVelocity(const Particle &particle) const {
    return particle.getV();
  }

  template <class Particle, class Vector>
  void setVelocity(Particle &particle, const Vector &velocity) const {
    particle.setV(velocity);
  }

  template <class Particle>
  auto getForce(const Particle &particle) const {
    return particle.getF();
  }
};