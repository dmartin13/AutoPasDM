/**
 * @file TimeDiscretization.cpp
 * @author N. Fottner
 * @date 13/05/19
 */
#include "TimeDiscretization.h"

#include <atomic>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "autopas/utils/ExceptionHandler.h"
#include "autopas/utils/WrapOpenMP.h"

namespace TimeDiscretization {
namespace {

#ifndef AUTOPAS_ENABLE_DYNAMIC_CONTAINERS
bool updatePositionAndResetForces(ParticleType &particle,
                                  const ParticlePropertiesLibraryType &particlePropertiesLibrary, const double deltaT,
                                  const std::array<double, 3> &globalForce, const double maxAllowedDistanceMoved,
                                  const double maxAllowedDistanceMovedSquared) {
  using autopas::utils::ArrayMath::dot;
  using namespace autopas::utils::ArrayMath::literals;

  const auto m = particlePropertiesLibrary.getMolMass(particle.getTypeId());
  auto v = particle.getV();
  auto f = particle.getF();
  particle.setOldF(f);
  particle.setF(globalForce);
  v *= deltaT;
  f *= (deltaT * deltaT / (2 * m));
  const auto displacement = v + f;

  if (not particle.addRDistanceCheck(displacement, maxAllowedDistanceMovedSquared) and
      maxAllowedDistanceMovedSquared > 0) {
    const auto distanceMoved = std::sqrt(dot(displacement, displacement));
    AUTOPAS_OPENMP(critical)
    std::cerr << "A particle moved farther than verletSkinPerTimestep/2: " << distanceMoved << " > "
              << maxAllowedDistanceMoved << "\n"
              << particle << "\nNew Position: " << particle.getR() + displacement << std::endl;
    return true;
  }

  return false;
}
#else
void updatePositionAndResetForces(ParticleType &particle,
                                  const ParticlePropertiesLibraryType &particlePropertiesLibrary, const double deltaT,
                                  const std::array<double, 3> &globalForce) {
  using namespace autopas::utils::ArrayMath::literals;

  const auto m = particlePropertiesLibrary.getMolMass(particle.getTypeId());
  auto v = particle.getV();
  auto f = particle.getF();
  particle.setOldF(f);
  particle.setF(globalForce);
  v *= deltaT;
  f *= (deltaT * deltaT / (2 * m));
  particle.addR(v + f);
}
#endif

void updateQuaternionAndResetTorque(ParticleType &particle,
                                    const ParticlePropertiesLibraryType &particlePropertiesLibrary, const double deltaT,
                                    const std::array<double, 3> &globalForce) {
  using namespace autopas::utils::ArrayMath::literals;
  using autopas::utils::ArrayMath::cross;
  using autopas::utils::ArrayMath::div;
  using autopas::utils::ArrayMath::dot;
  using autopas::utils::ArrayMath::normalize;
  using autopas::utils::quaternion::qMul;
  using autopas::utils::quaternion::rotatePosition;
  using autopas::utils::quaternion::rotatePositionBackwards;
  using autopas::utils::quaternion::rotateVectorOfPositions;

#if MD_FLEXIBLE_MODE == MULTISITE
  const auto halfDeltaT = 0.5 * deltaT;
  const double tol = 1e-13;
  const double tolSquared = tol * tol;

  const auto q = particle.getQuaternion();
  const auto angVelW = particle.getAngularVel();
  const auto angVelM = rotatePositionBackwards(q, angVelW);
  const auto torqueW = particle.getTorque();
  const auto torqueM = rotatePositionBackwards(q, torqueW);
  const auto I = particlePropertiesLibrary.getMomentOfInertia(particle.getTypeId());

  const auto angMomentumM = I * angVelM;
  const auto derivativeAngMomentumM = torqueM - cross(angVelM, angMomentumM);
  const auto angMomentumMHalfStep = angMomentumM + derivativeAngMomentumM * halfDeltaT;

  auto derivativeQHalfStep = qMul(q, div(angMomentumMHalfStep, I)) * 0.5;
  auto qHalfStep = normalize(q + derivativeQHalfStep * halfDeltaT);
  const auto angVelWHalfStep = angVelW + rotatePosition(q, torqueM / I) * halfDeltaT;

  auto qHalfStepOld = qHalfStep;
  qHalfStepOld[0] += 2 * tol;

  while (dot(qHalfStep - qHalfStepOld, qHalfStep - qHalfStepOld) > tolSquared) {
    qHalfStepOld = qHalfStep;
    const auto angVelMHalfStep = rotatePositionBackwards(qHalfStepOld, angVelWHalfStep);
    derivativeQHalfStep = qMul(qHalfStepOld, angVelMHalfStep) * 0.5;
    qHalfStep = normalize(q + derivativeQHalfStep * halfDeltaT);
  }

  const auto qFullStep = normalize(q + derivativeQHalfStep * deltaT);
  particle.setQuaternion(qFullStep);
  particle.setAngularVel(angVelWHalfStep);

  particle.setTorque({0., 0., 0.});
  if (std::any_of(globalForce.begin(), globalForce.end(),
                  [](double i) { return std::abs(i) > std::numeric_limits<double>::epsilon(); })) {
    const auto unrotatedSitePositions = particlePropertiesLibrary.getSitePositions(particle.getTypeId());
    const auto rotatedSitePositions = rotateVectorOfPositions(qFullStep, unrotatedSitePositions);
    for (size_t site = 0; site < particlePropertiesLibrary.getNumSites(particle.getTypeId()); site++) {
      particle.addTorque(cross(rotatedSitePositions[site], globalForce));
    }
  }
#else
  (void)particle;
  (void)particlePropertiesLibrary;
  (void)deltaT;
  (void)globalForce;
  autopas::utils::ExceptionHandler::exception(
      "Attempting to perform rotational integrations when md-flexible has not been compiled with multi-site support!");
#endif
}

void updateVelocity(ParticleType &particle, const ParticlePropertiesLibraryType &particlePropertiesLibrary,
                    const double deltaT) {
  using namespace autopas::utils::ArrayMath::literals;

  const auto molecularMass = particlePropertiesLibrary.getMolMass(particle.getTypeId());
  const auto force = particle.getF();
  const auto oldForce = particle.getOldF();
  const auto changeInVel = (force + oldForce) * (deltaT / (2 * molecularMass));
  particle.addV(changeInVel);
}

void updateAngularVelocity(ParticleType &particle, const ParticlePropertiesLibraryType &particlePropertiesLibrary,
                           const double deltaT) {
  using namespace autopas::utils::ArrayMath::literals;
  using autopas::utils::quaternion::rotatePosition;
  using autopas::utils::quaternion::rotatePositionBackwards;

#if MD_FLEXIBLE_MODE == MULTISITE
  const auto torqueW = particle.getTorque();
  const auto q = particle.getQuaternion();
  const auto I = particlePropertiesLibrary.getMomentOfInertia(particle.getTypeId());
  const auto torqueM = rotatePositionBackwards(q, torqueW);
  const auto torqueDivMoIM = torqueM / I;
  const auto torqueDivMoIW = rotatePosition(q, torqueDivMoIM);
  particle.addAngularVel(torqueDivMoIW * 0.5 * deltaT);
#else
  (void)particle;
  (void)particlePropertiesLibrary;
  (void)deltaT;
  autopas::utils::ExceptionHandler::exception(
      "Attempting to perform rotational integrations when md-flexible has not been compiled with multi-site support!");
#endif
}

}  // namespace

void calculatePositionsAndResetForces(autopasdm::AutoPasDM<ParticleType> &container,
                                      const ParticlePropertiesLibraryType &particlePropertiesLibrary,
                                      const double &deltaT, const std::array<double, 3> &globalForce,
                                      bool fastParticlesThrow) {
#ifdef AUTOPAS_ENABLE_DYNAMIC_CONTAINERS
  container.applyToOwnedParticles(
      [&](auto &particle) { updatePositionAndResetForces(particle, particlePropertiesLibrary, deltaT, globalForce); });
#else
  const auto maxAllowedDistanceMoved = container.getVerletSkin() / container.getVerletRebuildFrequency() / 2.;
  const auto maxAllowedDistanceMovedSquared = maxAllowedDistanceMoved * maxAllowedDistanceMoved;
  std::atomic_bool particleTooFast{false};

  container.applyToOwnedParticles([&](auto &particle) {
    if (updatePositionAndResetForces(particle, particlePropertiesLibrary, deltaT, globalForce, maxAllowedDistanceMoved,
                                     maxAllowedDistanceMovedSquared)) {
      particleTooFast.store(true, std::memory_order_relaxed);
    }
  });

  if (particleTooFast.load(std::memory_order_relaxed) and fastParticlesThrow) {
    throw std::runtime_error("At least one particle was too fast!");
  }
#endif
}

void calculatePositionsAndResetForces(autopas::AutoPas<ParticleType> &autoPasContainer,
                                      const ParticlePropertiesLibraryType &particlePropertiesLibrary,
                                      const double &deltaT, const std::array<double, 3> &globalForce,
                                      bool fastParticlesThrow) {
#ifdef AUTOPAS_ENABLE_DYNAMIC_CONTAINERS
  AUTOPAS_OPENMP(parallel)
  for (auto iter = autoPasContainer.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
    updatePositionAndResetForces(*iter, particlePropertiesLibrary, deltaT, globalForce);
  }
#else
  const auto maxAllowedDistanceMoved =
      autoPasContainer.getVerletSkin() / autoPasContainer.getVerletRebuildFrequency() / 2.;
  const auto maxAllowedDistanceMovedSquared = maxAllowedDistanceMoved * maxAllowedDistanceMoved;
  std::atomic_bool particleTooFast{false};

  AUTOPAS_OPENMP(parallel)
  for (auto iter = autoPasContainer.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
    if (updatePositionAndResetForces(*iter, particlePropertiesLibrary, deltaT, globalForce, maxAllowedDistanceMoved,
                                     maxAllowedDistanceMovedSquared)) {
      particleTooFast.store(true, std::memory_order_relaxed);
    }
  }

  if (particleTooFast.load(std::memory_order_relaxed) and fastParticlesThrow) {
    throw std::runtime_error("At least one particle was too fast!");
  }
#endif
}

void calculateQuaternionsAndResetTorques(autopasdm::AutoPasDM<ParticleType> &container,
                                         const ParticlePropertiesLibraryType &particlePropertiesLibrary,
                                         const double &deltaT, const std::array<double, 3> &globalForce) {
#if MD_FLEXIBLE_MODE == MULTISITE
  container.applyToOwnedParticles([&](auto &particle) {
    updateQuaternionAndResetTorque(particle, particlePropertiesLibrary, deltaT, globalForce);
  });
#else
  (void)container;
  (void)particlePropertiesLibrary;
  (void)deltaT;
  (void)globalForce;
  autopas::utils::ExceptionHandler::exception(
      "Attempting to perform rotational integrations when md-flexible has not been compiled with multi-site support!");
#endif
}

void calculateQuaternionsAndResetTorques(autopas::AutoPas<ParticleType> &autoPasContainer,
                                         const ParticlePropertiesLibraryType &particlePropertiesLibrary,
                                         const double &deltaT, const std::array<double, 3> &globalForce) {
#if MD_FLEXIBLE_MODE == MULTISITE
  AUTOPAS_OPENMP(parallel)
  for (auto iter = autoPasContainer.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
    updateQuaternionAndResetTorque(*iter, particlePropertiesLibrary, deltaT, globalForce);
  }
#else
  (void)autoPasContainer;
  (void)particlePropertiesLibrary;
  (void)deltaT;
  (void)globalForce;
  autopas::utils::ExceptionHandler::exception(
      "Attempting to perform rotational integrations when md-flexible has not been compiled with multi-site support!");
#endif
}

void calculateVelocities(autopasdm::AutoPasDM<ParticleType> &container,
                         const ParticlePropertiesLibraryType &particlePropertiesLibrary, const double &deltaT) {
  container.applyToOwnedParticles([&](auto &particle) { updateVelocity(particle, particlePropertiesLibrary, deltaT); });
}

void calculateVelocities(autopas::AutoPas<ParticleType> &autoPasContainer,
                         const ParticlePropertiesLibraryType &particlePropertiesLibrary, const double &deltaT) {
  AUTOPAS_OPENMP(parallel)
  for (auto iter = autoPasContainer.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
    updateVelocity(*iter, particlePropertiesLibrary, deltaT);
  }
}

void calculateAngularVelocities(autopasdm::AutoPasDM<ParticleType> &container,
                                const ParticlePropertiesLibraryType &particlePropertiesLibrary, const double &deltaT) {
#if MD_FLEXIBLE_MODE == MULTISITE
  container.applyToOwnedParticles(
      [&](auto &particle) { updateAngularVelocity(particle, particlePropertiesLibrary, deltaT); });
#else
  (void)container;
  (void)particlePropertiesLibrary;
  (void)deltaT;
  autopas::utils::ExceptionHandler::exception(
      "Attempting to perform rotational integrations when md-flexible has not been compiled with multi-site support!");
#endif
}

void calculateAngularVelocities(autopas::AutoPas<ParticleType> &autoPasContainer,
                                const ParticlePropertiesLibraryType &particlePropertiesLibrary, const double &deltaT) {
#if MD_FLEXIBLE_MODE == MULTISITE
  AUTOPAS_OPENMP(parallel)
  for (auto iter = autoPasContainer.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
    updateAngularVelocity(*iter, particlePropertiesLibrary, deltaT);
  }
#else
  (void)autoPasContainer;
  (void)particlePropertiesLibrary;
  (void)deltaT;
  autopas::utils::ExceptionHandler::exception(
      "Attempting to perform rotational integrations when md-flexible has not been compiled with multi-site support!");
#endif
}

}  // namespace TimeDiscretization
