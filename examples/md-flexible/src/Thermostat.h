/**
 * @file Thermostat.h
 * @author N. Fottner
 * @date 27/8/19
 */

#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "TypeDefinitions.h"
#include "autopas/AutoPasDecl.h"
#include "autopas/utils/ArrayMath.h"
#include "autopas/utils/WrapOpenMP.h"

/**
 * Thermostat to adjust the Temperature of the Simulation.
 */
namespace Thermostat {
namespace detail {
/**
 * Sum a particle-local quantity over owned particles.
 *
 * AutoPasDM provides this operation directly. The AutoPas fallback is
 * retained for the original md-flexible unit tests and performs a local reduction only.
 */
template <class Container, class Value, class Transform>
Value sumOwnedParticles(const Container &container, Value initialValue, Transform &&transform) {
  if constexpr (requires { container.sumOwnedParticles(initialValue, transform); }) {
    return container.sumOwnedParticles(initialValue, std::forward<Transform>(transform));
  } else {
    auto result = initialValue;
    auto &&transformRef = transform;
    AUTOPAS_OPENMP(parallel reduction(+ : result))
    for (auto iter = container.begin(); iter.isValid(); ++iter) {
      result += transformRef(*iter);
    }
    return result;
  }
}

/**
 * Execute a particle-local update on all owned particles.
 *
 * AutoPasDM owns the execution strategy in production. The AutoPas fallback
 * exists only to keep the original local thermostat tests usable.
 */
template <class Container, class Kernel>
void applyToOwnedParticles(Container &container, Kernel &&kernel) {
  if constexpr (requires { container.applyToOwnedParticles(kernel); }) {
    container.applyToOwnedParticles(std::forward<Kernel>(kernel));
  } else {
    auto &&kernelRef = kernel;
    AUTOPAS_OPENMP(parallel)
    for (auto iter = container.begin(); iter.isValid(); ++iter) {
      kernelRef(*iter);
    }
  }
}

template <class Container, class Value>
Value globalSum(const Container &container, Value localValue) {
  if constexpr (requires { container.globalSum(localValue); }) {
    return container.globalSum(localValue);
  } else {
    // Legacy AutoPas thermostat tests are local-only. Distributed communication is
    // deliberately provided only by AutoPasDM.
    return localValue;
  }
}

template <class Container>
std::size_t globalNumberOfOwnedParticles(const Container &container) {
  if constexpr (requires { container.getGlobalNumberOfOwnedParticles(); }) {
    return container.getGlobalNumberOfOwnedParticles();
  } else {
    return container.getNumberOfParticles();
  }
}
}  // namespace detail

/**
 * Calculates temperature of system.
 * Assuming dimension-less units and Boltzmann constant = 1.
 *
 * For AutoPasDM the returned temperature is global across all ranks.
 * The AutoPas fallback is intended for the original local md-flexible tests.
 *
 * @tparam Container Particle container type.
 * @tparam ParticlePropertiesLibraryTemplate Type of ParticlePropertiesLibrary Object (no pointer)
 * @param container
 * @param particlePropertiesLibrary
 * @return Temperature of system.
 */
template <class Container, class ParticlePropertiesLibraryTemplate>
double calcTemperature(const Container &container, ParticlePropertiesLibraryTemplate &particlePropertiesLibrary) {
  const auto kineticEnergyMul2Local = detail::sumOwnedParticles(container, 0.0, [&](const auto &particle) {
    const auto velocity = particle.getV();
    auto kineticEnergyMul2 =
        particlePropertiesLibrary.getMolMass(particle.getTypeId()) * autopas::utils::ArrayMath::dot(velocity, velocity);
#if MD_FLEXIBLE_MODE == MULTISITE
    const auto angularVelocity = particle.getAngularVel();
    kineticEnergyMul2 +=
        autopas::utils::ArrayMath::dot(particlePropertiesLibrary.getMomentOfInertia(particle.getTypeId()),
                                       autopas::utils::ArrayMath::mul(angularVelocity, angularVelocity));
#endif
    return kineticEnergyMul2;
  });

  constexpr unsigned int degreesOfFreedom {
#if MD_FLEXIBLE_MODE == MULTISITE
    6
#else
    3
#endif
  };

  const auto kineticEnergyMul2Global = detail::globalSum(container, kineticEnergyMul2Local);
  return kineticEnergyMul2Global /
         (static_cast<double>(detail::globalNumberOfOwnedParticles(container)) * degreesOfFreedom);
}

/**
 * Calculates temperature of system, for each component separately.
 *
 * Kinetic Energy for each molecule is
 *    1/2 * mass * dot(vel, vel) + 1/2 Sum_{0 <= i < 3} MoI_i * angVel_i^2
 * where MoI is the diagonal Moment of Inertia. This formula comes from Rapport, The Art of MD, equation (8.2.34).
 *
 * The second term is only applied for Multi-Site MD.
 *
 * Assuming dimension-less units and Boltzmann constant = 1.
 *
 * @tparam Container Particle container type.
 * @tparam ParticlePropertiesLibraryTemplate Type of ParticlePropertiesLibrary Object (no pointer)
 * @param container
 * @param particlePropertiesLibrary
 * @return map of: particle typeID -> global temperature for this type
 */
template <class Container, class ParticlePropertiesLibraryTemplate>
auto calcTemperatureComponent(const Container &container,
                              ParticlePropertiesLibraryTemplate &particlePropertiesLibrary) {
  using autopas::utils::ArrayMath::dot;
  using namespace autopas::utils::ArrayMath::literals;

  const auto numberComponents =
#if MD_FLEXIBLE_MODE == SINGLESITE
      particlePropertiesLibrary.getNumberRegisteredSiteTypes();
#elif MD_FLEXIBLE_MODE == MULTISITE
      particlePropertiesLibrary.getNumberRegisteredMolTypes();
#endif

#if MD_FLEXIBLE_MODE == MULTISITE
  constexpr unsigned int degreesOfFreedom{6};
#else
  constexpr unsigned int degreesOfFreedom{3};
#endif

  std::map<size_t, double> temperatureMap;

  for (size_t typeID = 0; typeID < numberComponents; ++typeID) {
    const auto kineticEnergyMul2Local = detail::sumOwnedParticles(container, 0.0, [&](const auto &particle) {
      if (particle.getTypeId() != typeID) {
        return 0.0;
      }

      const auto &velocity = particle.getV();
      auto kineticEnergyMul2 = particlePropertiesLibrary.getMolMass(typeID) * dot(velocity, velocity);
#if MD_FLEXIBLE_MODE == MULTISITE
      const auto &angularVelocity = particle.getAngularVel();
      kineticEnergyMul2 += dot(particlePropertiesLibrary.getMomentOfInertia(typeID), angularVelocity * angularVelocity);
#endif
      return kineticEnergyMul2;
    });

    const auto numberParticlesLocal = detail::sumOwnedParticles(
        container, size_t{0},
        [&](const auto &particle) -> size_t { return particle.getTypeId() == typeID ? 1ul : 0ul; });

    const auto kineticEnergyMul2Global = detail::globalSum(container, kineticEnergyMul2Local);
    const auto numberParticlesGlobal = detail::globalSum(container, numberParticlesLocal);

    temperatureMap[typeID] = kineticEnergyMul2Global / (static_cast<double>(numberParticlesGlobal) * degreesOfFreedom);
  }

  return temperatureMap;
}

/**
 * Adds brownian motion to the given system.
 *
 * This is achieved by each degree-of-freedom for the Kinetic Energy being sampled via the normal distribution and
 * scaled appropriately, as determined by the equipartition theorem.
 *
 * For multi-site MD we assume that the kinetic energy of the system can be split equally into translational and
 * rotational kinetic energies.
 *
 * In all cases, we assume a Boltzmann constant of 1.
 *
 * @tparam Container Particle container type.
 * @tparam ParticlePropertiesLibraryTemplate Type of ParticlePropertiesLibrary Object (no pointer)
 * @param container
 * @param particlePropertiesLibrary
 * @param targetTemperature temperature of the system after applying the function on a system with temperature = 0.
 */
template <class Container, class ParticlePropertiesLibraryTemplate>
void addBrownianMotion(Container &container, ParticlePropertiesLibraryTemplate &particlePropertiesLibrary,
                       const double targetTemperature) {
  using namespace autopas::utils::ArrayMath::literals;

  std::map<size_t, double> translationalVelocityScale;
  std::map<size_t, std::array<double, 3>> rotationalVelocityScale;

  const auto numberComponents =
#if MD_FLEXIBLE_MODE == SINGLESITE
      particlePropertiesLibrary.getNumberRegisteredSiteTypes();
#elif MD_FLEXIBLE_MODE == MULTISITE
      particlePropertiesLibrary.getNumberRegisteredMolTypes();
#endif

  for (size_t typeID = 0; typeID < numberComponents; ++typeID) {
    translationalVelocityScale.emplace(typeID,
                                       std::sqrt(targetTemperature / particlePropertiesLibrary.getMolMass(typeID)));
#if MD_FLEXIBLE_MODE == MULTISITE
    const auto momentOfInertia = particlePropertiesLibrary.getMomentOfInertia(typeID);
    rotationalVelocityScale.emplace(typeID, std::array<double, 3>{std::sqrt(targetTemperature / momentOfInertia[0]),
                                                                  std::sqrt(targetTemperature / momentOfInertia[1]),
                                                                  std::sqrt(targetTemperature / momentOfInertia[2])});
#endif
  }

  const auto maxThreads = static_cast<size_t>(autopas::autopas_get_max_threads());
  std::vector<std::default_random_engine> randomEngines;
  std::vector<std::normal_distribution<double>> normalDistributions;
  randomEngines.reserve(maxThreads);
  normalDistributions.reserve(maxThreads);
  for (size_t thread = 0; thread < maxThreads; ++thread) {
    randomEngines.emplace_back(42 + thread);
    normalDistributions.emplace_back(0.0, 1.0);
  }

  detail::applyToOwnedParticles(container, [&](auto &particle) {
    const auto thread = static_cast<size_t>(autopas::autopas_get_thread_num());
    auto &randomEngine = randomEngines[thread];
    auto &normalDistribution = normalDistributions[thread];

    const std::array<double, 3> normal3DVecTranslational = {
        normalDistribution(randomEngine), normalDistribution(randomEngine), normalDistribution(randomEngine)};
    particle.addV(normal3DVecTranslational * translationalVelocityScale.at(particle.getTypeId()));
#if MD_FLEXIBLE_MODE == MULTISITE
    const std::array<double, 3> normal3DVecRotational = {
        normalDistribution(randomEngine), normalDistribution(randomEngine), normalDistribution(randomEngine)};
    particle.addAngularVel(normal3DVecRotational * rotationalVelocityScale.at(particle.getTypeId()));
#endif
  });
}

/**
 * Scales velocity of particles towards a given temperature. For Multi-site simulations, angular velocity is also
 * scaled.
 *
 * @tparam Container Particle container type.
 * @tparam ParticlePropertiesLibraryTemplate Type of ParticlePropertiesLibrary Object (no pointer)
 * @param container
 * @param particlePropertiesLibrary
 * @param targetTemperature
 * @param deltaTemperature Maximum temperature change.
 */
template <class Container, class ParticlePropertiesLibraryTemplate>
void apply(Container &container, ParticlePropertiesLibraryTemplate &particlePropertiesLibrary,
           const double targetTemperature, const double deltaTemperature) {
  using namespace autopas::utils::ArrayMath::literals;

  AutoPasLog(DEBUG, "Applying Thermostat");

  const auto currentTemperatureMap = calcTemperatureComponent(container, particlePropertiesLibrary);
  const double absoluteDeltaTemperature = std::abs(deltaTemperature);

  std::remove_const_t<decltype(currentTemperatureMap)> scalingMap;
  for (const auto &[particleTypeID, currentTemperature] : currentTemperatureMap) {
    const auto immediateTargetTemperature =
        currentTemperature < targetTemperature
            ? std::min(currentTemperature + absoluteDeltaTemperature, targetTemperature)
            : std::max(currentTemperature - absoluteDeltaTemperature, targetTemperature);

    scalingMap[particleTypeID] = std::sqrt(immediateTargetTemperature / currentTemperature);

    AutoPasLog(DEBUG, "Current temperature of typeID {}: {}", particleTypeID, currentTemperature);
    AutoPasLog(DEBUG, "Temperature of typeID {} after application of thermostat: {}", particleTypeID,
               immediateTargetTemperature);
  }

  detail::applyToOwnedParticles(container, [&](auto &particle) {
    particle.setV(particle.getV() * scalingMap.at(particle.getTypeId()));
#if MD_FLEXIBLE_MODE == MULTISITE
    particle.setAngularVel(particle.getAngularVel() * scalingMap.at(particle.getTypeId()));
#endif
  });

  [[maybe_unused]] const auto currentTemperatures = calcTemperatureComponent(container, particlePropertiesLibrary);
}
}  // namespace Thermostat
