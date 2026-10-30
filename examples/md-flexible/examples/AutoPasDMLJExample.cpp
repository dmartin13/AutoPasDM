#include <array>
#include <iomanip>
#include <iostream>

#include "autopas/options/ContainerOption.h"
#include "autopas/options/DataLayoutOption.h"
#include "autopas/options/Newton3Option.h"
#include "autopas/options/TraversalOption.h"
#include "autopasdm/AutoPasDM.h"
#include "autopasdm/Runtime.h"
#include "autopasdm/integration/AutoPasForceEvaluator.h"
#include "autopasdm/integration/StormerVerlet.h"
#include "molecularDynamicsLibrary/LJFunctor.h"
#include "src/TypeDefinitions.h"
#include "src/distributed/MoleculeLJParticleSerializer.h"
#include "src/integration/MDFlexibleParticleOperations.h"

#if MD_FLEXIBLE_MODE == MULTISITE
#error "The AutoPasDM LJ example currently supports SINGLESITE only."
#endif

int main(int argc, char **argv) {
  autopasdm::Runtime runtime(argc, argv);

  constexpr double cutoff = 1.0;
  const std::array<double, 3> globalMin{0.0, 0.0, 0.0};
  const std::array<double, 3> globalMax{10.0, 10.0, 10.0};

  autopasdm::AutoPasDM<ParticleType> particles(runtime, globalMin, globalMax, cutoff, [](auto &autoPas) {
    autoPas.setAllowedContainers({autopas::ContainerOption::directSum});
    autoPas.setAllowedTraversals({autopas::TraversalOption::ds_sequential});
    autoPas.setAllowedDataLayouts({autopas::DataLayoutOption::aos});
    autoPas.setAllowedNewton3Options({autopas::Newton3Option::disabled});
    autoPas.setVerletSkin(0.0);
  });

  ParticlePropertiesLibraryType particlePropertiesLibrary(cutoff);
  particlePropertiesLibrary.addSiteType(0, 1.0);
  particlePropertiesLibrary.addLJParametersToSite(0, 1.0, 1.0);
  particlePropertiesLibrary.calculateMixingCoefficients();

  const auto localMin = particles.localBoxMin();
  const auto localMax = particles.localBoxMax();

  for (int i = 0; i < 2; ++i) {
    ParticleType particle;
    particle.setID(static_cast<unsigned long>(particles.rank() * 2 + i));
    particle.setTypeId(0);
    particle.setV({0.0, 0.0, 0.0});
    particle.setF({0.0, 0.0, 0.0});
    particle.setOldF({0.0, 0.0, 0.0});
    particle.setOwnershipState(autopas::OwnershipState::owned);

    const double x = i == 0 ? localMin[0] + 0.45 : localMax[0] - 0.45;
    particle.setR({x, 5.0, 5.0});
    particles.addParticle(particle);
  }

  using LJFunctor = mdLib::LJFunctor<ParticleType, true, true, autopas::FunctorN3Modes::Both, false, false>;
  LJFunctor functor(cutoff, particlePropertiesLibrary);

  MDFlexibleParticleOperations particleOperations{
      particlePropertiesLibrary,
      {0.0, 0.0, 0.0},
  };

  autopasdm::integration::AutoPasForceEvaluator forceEvaluator{functor};
  autopasdm::integration::StormerVerlet integrator;

  static_assert(autopasdm::integration::TimeIntegrator<decltype(integrator), decltype(particles),
                                                       decltype(particleOperations), decltype(forceEvaluator)>);

  constexpr double deltaT = 1e-4;

  (void)integrator.initialize(particles, particleOperations, forceEvaluator);

  for (int step = 0; step < 10; ++step) {
    (void)integrator.step(particles, deltaT, particleOperations, forceEvaluator);
  }

  particles.barrier();
  for (int outputRank = 0; outputRank < particles.numberOfRanks(); ++outputRank) {
    if (particles.rank() == outputRank) {
      std::cout << "Rank " << particles.rank() << " owns " << particles.getLocalNumberOfOwnedParticles()
                << " particles and has " << particles.getLocalNumberOfHaloParticles() << " halo particles.\n";

      std::cout << std::setprecision(12);
      particles.forEachOwnedParticle([](const auto &particle) {
        const auto &r = particle.getR();
        const auto &f = particle.getF();
        std::cout << "  particle " << particle.getID() << " r=(" << r[0] << ", " << r[1] << ", " << r[2] << ") f=("
                  << f[0] << ", " << f[1] << ", " << f[2] << ")\n";
      });
    }
    particles.barrier();
  }

  particles.finalize();
  return 0;
}
