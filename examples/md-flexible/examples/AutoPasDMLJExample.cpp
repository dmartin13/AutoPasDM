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

// This example uses single-site particles. Multisite mode is not supported.
#if MD_FLEXIBLE_MODE == MULTISITE
#error "The AutoPasDM LJ example currently supports SINGLESITE only."
#endif

int main(int argc, char **argv) {
  // Initialize the MPI runtime.
  autopasdm::Runtime runtime(argc, argv);

  // Define the global simulation parameters and the boundaries of the
  // complete simulation domain.
  constexpr double cutoff = 1.0;
  const std::array<double, 3> globalMin{0.0, 0.0, 0.0};
  const std::array<double, 3> globalMax{10.0, 10.0, 10.0};

  // Create the distributed particle system.
  // AutoPasDM manages each rank's local domain, particle distribution,
  // and halo particles.
  //
  // The lambda configures the underlying AutoPas container:
  // Direct Sum evaluates all relevant particle pairs directly. The other
  // options select the traversal, data layout, and Newton3 mode.
  autopasdm::AutoPasDM<ParticleType> autopasDM(runtime, globalMin, globalMax, cutoff, [](auto &autoPas) {
    autoPas.setAllowedContainers({autopas::ContainerOption::directSum});
    autoPas.setAllowedTraversals({autopas::TraversalOption::ds_sequential});
    autoPas.setAllowedDataLayouts({autopas::DataLayoutOption::aos});
    autoPas.setAllowedNewton3Options({autopas::Newton3Option::disabled});
    autoPas.setVerletSkin(0.0);
  });

  // Set the material properties used by the Lennard-Jones functor:
  // particle type 0 has mass 1 and LJ parameters epsilon = 1 and sigma = 1.
  // calculateMixingCoefficients() computes the required coefficients.
  ParticlePropertiesLibraryType particlePropertiesLibrary(cutoff);
  particlePropertiesLibrary.addSiteType(0, 1.0);
  particlePropertiesLibrary.addLJParametersToSite(0, 1.0, 1.0);
  particlePropertiesLibrary.calculateMixingCoefficients();

  // Get the boundaries of the subdomain assigned to this MPI rank.
  // These may differ from the global domain boundaries.
  const auto localMin = autopasDM.localBoxMin();
  const auto localMax = autopasDM.localBoxMax();

  // Add two particles per rank.
  // The ID is derived from the rank and the local particle index,
  // making it unique across ranks.
  for (int i = 0; i < 2; ++i) {
    ParticleType particle;
    particle.setID(static_cast<unsigned long>(autopasDM.rank() * 2 + i));
    particle.setTypeId(0);

    // Set the initial velocity and force values.
    // The force will be computed by the LJ functor.
    particle.setV({0.0, 0.0, 0.0});
    particle.setF({0.0, 0.0, 0.0});
    particle.setOldF({0.0, 0.0, 0.0});

    // Initially mark these particles as owned by this rank.
    particle.setOwnershipState(autopas::OwnershipState::owned);

    // Place one particle near each local x boundary.
    // Both particles are placed at the center of the domain in y and z.
    const double x = i == 0 ? localMin[0] + 0.45 : localMax[0] - 0.45;
    particle.setR({x, 5.0, 5.0});

    // Add the particle to AutoPasDM.
    autopasDM.addParticle(particle);
  }

  // The Lennard-Jones functor computes the interaction forces.
  // The template parameters configure the particle type and supported
  // Newton3 modes, among other options.
  using LJFunctor = mdLib::LJFunctor<ParticleType, true, true, autopas::FunctorN3Modes::Both, false, false>;
  LJFunctor functor(cutoff, particlePropertiesLibrary);

  // These operations define how the integrator updates MD-specific particle
  // fields, such as positions and velocities. The external force is zero here.
  MDFlexibleParticleOperations particleOperations{
      particlePropertiesLibrary,
      {0.0, 0.0, 0.0},
  };

  // Wrap the LJ functor in a force evaluator that the integrator can call.
  autopasdm::timeintegration::AutoPasForceEvaluator forceEvaluator{functor};

  // Select the Störmer-Verlet integrator for time integration.
  autopasdm::timeintegration::StormerVerlet integrator;

  // Check at compile time that the integrator, particle system,
  // particle operations, and force evaluator provide the required interface.
  static_assert(autopasdm::timeintegration::TimeIntegrator<decltype(integrator), decltype(autopasDM),
                                                           decltype(particleOperations), decltype(forceEvaluator)>);

  // Set the time step size.
  constexpr double deltaT = 1e-4;

  // Compute the initial forces required by the first Verlet step.
  // The return value can indicate, for example, whether AutoPas is tuning.
  // This small example does not handle that status further.
  (void)integrator.initialize(autopasDM, particleOperations, forceEvaluator);

  // Advance the simulation by ten time steps.
  // Each step updates the particles using Störmer-Verlet and recomputes the forces.
  for (int step = 0; step < 10; ++step) {
    (void)integrator.step(autopasDM, deltaT, particleOperations, forceEvaluator);
  }

  // Synchronize all ranks before printing the results rank by rank.
  autopasDM.barrier();

  for (int outputRank = 0; outputRank < autopasDM.numberOfRanks(); ++outputRank) {
    if (autopasDM.rank() == outputRank) {
      std::cout << "Rank " << autopasDM.rank() << " owns " << autopasDM.getLocalNumberOfOwnedParticles()
                << " particles and has " << autopasDM.getLocalNumberOfHaloParticles() << " halo particles.\n";

      // Print the position and force of every particle owned by this rank.
      std::cout << std::setprecision(12);
      autopasDM.forEachOwnedParticle([](const auto &particle) {
        const auto &r = particle.getR();
        const auto &f = particle.getF();
        std::cout << "  particle " << particle.getID() << " r=(" << r[0] << ", " << r[1] << ", " << r[2] << ") f=("
                  << f[0] << ", " << f[1] << ", " << f[2] << ")\n";
      });
    }

    // Synchronize before printing the next rank's output.
    autopasDM.barrier();
  }

  // Explicitly clean up the distributed particle system.
  autopasDM.finalize();
  return 0;
}