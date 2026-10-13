#include <array>
#include <iostream>

#include "autopas/options/ContainerOption.h"
#include "autopas/options/DataLayoutOption.h"
#include "autopas/options/Newton3Option.h"
#include "autopas/options/TraversalOption.h"
#include "autopasdm/AutoPasDM.h"
#include "autopasdm/PairCountingFunctor.h"
#include "autopasdm/Particle.h"
#include "autopasdm/Runtime.h"

int main(int argc, char **argv) {
  autopasdm::Runtime runtime(argc, argv);

  const std::array<double, 3> globalMin{0.0, 0.0, 0.0};
  const std::array<double, 3> globalMax{10.0, 10.0, 10.0};
  constexpr double cutoff = 0.5;

  autopasdm::AutoPasDM<autopasdm::Particle> particles(runtime, globalMin, globalMax, cutoff, [](auto &autoPas) {
    autoPas.setAllowedContainers({autopas::ContainerOption::directSum});
    autoPas.setAllowedTraversals({autopas::TraversalOption::ds_sequential});
    autoPas.setAllowedDataLayouts({autopas::DataLayoutOption::aos});
    autoPas.setAllowedNewton3Options({autopas::Newton3Option::enabled});
    autoPas.setVerletSkin(0.0);
  });

  for (int i = 0; i < 10; ++i) {
    autopasdm::Particle particle;
    particle.setID(particles.rank() * 1000 + i);

    const double x0 = particles.localBoxMin()[0];
    const double x1 = particles.localBoxMax()[0];
    particle.setR({x0 + (x1 - x0) * (i + 0.5) / 10.0, 5.0, 5.0});
    particles.addParticle(particle);
  }

  particles.applyToOwnedParticles([](auto &particle) {
    auto position = particle.getR();
    position[0] += 1.0;
    particle.setR(position);
  });

  autopasdm::PairCountingFunctor<autopasdm::Particle> pairCounter(cutoff, particles.localBoxMin(),
                                                                  particles.localBoxMax());
  particles.computeInteractions(&pairCounter);

  particles.barrier();
  for (int outputRank = 0; outputRank < particles.numberOfRanks(); ++outputRank) {
    if (particles.rank() == outputRank) {
      std::cout << "Rank " << particles.rank() << " owns " << particles.getLocalNumberOfOwnedParticles()
                << " particles and has " << particles.getLocalNumberOfHaloParticles() << " halo particles.\n";

      std::cout << "  owned particle ids: ";
      particles.forEachOwnedParticle([](const auto &particle) { std::cout << particle.getID() << " "; });
      std::cout << "\n";

      std::cout << "  interactions within cutoff: " << pairCounter.numPairs() << "\n";
      std::cout << "    owned-owned: " << pairCounter.numOwnedOwnedPairs() << "\n";
      std::cout << "    owned-halo:  " << pairCounter.numOwnedHaloPairs() << "\n";
      std::cout << "    halo-halo:   " << pairCounter.numHaloHaloPairs() << "\n";

      std::cout << "  owned-halo pairs: ";
      for (const auto &[idA, idB] : pairCounter.ownedHaloPairs()) {
        std::cout << "(" << idA << "," << idB << ") ";
      }
      std::cout << "\n";
    }
    particles.barrier();
  }

  particles.finalize();
  return 0;
}
