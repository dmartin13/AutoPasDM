#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

#include "MoleculeLJSerializer.h"
#include "autopas/baseFunctors/Functor.h"
#include "autopasdm/AutoPasDM.h"
#include "autopasdm/Runtime.h"
#include "molecularDynamicsLibrary/LJFunctor.h"
#include "molecularDynamicsLibrary/MoleculeLJ.h"

int main(int argc, char **argv) {
  autopasdm::Runtime runtime(argc, argv);

  using Particle = mdLib::MoleculeLJ;
  constexpr double cutoff = 2.5;
  constexpr double timeStep = 0.002;
  constexpr double mass = 1.0;
  constexpr std::size_t steps = 20;
  constexpr std::size_t particlesPerDimension = 4;
  const std::array<double, 3> boxMin{0.0, 0.0, 0.0};
  const std::array<double, 3> boxMax{8.0, 8.0, 8.0};

  autopasdm::AutoPasDM<Particle> particles(runtime, boxMin, boxMax, cutoff, std::array<bool, 3>{true, true, true});

  std::vector<Particle> initialParticles;
  if (particles.isRoot()) {
    const double spacing = (boxMax[0] - boxMin[0]) / static_cast<double>(particlesPerDimension);
    unsigned long id = 0;
    for (std::size_t x = 0; x < particlesPerDimension; ++x) {
      for (std::size_t y = 0; y < particlesPerDimension; ++y) {
        for (std::size_t z = 0; z < particlesPerDimension; ++z) {
          const std::array<double, 3> position{boxMin[0] + (static_cast<double>(x) + 0.5) * spacing,
                                               boxMin[1] + (static_cast<double>(y) + 0.5) * spacing,
                                               boxMin[2] + (static_cast<double>(z) + 0.5) * spacing};
          const std::array<double, 3> velocity{x % 2 == 0 ? 0.05 : -0.05, 0.0, 0.0};
          initialParticles.emplace_back(position, velocity, id++);
        }
      }
    }
  }
  particles.addParticlesFromRoot(initialParticles);

  using LJFunctor = mdLib::LJFunctor<Particle, true, false, autopas::FunctorN3Modes::Both>;
  LJFunctor ljFunctor(cutoff);
  ljFunctor.setParticleProperties(24.0, 1.0);  // epsilon = sigma = 1

  const auto computeForces = [&]() {
    particles.applyToOwnedParticles([](auto &particle) { particle.setF({0.0, 0.0, 0.0}); });
    particles.computeInteractions(&ljFunctor);
  };

  computeForces();
  for (std::size_t step = 0; step < steps; ++step) {
    particles.applyToOwnedParticles([=](auto &particle) {
      auto r = particle.getR();
      auto v = particle.getV();
      const auto &f = particle.getF();
      for (std::size_t d = 0; d < 3; ++d) {
        v[d] += 0.5 * timeStep * f[d] / mass;
        r[d] += timeStep * v[d];
      }
      particle.setR(r);
      particle.setV(v);
    });

    computeForces();

    particles.applyToOwnedParticles([=](auto &particle) {
      auto v = particle.getV();
      const auto &f = particle.getF();
      for (std::size_t d = 0; d < 3; ++d) {
        v[d] += 0.5 * timeStep * f[d] / mass;
      }
      particle.setV(v);
    });
  }

  const auto globalParticleCount = particles.getGlobalNumberOfOwnedParticles();
  if (particles.isRoot()) {
    std::cout << "Finished " << steps << " LJ steps with " << globalParticleCount << " particles on "
              << particles.numberOfRanks() << " ranks.\n";
  }

  particles.finalize();
}
