#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "TestForceFunctor.h"
#include "autopas/options/ContainerOption.h"
#include "autopas/options/DataLayoutOption.h"
#include "autopas/options/Newton3Option.h"
#include "autopas/options/TraversalOption.h"
#include "autopasdm/AutoPasDM.h"
#include "autopasdm/Particle.h"
#include "autopasdm/Runtime.h"

namespace {

using Particle = autopasdm::Particle;

constexpr std::array<double, 3> globalMin{0., 0., 0.};
constexpr std::array<double, 3> globalMax{4., 1., 1.};
constexpr double cutoff = 0.2;

Particle makeParticle(unsigned long id, double x) {
  Particle particle;
  particle.setID(id);
  particle.setR({x, 0.5, 0.5});
  return particle;
}

auto makeConfigurator() {
  return [](auto &autoPas) {
    autoPas.setAllowedContainers({autopas::ContainerOption::directSum});
    autoPas.setAllowedTraversals({autopas::TraversalOption::ds_sequential});
    autoPas.setAllowedDataLayouts({autopas::DataLayoutOption::aos});
    autoPas.setAllowedNewton3Options({autopas::Newton3Option::enabled});
    autoPas.setVerletSkin(0.0);
  };
}

std::vector<unsigned long> ownedIds(const autopasdm::AutoPasDM<Particle> &particles) {
  std::vector<unsigned long> ids;
  particles.forEachOwnedParticle([&](const auto &particle) { ids.push_back(particle.getID()); });
  std::sort(ids.begin(), ids.end());
  return ids;
}

constexpr std::array<double, 3> forceGlobalMin{0., 0., 0.};
constexpr std::array<double, 3> forceGlobalMax{8.8, 2., 2.};
constexpr double forceCutoff = 1.5;

std::vector<Particle> makeForceTestParticles() {
  constexpr std::array<double, 8> positions{0.4, 1.4, 2.7, 3.7, 4.9, 5.9, 7.2, 8.2};

  std::vector<Particle> particles;
  particles.reserve(positions.size());
  for (std::size_t id = 0; id < positions.size(); ++id) {
    auto particle = makeParticle(static_cast<unsigned long>(id), positions[id]);
    particle.setF({0., 0., 0.});
    particles.push_back(particle);
  }
  return particles;
}

void expectForceTestResult(int expectedNumberOfRanks, bool explicitPreparation = false) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), expectedNumberOfRanks);

  autopasdm::AutoPasDM<Particle> particles(runtime, forceGlobalMin, forceGlobalMax, forceCutoff, makeConfigurator());
  particles.addParticlesFromRoot(makeForceTestParticles());

  autopasdm::testing::TestForceFunctor<Particle> functor(forceCutoff);
  if (explicitPreparation) {
    particles.prepareInteractions();
    particles.computeInteractionsPrepared(&functor);
  } else {
    particles.computeInteractions(&functor);
  }

  constexpr std::array<double, 8> expectedForceX{0.0, 0.3, -0.3, 0.2, -0.2, 0.3, -0.3, 0.0};

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 8);
  EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 8 / expectedNumberOfRanks);

  std::size_t checkedParticles = 0;
  particles.forEachOwnedParticle([&](const auto &particle) {
    ASSERT_LT(particle.getID(), expectedForceX.size());
    const auto &force = particle.getF();
    EXPECT_NEAR(force[0], expectedForceX[particle.getID()], 1e-12) << "particle id " << particle.getID();
    EXPECT_NEAR(force[1], 0.0, 1e-12) << "particle id " << particle.getID();
    EXPECT_NEAR(force[2], 0.0, 1e-12) << "particle id " << particle.getID();
    ++checkedParticles;
  });

  EXPECT_EQ(checkedParticles, 8 / expectedNumberOfRanks);
  particles.finalize();
}

TEST(AutoPasDMTest, AddsReplicatedInputFromRootExactlyOnce) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 4);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  std::vector<Particle> replicatedInput;
  for (unsigned long id = 0; id < 8; ++id) {
    const auto owner = static_cast<int>(id / 2);
    const auto offsetWithinDomain = id % 2 == 0 ? 0.25 : 0.75;
    replicatedInput.push_back(makeParticle(id, static_cast<double>(owner) + offsetWithinDomain));
  }

  // Deliberately pass the same replicated collection on every rank. Only root's
  // collection may be considered by addParticlesFromRoot().
  particles.addParticlesFromRoot(replicatedInput);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 8);
  EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 2);

  const auto firstExpectedId = static_cast<unsigned long>(2 * runtime.rank());
  EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{firstExpectedId, firstExpectedId + 1}));

  particles.finalize();
}

TEST(AutoPasDMTest, RedistributesContributionsFromAllRanksToTheirOwners) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 4);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  const auto rank = runtime.rank();
  const auto rightRank = (rank + 1) % runtime.size();

  // Every rank contributes one particle that already belongs locally and one
  // particle whose owning rank is its right neighbor.
  const std::vector<Particle> localInput{
      makeParticle(static_cast<unsigned long>(200 + rank), static_cast<double>(rank) + 0.75),
      makeParticle(static_cast<unsigned long>(100 + rank), static_cast<double>(rightRank) + 0.25),
  };

  particles.addDistributedParticles(localInput);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 8);
  EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 2);

  const auto leftRank = (rank + runtime.size() - 1) % runtime.size();
  auto expectedIds =
      std::vector<unsigned long>{static_cast<unsigned long>(200 + rank), static_cast<unsigned long>(100 + leftRank)};
  std::sort(expectedIds.begin(), expectedIds.end());
  EXPECT_EQ(ownedIds(particles), expectedIds);

  particles.finalize();
}

TEST(AutoPasDMTest, UsesThreeDimensionalProcessGridForInitialOwnership) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 8);

  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{4., 4., 4.};
  constexpr std::array<bool, 3> subdivideDimensions{true, true, true};

  autopasdm::AutoPasDM<Particle> particles(runtime, boxMin, boxMax, cutoff, subdivideDimensions, makeConfigurator());

  std::vector<Particle> replicatedInput;
  for (int x = 0; x < 2; ++x) {
    for (int y = 0; y < 2; ++y) {
      for (int z = 0; z < 2; ++z) {
        const auto owner = static_cast<unsigned long>(x * 4 + y * 2 + z);
        Particle particle;
        particle.setID(owner);
        particle.setR({1. + 2. * x, 1. + 2. * y, 1. + 2. * z});
        replicatedInput.push_back(particle);
      }
    }
  }

  particles.addParticlesFromRoot(replicatedInput);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 8);
  EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 1);
  EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{static_cast<unsigned long>(runtime.rank())}));

  const int rank = runtime.rank();
  const int x = rank / 4;
  const int y = (rank % 4) / 2;
  const int z = rank % 2;

  EXPECT_EQ(particles.localBoxMin(), (std::array<double, 3>{2. * x, 2. * y, 2. * z}));
  EXPECT_EQ(particles.localBoxMax(), (std::array<double, 3>{2. * (x + 1), 2. * (y + 1), 2. * (z + 1)}));

  particles.finalize();
}

TEST(AutoPasDMTest, ResizesLocalBoxAndMigratesParticles) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 4);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  particles.addParticlesFromRoot(std::vector<Particle>{
      makeParticle(0, 0.9),
      makeParticle(1, 1.1),
      makeParticle(2, 1.5),
      makeParticle(3, 2.5),
      makeParticle(4, 3.5),
  });

  // Particle 0 leaves rank 0's old box but lies inside its expanded new box.
  // This exercises emigrants returned by updateContainer() that become local again.
  particles.applyToOwnedParticles([](auto &particle) {
    if (particle.getID() == 0) {
      particle.setR({1.1, 0.5, 0.5});
    }
  });

  auto newLocalMin = particles.localBoxMin();
  auto newLocalMax = particles.localBoxMax();
  if (runtime.rank() == 0) {
    newLocalMax[0] = 1.2;
  } else if (runtime.rank() == 1) {
    newLocalMin[0] = 1.2;
  }

  particles.prepareInteractions(newLocalMin, newLocalMax);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 5);
  EXPECT_EQ(particles.localBoxMin(), newLocalMin);
  EXPECT_EQ(particles.localBoxMax(), newLocalMax);

  if (runtime.rank() == 0) {
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{0, 1}));
  } else if (runtime.rank() == 1) {
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{2}));
  } else if (runtime.rank() == 2) {
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{3}));
  } else {
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{4}));
  }

  particles.finalize();
}

TEST(AutoPasDMTest, AppliesInvertedPressureLoadBalancingAndMigratesParticles) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 2);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  particles.addParticlesFromRoot(std::vector<Particle>{
      makeParticle(0, 0.5),
      makeParticle(1, 1.75),
      makeParticle(2, 2.25),
      makeParticle(3, 3.5),
  });

  // Rank 0 reports three times as much work as rank 1. The theoretically balanced
  // shared boundary is x=1.0 and the damped inverted-pressure step moves it from
  // x=2.0 to x=1.5. Particle 1 must therefore migrate from rank 0 to rank 1.
  const double localWork = runtime.rank() == 0 ? 3. : 1.;
  particles.prepareInteractions(localWork);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 4);
  if (runtime.rank() == 0) {
    EXPECT_DOUBLE_EQ(particles.localBoxMin()[0], 0.);
    EXPECT_DOUBLE_EQ(particles.localBoxMax()[0], 1.5);
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{0}));
  } else {
    EXPECT_DOUBLE_EQ(particles.localBoxMin()[0], 1.5);
    EXPECT_DOUBLE_EQ(particles.localBoxMax()[0], 4.);
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{1, 2, 3}));
  }

  particles.finalize();
}

TEST(AutoPasDMTest, RedistributesParticlesUsingAdaptiveTargetRanks) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 2);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  // Move the shared boundary from x=2.0 to x=1.5 before inserting particles.
  const double localWork = runtime.rank() == 0 ? 3. : 1.;
  particles.prepareInteractions(localWork);

  EXPECT_DOUBLE_EQ(runtime.rank() == 0 ? particles.localBoxMax()[0] : particles.localBoxMin()[0], 1.5);

  // x=1.75 belonged to rank 0 in the initial uniform grid, but belongs to rank 1
  // after load balancing. This therefore exercises targetRank() with adaptive splits.
  particles.addParticlesFromRoot(std::vector<Particle>{makeParticle(42, 1.75)});

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 1);
  if (runtime.rank() == 0) {
    EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 0);
  } else {
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{42}));
  }

  particles.finalize();
}

TEST(AutoPasDMTest, AppliesRepeatedInvertedPressureLoadBalancingFromCurrentDomain) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 2);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  particles.addParticlesFromRoot(std::vector<Particle>{
      makeParticle(0, 0.5),
      makeParticle(1, 1.4),
      makeParticle(2, 1.75),
      makeParticle(3, 2.25),
      makeParticle(4, 3.5),
  });

  const double localWork = runtime.rank() == 0 ? 3. : 1.;

  // The first balancing step moves the shared boundary from x=2.0 to x=1.5.
  // Particle 2 therefore migrates from rank 0 to rank 1.
  particles.prepareInteractions(localWork);

  if (runtime.rank() == 0) {
    EXPECT_DOUBLE_EQ(particles.localBoxMax()[0], 1.5);
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{0, 1}));
  } else {
    EXPECT_DOUBLE_EQ(particles.localBoxMin()[0], 1.5);
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{2, 3, 4}));
  }

  // With the same work imbalance, the theoretical boundary is still x=1.0.
  // The second damped update must start from the current x=1.5 boundary and
  // therefore move it to x=1.25, rather than restarting from the initial grid.
  // Particle 1 then migrates during this second balancing step.
  particles.prepareInteractions(localWork);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 5);
  if (runtime.rank() == 0) {
    EXPECT_DOUBLE_EQ(particles.localBoxMin()[0], 0.);
    EXPECT_DOUBLE_EQ(particles.localBoxMax()[0], 1.25);
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{0}));
  } else {
    EXPECT_DOUBLE_EQ(particles.localBoxMin()[0], 1.25);
    EXPECT_DOUBLE_EQ(particles.localBoxMax()[0], 4.);
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{1, 2, 3, 4}));
  }

  particles.finalize();
}

TEST(AutoPasDMTest, BalancesMigratesAndComputesAcrossAdaptiveThreeDimensionalCorner) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 8);

  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{4., 4., 4.};
  constexpr std::array<bool, 3> subdivideDimensions{true, true, true};
  constexpr double threeDimensionalCutoff = 0.5;

  autopasdm::AutoPasDM<Particle> particles(runtime, boxMin, boxMax, threeDimensionalCutoff, subdivideDimensions,
                                           makeConfigurator());

  Particle lowerCornerParticle;
  lowerCornerParticle.setID(8000);
  lowerCornerParticle.setR({1.4, 1.4, 1.4});
  lowerCornerParticle.setF({0., 0., 0.});

  Particle upperCornerParticle;
  upperCornerParticle.setID(8001);
  upperCornerParticle.setR({1.6, 1.6, 1.6});
  upperCornerParticle.setF({0., 0., 0.});

  // Both particles initially belong to rank 0. The work distribution below gives
  // every lower process plane three times the average work of the corresponding
  // upper plane. In every dimension, inverted pressure therefore moves the shared
  // boundary from 2.0 to 1.5. Particle 8001 then migrates across x, y, and z to
  // rank 7, while particle 8000 remains on rank 0.
  particles.addParticlesFromRoot(std::vector<Particle>{lowerCornerParticle, upperCornerParticle});

  const int rank = runtime.rank();
  const int x = rank / 4;
  const int y = (rank % 4) / 2;
  const int z = rank % 2;
  const double localWork = (x == 0 ? 3. : 1.) * (y == 0 ? 3. : 1.) * (z == 0 ? 3. : 1.);

  particles.prepareInteractions(localWork);

  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    const int coordinate = dimension == 0 ? x : (dimension == 1 ? y : z);
    EXPECT_DOUBLE_EQ(particles.localBoxMin()[dimension], coordinate == 0 ? 0. : 1.5);
    EXPECT_DOUBLE_EQ(particles.localBoxMax()[dimension], coordinate == 0 ? 1.5 : 4.);
  }

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 2);
  if (rank == 0) {
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{8000}));
  } else if (rank == 7) {
    EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{8001}));
  } else {
    EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 0);
  }

  // The particles are now owned by diagonally opposite subdomains and interact
  // across the newly shifted x/y/z corner. A correct force therefore also verifies
  // that halo propagation uses the adaptive box geometry after load balancing.
  autopasdm::testing::TestForceFunctor<Particle> functor(threeDimensionalCutoff);
  particles.computeInteractionsPrepared(&functor);

  if (rank == 0) {
    particles.forEachOwnedParticle([](const auto &particle) {
      EXPECT_EQ(particle.getID(), 8000);
      EXPECT_NEAR(particle.getF()[0], 0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[1], 0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[2], 0.2, 1e-12);
    });
  } else if (rank == 7) {
    particles.forEachOwnedParticle([](const auto &particle) {
      EXPECT_EQ(particle.getID(), 8001);
      EXPECT_NEAR(particle.getF()[0], -0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[1], -0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[2], -0.2, 1e-12);
    });
  }

  particles.finalize();
}

TEST(AutoPasDMTest, ReducesDoubleValuesGlobally) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 4);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  const double localValue = 0.25 * static_cast<double>(runtime.rank() + 1);
  EXPECT_DOUBLE_EQ(particles.globalSum(localValue), 2.5);

  particles.finalize();
}

TEST(AutoPasDMTest, AppliesKernelOnlyToOwnedParticlesInRegion) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 1);

  autopasdm::AutoPasDM<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  auto firstParticle = makeParticle(0, 0.25);
  auto secondParticle = makeParticle(1, 1.25);
  auto thirdParticle = makeParticle(2, 3.75);
  firstParticle.setF({0., 0., 0.});
  secondParticle.setF({0., 0., 0.});
  thirdParticle.setF({0., 0., 0.});
  particles.addParticlesFromRoot(std::vector<Particle>{firstParticle, secondParticle, thirdParticle});

  particles.applyToOwnedParticlesInRegion({-1., 0., 0.}, {1., 1., 1.}, [](auto &particle) {
    particle.setF({42., 0., 0.});
  });

  particles.forEachOwnedParticle([](const auto &particle) {
    if (particle.getID() == 0) {
      EXPECT_DOUBLE_EQ(particle.getF()[0], 42.);
    } else {
      EXPECT_DOUBLE_EQ(particle.getF()[0], 0.);
    }
  });

  particles.finalize();
}

TEST(AutoPasDMTest, ComputesInteractionAcrossThreeDimensionalCorner) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 8);

  constexpr std::array<double, 3> boxMin{0., 0., 0.};
  constexpr std::array<double, 3> boxMax{4., 4., 4.};
  constexpr std::array<bool, 3> subdivideDimensions{true, true, true};
  constexpr double threeDimensionalCutoff = 0.5;

  autopasdm::AutoPasDM<Particle> particles(runtime, boxMin, boxMax, threeDimensionalCutoff, subdivideDimensions,
                                           makeConfigurator());

  Particle lowerCornerParticle;
  lowerCornerParticle.setID(7000);
  lowerCornerParticle.setR({1.9, 1.9, 1.9});
  lowerCornerParticle.setF({0., 0., 0.});

  Particle upperCornerParticle;
  upperCornerParticle.setID(7001);
  upperCornerParticle.setR({2.1, 2.1, 2.1});
  upperCornerParticle.setF({0., 0., 0.});

  particles.addParticlesFromRoot(std::vector<Particle>{lowerCornerParticle, upperCornerParticle});

  autopasdm::testing::TestForceFunctor<Particle> functor(threeDimensionalCutoff);
  particles.computeInteractions(&functor);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 2);

  if (runtime.rank() == 0) {
    ASSERT_EQ(particles.getLocalNumberOfOwnedParticles(), 1);
    particles.forEachOwnedParticle([](const auto &particle) {
      EXPECT_EQ(particle.getID(), 7000);
      EXPECT_NEAR(particle.getF()[0], 0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[1], 0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[2], 0.2, 1e-12);
    });
  } else if (runtime.rank() == 7) {
    ASSERT_EQ(particles.getLocalNumberOfOwnedParticles(), 1);
    particles.forEachOwnedParticle([](const auto &particle) {
      EXPECT_EQ(particle.getID(), 7001);
      EXPECT_NEAR(particle.getF()[0], -0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[1], -0.2, 1e-12);
      EXPECT_NEAR(particle.getF()[2], -0.2, 1e-12);
    });
  } else {
    EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 0);
  }

  particles.finalize();
}

TEST(AutoPasDMTest, ComputesPeriodicForcesWithSingleRank) { expectForceTestResult(1); }

TEST(AutoPasDMTest, ComputesSamePeriodicForcesAcrossFourRanks) { expectForceTestResult(4); }

TEST(AutoPasDMTest, ComputesForcesAfterExplicitPreparation) { expectForceTestResult(4, true); }

}  // namespace
