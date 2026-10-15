/**
 * @file Simulation.cpp
 * @author F. Gratl
 * @date 01.03.2021
 */
#include "Simulation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "TypeDefinitions.h"
#include "autopas/AutoPasDecl.h"
#include "autopas/utils/WrapOpenMP.h"

// Declare the main AutoPas class and the computeInteractions() methods with all used functors as extern template
// instantiation. They are instantiated in the respective cpp file inside the templateInstantiations folder.
//! @cond Doxygen_Suppress
extern template class autopas::AutoPas<ParticleType>;
#if defined(MD_FLEXIBLE_FUNCTOR_AUTOVEC)
extern template bool autopas::AutoPas<ParticleType>::computeInteractions(LJFunctorTypeAutovec *);
#endif
#if defined(MD_FLEXIBLE_FUNCTOR_AVX) && defined(__AVX__)
extern template bool autopas::AutoPas<ParticleType>::computeInteractions(LJFunctorTypeAVX *);
#endif
#if defined(MD_FLEXIBLE_FUNCTOR_SVE) && defined(__ARM_FEATURE_SVE)
extern template bool autopas::AutoPas<ParticleType>::computeInteractions(LJFunctorTypeSVE *);
#endif
#if defined(MD_FLEXIBLE_FUNCTOR_AT_AUTOVEC)
extern template bool autopas::AutoPas<ParticleType>::computeInteractions(ATFunctor *);
#endif
#if defined(MD_FLEXIBLE_FUNCTOR_HWY)
extern template bool autopas::AutoPas<ParticleType>::computeInteractions(LJFunctorTypeHWY *);
#endif
//! @endcond

#include <sys/ioctl.h>
#include <unistd.h>

#include <iostream>

#include "ReflectiveBoundary.h"
#include "Thermostat.h"
#include "TimeDiscretization.h"
#include "autopas/utils/MemoryProfiler.h"
#include "configuration/MDFlexConfig.h"

namespace {
/**
 * Tries to identify the width of the terminal where the simulation is running.
 * If no width can be identified, the function defaults to 80.
 * @return width of the terminal.
 */
size_t getTerminalWidth() {
  size_t terminalWidth = 0;
  // test all std pipes to get the current terminal width
  for (auto fd : {STDOUT_FILENO, STDIN_FILENO, STDERR_FILENO}) {
    if (isatty(fd)) {
      struct winsize w {};
      ioctl(fd, TIOCGWINSZ, &w);
      terminalWidth = w.ws_col;
      break;
    }
  }

  // if width is still zero try the environment variable COLUMNS
  if (terminalWidth == 0) {
    if (auto *terminalWidthCharArr = std::getenv("COLUMNS")) {
      // this pointer could be used to detect parsing errors via terminalWidthCharArr == end
      // but since we have a fallback further down we are ok if this fails silently.
      char *end{};
      terminalWidth = std::strtol(terminalWidthCharArr, &end, 10);
    }
  }

  // if all of the above fail fall back to a fixed width
  if (terminalWidth == 0) {
    // this seems to be the default width in most terminal windows
    terminalWidth = 80;
  }

  return terminalWidth;
}

std::array<autopasdm::BoundaryType, 3> toDistributedBoundaryTypes(
    const std::array<options::BoundaryTypeOption, 3> &boundaryTypes) {
  std::array<autopasdm::BoundaryType, 3> result{};

  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    switch (static_cast<options::BoundaryTypeOption::Value>(boundaryTypes[dimension])) {
      case options::BoundaryTypeOption::periodic:
        result[dimension] = autopasdm::BoundaryType::periodic;
        break;
      case options::BoundaryTypeOption::reflective:
        result[dimension] = autopasdm::BoundaryType::reflective;
        break;
      case options::BoundaryTypeOption::none:
        result[dimension] = autopasdm::BoundaryType::none;
        break;
    }
  }

  return result;
}
}  // namespace

Simulation::Simulation(const MDFlexConfig &configuration, autopasdm::Runtime &runtime)
    : _configuration(configuration), _totalEnergySensor(configuration.energySensorOption.value) {
  _timers.total.start();
  _timers.initialization.start();
  _totalEnergySensor.startMeasurement();

  // only create the writer if necessary since this also creates the output dir
  if (not configuration.vtkFileName.value.empty()) {
    _vtkWriter.emplace(_configuration.vtkFileName.value, _configuration.vtkOutputFolder.value,
                       std::to_string(_configuration.iterations.value).size(), runtime);
  }

  const auto rank = runtime.rank();
  const auto *fillerBeforeSuffix =
      _configuration.outputSuffix.value.empty() or _configuration.outputSuffix.value.front() == '_' ? "" : "_";
  const auto *fillerAfterSuffix =
      _configuration.outputSuffix.value.empty() or _configuration.outputSuffix.value.back() == '_' ? "" : "_";
  const auto outputSuffix =
      "Rank" + std::to_string(rank) + fillerBeforeSuffix + _configuration.outputSuffix.value + fillerAfterSuffix;

  if (rank == 0) {
    _globalLogger = std::make_unique<GlobalVariableLogger>(outputSuffix);
  }

  if (_configuration.logFileName.value.empty()) {
    _outputStream = &std::cout;
  } else {
    _logFile = std::make_shared<std::ofstream>();
    _logFile->open(_configuration.logFileName.value + "_" + outputSuffix);
    _outputStream = &(*_logFile);
  }

  if (_configuration.getInteractionTypes().empty()) {
    std::string functorName{};
    std::tie(_configuration.functorOption.value, functorName) =
#if defined(MD_FLEXIBLE_FUNCTOR_HWY)
        std::make_pair(MDFlexConfig::FunctorOption::lj12_6_HWY, "Lennard-Jones HWY Functor.");
#elif defined(MD_FLEXIBLE_FUNCTOR_AVX) && defined(__AVX__)
        std::make_pair(MDFlexConfig::FunctorOption::lj12_6_AVX, "Lennard-Jones AVX Functor.");
#elif defined(MD_FLEXIBLE_FUNCTOR_SVE) && defined(__ARM_FEATURE_SVE)
        std::make_pair(MDFlexConfig::FunctorOption::lj12_6_SVE, "Lennard-Jones SVE Functor.");
#elif defined(MD_FLEXIBLE_FUNCTOR_AUTOVEC)
        std::make_pair(MDFlexConfig::FunctorOption::lj12_6, "Lennard-Jones AutoVec Functor.");
#else
        std::make_pair(MDFlexConfig::FunctorOption::none, "");
    std::cerr << "No functor specified and no valid fall-back LJ functor available!" << std::endl;
#endif
    _configuration.addInteractionType(autopas::InteractionTypeOption::pairwise);
    std::cout << "WARNING: No functor was specified. Defaulting to " << functorName << std::endl;
  }

  // AutoPasDM owns the node-local AutoPas instance. md-flexible interacts
  // only with the distributed particle API and does not access the local container directly.
  _autoPasDMContainer = std::make_shared<DistributedContainerType>(
      runtime, _configuration.boxMin.value, _configuration.boxMax.value, _configuration.cutoff.value,
      _configuration.subdivideDimension.value, toDistributedBoundaryTypes(_configuration.boundaryOption.value),
      [&](auto &autoPas) {
        autoPas.setAllowedCellSizeFactors(*_configuration.cellSizeFactors.value);
        autoPas.setAllowedContainers(_configuration.containerOptions.value);
        autoPas.setAllowedInteractionTypeOptions(_configuration.getInteractionTypes());

        // Pairwise specific options
        autoPas.setAllowedDataLayouts(_configuration.dataLayoutOptions.value, autopas::InteractionTypeOption::pairwise);
        autoPas.setAllowedNewton3Options(_configuration.newton3Options.value, autopas::InteractionTypeOption::pairwise);
        autoPas.setAllowedTraversals(_configuration.traversalOptions.value, autopas::InteractionTypeOption::pairwise);
        autoPas.setAllowedVecPatterns(_configuration.vecPatternOptions.value, autopas::InteractionTypeOption::pairwise);
        autoPas.setAllowedLoadEstimators(_configuration.loadEstimatorOptions.value);

        // Triwise specific options
        autoPas.setAllowedDataLayouts(_configuration.dataLayoutOptions3B.value,
                                      autopas::InteractionTypeOption::triwise);
        autoPas.setAllowedNewton3Options(_configuration.newton3Options3B.value,
                                         autopas::InteractionTypeOption::triwise);
        autoPas.setAllowedTraversals(_configuration.traversalOptions3B.value, autopas::InteractionTypeOption::triwise);

        // General AutoPas options. Box and cutoff are owned by AutoPasDM and are
        // therefore intentionally not configured here.
        autoPas.setRelativeOptimumRange(_configuration.relativeOptimumRange.value);
        autoPas.setMaxTuningPhasesWithoutTest(_configuration.maxTuningPhasesWithoutTest.value);
        autoPas.setRelativeBlacklistRange(_configuration.relativeBlacklistRange.value);
        autoPas.setEvidenceFirstPrediction(_configuration.evidenceFirstPrediction.value);
        autoPas.setExtrapolationMethodOption(_configuration.extrapolationMethodOption.value);
        autoPas.setNumSamples(_configuration.tuningSamples.value);
        autoPas.setEarlyStoppingFactor(_configuration.earlyStoppingFactor.value);
        autoPas.setMaxEvidence(_configuration.tuningMaxEvidence.value);
        autoPas.setRuleFileName(_configuration.ruleFilename.value);
        autoPas.setFuzzyRuleFileName(_configuration.fuzzyRuleFilename.value);
        autoPas.setSelectorStrategy(_configuration.selectorStrategy.value);
        autoPas.setTuningInterval(_configuration.tuningInterval.value);
        autoPas.setTuningStrategyOption(_configuration.tuningStrategyOptions.value);
        autoPas.setTuningMetricOption(_configuration.tuningMetricOption.value);
        autoPas.setUseLOESSSmoothening(_configuration.useLOESSSmoothening.value);
        autoPas.setEnergySensorOption(_configuration.energySensorOption.value);
        autoPas.setMPITuningMaxDifferenceForBucket(_configuration.MPITuningMaxDifferenceForBucket.value);
        autoPas.setMPITuningWeightForMaxDensity(_configuration.MPITuningWeightForMaxDensity.value);
        autoPas.setVerletClusterSize(_configuration.verletClusterSize.value);
        autoPas.setVerletRebuildFrequency(_configuration.verletRebuildFrequency.value);
        autoPas.setVerletSkin(_configuration.verletSkinRadius.value);
        autoPas.setDeltaT(_configuration.deltaT.value);
        autoPas.setUseTuningLogger(_configuration.useTuningLogger.value);
        autoPas.setAoSSortingThreshold(_configuration.aosSortingThreshold.value);
        autoPas.setSoASortingThreshold(_configuration.soaSortingThreshold.value);
        autoPas.setOutputSuffix(outputSuffix);
      });

  autopas::Logger::get()->set_level(_configuration.logLevel.value);

  // Throw an error if there is not more than one configuration to test in the search space but more than one tuning
  // phase is requested
  if (_autoPasDMContainer->localSearchSpaceIsTrivial() and _configuration.tuningPhases.value > 0) {
    throw std::runtime_error(
        "Search space must not be trivial if the simulation time is limited by the number tuning phases");
  }

  // Load particles from the config file (object generators, checkpoint)
  loadParticles();

  if (_configuration.useThermostat.value and _configuration.deltaT.value != 0) {
    if (_configuration.addBrownianMotion.value) {
      Thermostat::addBrownianMotion(*_autoPasDMContainer, *(_configuration.getParticlePropertiesLibrary()),
                                    _configuration.initTemperature.value);
    }

    // Set the simulation directly to the desired initial temperature.
    Thermostat::apply(*_autoPasDMContainer, *(_configuration.getParticlePropertiesLibrary()),
                      _configuration.initTemperature.value, std::numeric_limits<double>::max());
  }

  _timers.initialization.stop();
}

void Simulation::finalize() {
  _timers.total.stop();
  _totalEnergySensor.endMeasurement();
  _autoPasDMContainer->barrier();

  logSimulationState();
  logMeasurements();
  _autoPasDMContainer->finalize();
}

void Simulation::run() {
  _timers.simulate.start();

  double maximumSigma = 0.;
  for (const auto &[siteType, sigma] : _configuration.sigmaMap.value) {
    (void)siteType;
    maximumSigma = std::max(maximumSigma, sigma);
  }

  while (needsMoreIterations()) {
    if (_vtkWriter.has_value() and _iteration % _configuration.vtkWriteFrequency.value == 0) {
      _timers.vtk.start();
      _vtkWriter->recordTimestep(_iteration, *_autoPasDMContainer);
      _timers.vtk.stop();
    }

    _timers.computationalLoad.start();
    if (_configuration.deltaT.value != 0 and not _simulationIsPaused) {
      updatePositionsAndResetForces();
#if MD_FLEXIBLE_MODE == MULTISITE
      updateQuaternions();
#endif
    }

    // Match md-flexible's adaptive-grid cadence. The application measures the local
    // computational work before distributed synchronization and only requests a balancing
    // step at the configured interval. AutoPasDM owns the actual geometry update,
    // resize, migration, and halo exchange.
    const bool shouldLoadBalance = _configuration.loadBalancer.value == LoadBalancerOption::invertedPressure and
                                   _configuration.deltaT.value != 0 and not _simulationIsPaused and
                                   _iteration % _configuration.loadBalancingInterval.value == 0;

    _timers.distributedSynchronization.start();
    if (shouldLoadBalance) {
      const auto computationalLoad = static_cast<double>(_timers.computationalLoad.stop());
      _autoPasDMContainer->prepareInteractions(computationalLoad);
    } else {
      _autoPasDMContainer->prepareInteractions();
    }
    _timers.distributedSynchronization.stop();

    _timers.reflectParticlesAtBoundaries.start();
    ReflectiveBoundary::apply(*_autoPasDMContainer, *_configuration.getParticlePropertiesLibrary(), maximumSigma);
    _timers.reflectParticlesAtBoundaries.stop();

    if (shouldLoadBalance) {
      // Continue measuring application-side computation after the communication phase.
      _timers.computationalLoad.start();
    }

    updateInteractionForces();

    if (_configuration.pauseSimulationDuringTuning.value) {
      // If PauseSimulationDuringTuning is enabled we need to update the _simulationIsPaused flag
      updateSimulationPauseState();
    }

    if (_configuration.deltaT.value != 0 and not _simulationIsPaused) {
      updateVelocities();
#if MD_FLEXIBLE_MODE == MULTISITE
      updateAngularVelocities();
#endif
      updateThermostat();
    }
    _timers.computationalLoad.stop();
#ifdef MD_FLEXIBLE_CALC_GLOBALS
    const auto potentialEnergyGlobal = _autoPasDMContainer->globalSum(_totalPotentialEnergy);
    const auto virialSumGlobal = _autoPasDMContainer->globalSum(_totalVirialSum);
    if (_autoPasDMContainer->isRoot()) {
      _globalLogger->logGlobals(_iteration, potentialEnergyGlobal, virialSumGlobal);
    }
    _totalPotentialEnergy = 0.;
    _totalVirialSum = 0.;
#endif
    if (not _simulationIsPaused) {
      ++_iteration;
    }

    if (autopas::Logger::get()->level() <= autopas::Logger::LogLevel::debug) {
      std::cout << "Current Memory usage on rank " << _autoPasDMContainer->rank() << ": "
                << autopas::memoryProfiler::currentMemoryUsage() << " kB\n";
    }

    if (_autoPasDMContainer->isRoot()) {
      auto [maxIterationsEstimate, maxIterationsIsPrecise] = estimateNumberOfIterations();
      if (not _configuration.dontShowProgressBar.value) {
        printProgress(_iteration, maxIterationsEstimate, maxIterationsIsPrecise);
      }
    }
  }
  _timers.simulate.stop();

  // Record last state of simulation.
  if (_vtkWriter.has_value()) {
    _vtkWriter->recordTimestep(_iteration, *_autoPasDMContainer);
  }
}

std::tuple<size_t, bool> Simulation::estimateNumberOfIterations() const {
  if (_configuration.tuningPhases.value > 0) {
    const size_t configsTestedPerTuningPhase = [&]() {
      if (std::any_of(_configuration.tuningStrategyOptions.value.begin(),
                      _configuration.tuningStrategyOptions.value.end(),
                      [](const auto &stratOpt) { return stratOpt == autopas::TuningStrategyOption::randomSearch; })) {
        return static_cast<size_t>(_configuration.tuningMaxEvidence.value);
      } else {
        // @TODO: this can be improved by considering the tuning strategy
        //      or averaging number of iterations per tuning phase and dynamically adapt prediction

        // This estimate is only valid for full search and no restrictions on the cartesian product.
        // add static to only evaluate this once
        const size_t searchSpaceSizePairwise =
            _configuration.getInteractionTypes().count(autopas::InteractionTypeOption::pairwise) == 0
                ? 0
                : autopas::SearchSpaceGenerators::cartesianProduct(
                      _configuration.containerOptions.value, _configuration.traversalOptions.value,
                      _configuration.loadEstimatorOptions.value, _configuration.dataLayoutOptions.value,
                      _configuration.newton3Options.value, _configuration.cellSizeFactors.value.get(),
                      _configuration.vecPatternOptions.value, autopas::InteractionTypeOption::pairwise)
                      .size();

        const size_t searchSpaceSizeTriwise =
            _configuration.getInteractionTypes().count(autopas::InteractionTypeOption::triwise) == 0
                ? 0
                : autopas::SearchSpaceGenerators::cartesianProduct(
                      _configuration.containerOptions.value, _configuration.traversalOptions3B.value,
                      _configuration.loadEstimatorOptions.value, _configuration.dataLayoutOptions3B.value,
                      _configuration.newton3Options3B.value, _configuration.cellSizeFactors.value.get(),
                      _configuration.vecPatternOptions.value, autopas::InteractionTypeOption::triwise)
                      .size();

        return std::max(searchSpaceSizePairwise, searchSpaceSizeTriwise);
      }
    }();
    // non-tuning iterations + tuning iterations + one iteration after last phase
    auto estimate =
        (_configuration.tuningPhases.value - 1) * _configuration.tuningInterval.value +
        (_configuration.tuningPhases.value * _configuration.tuningSamples.value * configsTestedPerTuningPhase) + 1;
    return {estimate, false};
  } else {
    return {_configuration.iterations.value, true};
  }
}

void Simulation::printProgress(size_t iterationProgress, size_t maxIterations, bool maxIsPrecise) {
  // percentage of iterations complete
  const double fractionDone = static_cast<double>(iterationProgress) / static_cast<double>(maxIterations);

  // length of the number of maxIterations
  const auto numCharsOfMaxIterations = static_cast<int>(std::to_string(maxIterations).size());

  // trailing information string
  std::stringstream info;
  info << std::setw(3) << std::round(fractionDone * 100) << "% " << std::setw(numCharsOfMaxIterations)
       << iterationProgress << "/";
  if (not maxIsPrecise) {
    info << "~";
  }
  info << maxIterations;

  // actual progress bar
  std::stringstream progressbar;
  progressbar << "[";
  // get current terminal width
  const auto terminalWidth = getTerminalWidth();

  // the bar should fill the terminal window so subtract everything else (-2 for "] ")
  const int maxBarWidth = static_cast<int>(terminalWidth - info.str().size() - progressbar.str().size() - 2ul);
  // sanity check for underflow
  if (maxBarWidth > terminalWidth) {
    std::cerr << "Warning! Terminal width appears to be too small or could not be read. Disabling progress bar."
              << std::endl;
    _configuration.dontShowProgressBar.value = true;
    return;
  }
  const auto barWidth =
      std::max(std::min(static_cast<decltype(maxBarWidth)>(maxBarWidth * (fractionDone)), maxBarWidth), 1);
  // don't print arrow tip if >= 100%
  if (iterationProgress >= maxIterations) {
    progressbar << std::string(barWidth, '=');
  } else {
    progressbar << std::string(barWidth - 1, '=') << '>' << std::string(maxBarWidth - barWidth, ' ');
  }
  progressbar << "] ";
  // clear current line (=delete previous progress bar)
  std::cout << std::string(terminalWidth, '\r');
  // print everything
  std::cout << progressbar.str() << info.str() << std::flush;
}

std::string Simulation::timerToString(const std::string &name, long timeNS, int numberWidth, long maxTime) {
  // only print timers that were actually used
  if (timeNS == 0) {
    return "";
  }

  std::ostringstream ss;
  ss << std::fixed << std::setprecision(_floatStringPrecision) << name << " : " << std::setw(numberWidth) << std::right
     << timeNS
     << " ns ("
     // min width of the representation of seconds is numberWidth - 9 (from conversion) + 4 (for dot and digits after)
     << std::setw(numberWidth - 5) << ((double)timeNS * 1e-9) << "s)";
  if (maxTime != 0) {
    ss << " =" << std::setw(7) << std::right << ((double)timeNS / (double)maxTime * 100) << "%";
  }
  ss << "\n";
  return ss.str();
}

void Simulation::updatePositionsAndResetForces() {
  _timers.positionUpdate.start();
  TimeDiscretization::calculatePositionsAndResetForces(
      *_autoPasDMContainer, *(_configuration.getParticlePropertiesLibrary()), _configuration.deltaT.value,
      _configuration.globalForce.value, _configuration.fastParticlesThrow.value);
  _timers.positionUpdate.stop();
}

void Simulation::updateQuaternions() {
  _timers.quaternionUpdate.start();
  TimeDiscretization::calculateQuaternionsAndResetTorques(
      *_autoPasDMContainer, *(_configuration.getParticlePropertiesLibrary()), _configuration.deltaT.value,
      _configuration.globalForce.value);
  _timers.quaternionUpdate.stop();
}

void Simulation::updateInteractionForces() {
  _timers.forceUpdateTotal.start();

  _previousIterationWasTuningIteration = _currentIterationIsTuningIteration;
  _currentIterationIsTuningIteration = false;
  long timeIteration = 0;

  // Calculate pairwise forces
  if (_configuration.getInteractionTypes().count(autopas::InteractionTypeOption::pairwise)) {
    _timers.forceUpdatePairwise.start();
    _currentIterationIsTuningIteration |= calculatePairwiseForces();
    timeIteration += _timers.forceUpdatePairwise.stop();
  }
  // Calculate triwise forces
  if (_configuration.getInteractionTypes().count(autopas::InteractionTypeOption::triwise)) {
    _timers.forceUpdateTriwise.start();
    _currentIterationIsTuningIteration |= calculateTriwiseForces();
    timeIteration += _timers.forceUpdateTriwise.stop();
  }

  // count time spent for tuning
  if (_currentIterationIsTuningIteration) {
    _timers.forceUpdateTuning.addTime(timeIteration);
    ++_numTuningIterations;
  } else {
    _timers.forceUpdateNonTuning.addTime(timeIteration);
    // if the previous iteration was a tuning iteration and the current one is not
    // we have reached the end of a tuning phase
    if (_previousIterationWasTuningIteration) {
      ++_numTuningPhasesCompleted;
    }
  }

  _timers.forceUpdateTotal.stop();
}

void Simulation::updateVelocities() {
  const double deltaT = _configuration.deltaT.value;

  if (deltaT != 0) {
    _timers.velocityUpdate.start();
    TimeDiscretization::calculateVelocities(*_autoPasDMContainer, *(_configuration.getParticlePropertiesLibrary()),
                                            deltaT);
    _timers.velocityUpdate.stop();
  }
}

void Simulation::updateAngularVelocities() {
  const double deltaT = _configuration.deltaT.value;

  _timers.angularVelocityUpdate.start();
  TimeDiscretization::calculateAngularVelocities(*_autoPasDMContainer, *(_configuration.getParticlePropertiesLibrary()),
                                                 deltaT);
  _timers.angularVelocityUpdate.stop();
}

void Simulation::updateThermostat() {
  if (_configuration.useThermostat.value and (_iteration % _configuration.thermostatInterval.value) == 0) {
    _timers.thermostat.start();
    Thermostat::apply(*_autoPasDMContainer, *(_configuration.getParticlePropertiesLibrary()),
                      _configuration.targetTemperature.value, _configuration.deltaTemp.value);
    _timers.thermostat.stop();
  }
}

long Simulation::accumulateTime(const long &time) { return _autoPasDMContainer->globalSum(time); }

bool Simulation::calculatePairwiseForces() {
  const auto wasTuningIteration = applyWithChosenFunctor<bool>([&](auto &&functor) {
    auto isTuningIteration = _autoPasDMContainer->computeInteractionsPrepared(&functor);
#ifdef MD_FLEXIBLE_CALC_GLOBALS
    _totalPotentialEnergy += functor.getPotentialEnergy();
    _totalVirialSum += functor.getVirial();
#endif
    return isTuningIteration;
  });
  return wasTuningIteration;
}

bool Simulation::calculateTriwiseForces() {
  const auto wasTuningIteration = applyWithChosenFunctor3B<bool>([&](auto &&functor) {
    auto isTuningIteration = _autoPasDMContainer->computeInteractionsPrepared(&functor);
#ifdef MD_FLEXIBLE_CALC_GLOBALS
    _totalPotentialEnergy += functor.getPotentialEnergy();
    _totalVirialSum += functor.getVirial();
#endif
    return isTuningIteration;
  });
  return wasTuningIteration;
}

void Simulation::calculateGlobalForces(const std::array<double, 3> &globalForce) {
  _autoPasDMContainer->applyToOwnedParticles([&](auto &particle) { particle.addF(globalForce); });
}

void Simulation::logSimulationState() {
  const auto ownedParticles = _autoPasDMContainer->getGlobalNumberOfOwnedParticles();
  const auto haloParticles = _autoPasDMContainer->globalSum(_autoPasDMContainer->getLocalNumberOfHaloParticles());
  const auto totalNumberOfParticles = ownedParticles + haloParticles;

  if (_autoPasDMContainer->isRoot()) {
    std::cout << "\n\n"
              << "Total number of particles at the end of Simulation: " << totalNumberOfParticles << "\n"
              << "Owned: " << ownedParticles << "\n"
              << "Halo : " << haloParticles << "\n";
  }
}

void Simulation::updateSimulationPauseState() {
  // If we are at the beginning of a tuning phase, we need to freeze the simulation
  if (_currentIterationIsTuningIteration and (not _previousIterationWasTuningIteration)) {
    std::cout << "Iteration " << _iteration << ": Freezing simulation for tuning phase. Starting tuning phase...\n";
    _simulationIsPaused = true;
  }

  // If we are at the end of a tuning phase, we need to resume the simulation
  if (_previousIterationWasTuningIteration and (not _currentIterationIsTuningIteration)) {
    std::cout << "Iteration " << _iteration << ": Resuming simulation after tuning phase.\n";

    // reset the forces which accumulated during the tuning phase
    _autoPasDMContainer->applyToOwnedParticles(
        [&](auto &particle) { particle.setF(_configuration.globalForce.value); });

    // calculate the forces of the latest iteration again
    updateInteractionForces();

    _simulationIsPaused = false;
  }
}

void Simulation::logMeasurements() {
  const long positionUpdate = accumulateTime(_timers.positionUpdate.getTotalTime());
  const long quaternionUpdate = accumulateTime(_timers.quaternionUpdate.getTotalTime());
  const long updateContainer = accumulateTime(_timers.updateContainer.getTotalTime());
  const long forceUpdateTotal = accumulateTime(_timers.forceUpdateTotal.getTotalTime());
  const long forceUpdatePairwise = accumulateTime(_timers.forceUpdatePairwise.getTotalTime());
  const long forceUpdateTriwise = accumulateTime(_timers.forceUpdateTriwise.getTotalTime());
  const long forceUpdateTuning = accumulateTime(_timers.forceUpdateTuning.getTotalTime());
  const long forceUpdateNonTuning = accumulateTime(_timers.forceUpdateNonTuning.getTotalTime());
  const long velocityUpdate = accumulateTime(_timers.velocityUpdate.getTotalTime());
  const long angularVelocityUpdate = accumulateTime(_timers.angularVelocityUpdate.getTotalTime());
  const long simulate = accumulateTime(_timers.simulate.getTotalTime());
  const long vtk = accumulateTime(_timers.vtk.getTotalTime());
  const long initialization = accumulateTime(_timers.initialization.getTotalTime());
  const long total = accumulateTime(_timers.total.getTotalTime());
  const long thermostat = accumulateTime(_timers.thermostat.getTotalTime());
  const long haloParticleExchange = accumulateTime(_timers.haloParticleExchange.getTotalTime());
  const long reflectParticlesAtBoundaries = accumulateTime(_timers.reflectParticlesAtBoundaries.getTotalTime());
  const long migratingParticleExchange = accumulateTime(_timers.migratingParticleExchange.getTotalTime());
  const long loadBalancing = accumulateTime(_timers.loadBalancing.getTotalTime());
  const long distributedSynchronization = accumulateTime(_timers.distributedSynchronization.getTotalTime());
#ifdef AUTOPASDM_ENABLE_TIMERS
  const auto autoPasDMPerformanceCounters = _autoPasDMContainer->getPerformanceCountersSnapshot();
  const auto accumulateAutoPasDMTime = [&](std::uint64_t localTime) {
    return static_cast<long>(_autoPasDMContainer->globalSum(localTime));
  };
  const long autoPasDMSynchronization = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.synchronizationNs);
  const long autoPasDMUpdateContainer = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.updateContainerNs);
  const long autoPasDMLoadBalancing = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.loadBalancingNs);
  const long autoPasDMBoundarySynchronization =
      accumulateAutoPasDMTime(autoPasDMPerformanceCounters.boundarySynchronizationNs);
  const long autoPasDMResizeBox = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.resizeBoxNs);
  const long autoPasDMMigration = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.migrationNs);
  const long autoPasDMAddMigratedParticles =
      accumulateAutoPasDMTime(autoPasDMPerformanceCounters.addMigratedParticlesNs);
  const long autoPasDMOwnedParticleCollection =
      accumulateAutoPasDMTime(autoPasDMPerformanceCounters.ownedParticleCollectionNs);
  const long autoPasDMHaloExchange = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.haloExchangeNs);
  const long autoPasDMAddHaloParticles = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.addHaloParticlesNs);
#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
  const long autoPasDMMigrationClassification =
      accumulateAutoPasDMTime(autoPasDMPerformanceCounters.migrationClassificationNs);
  const long autoPasDMHaloInitialCopy = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.haloInitialCopyNs);
  const long autoPasDMHaloSelection = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.haloSelectionNs);
  const long autoPasDMHaloPeriodicShift = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.haloPeriodicShiftNs);
  const long autoPasDMHaloBufferInsertion = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.haloBufferInsertionNs);
  const long autoPasDMCommunicationPack = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.communicationPackNs);
  const long autoPasDMCommunicationCountExchange =
      accumulateAutoPasDMTime(autoPasDMPerformanceCounters.communicationCountExchangeNs);
  const long autoPasDMCommunicationPayloadWait =
      accumulateAutoPasDMTime(autoPasDMPerformanceCounters.communicationPayloadWaitNs);
  const long autoPasDMCommunicationUnpack = accumulateAutoPasDMTime(autoPasDMPerformanceCounters.communicationUnpackNs);

  const auto migrationParticlesSent =
      _autoPasDMContainer->globalSum(autoPasDMPerformanceCounters.migrationParticlesSent);
  const auto migrationParticlesReceived =
      _autoPasDMContainer->globalSum(autoPasDMPerformanceCounters.migrationParticlesReceived);
  const auto haloParticlesSent = _autoPasDMContainer->globalSum(autoPasDMPerformanceCounters.haloParticlesSent);
  const auto haloParticlesReceived = _autoPasDMContainer->globalSum(autoPasDMPerformanceCounters.haloParticlesReceived);
  const auto communicationBytesSent =
      _autoPasDMContainer->globalSum(autoPasDMPerformanceCounters.communicationBytesSent);
  const auto communicationBytesReceived =
      _autoPasDMContainer->globalSum(autoPasDMPerformanceCounters.communicationBytesReceived);
#endif
#endif
#ifdef AUTOPAS_ENABLE_ENERGY_MEASUREMENTS
  const double totalEnergy = _autoPasDMContainer->globalSum(_totalEnergySensor.getJoules());
#endif

  if (_autoPasDMContainer->isRoot()) {
    const long wallClockTime = _timers.total.getTotalTime();
    // use the two potentially largest timers to determine the number of chars needed
    const auto maximumNumberOfDigits =
        static_cast<int>(std::max(std::to_string(total).length(), std::to_string(wallClockTime).length()));
    std::cout << "Measurements:\n";
    std::cout << timerToString("Total accumulated                 ", total, maximumNumberOfDigits);
    std::cout << timerToString("  Initialization                  ", initialization, maximumNumberOfDigits, total);
    std::cout << timerToString("  Simulate                        ", simulate, maximumNumberOfDigits, total);
    std::cout << timerToString("    PositionUpdate                ", positionUpdate, maximumNumberOfDigits, simulate);
#if MD_FLEXIBLE_MODE == MULTISITE
    std::cout << timerToString("    QuaternionUpdate              ", quaternionUpdate, maximumNumberOfDigits, simulate);
#endif
    std::cout << timerToString("    UpdateContainer               ", updateContainer, maximumNumberOfDigits, simulate);
    std::cout << timerToString("    Boundaries                    ",
                               haloParticleExchange + reflectParticlesAtBoundaries + migratingParticleExchange,
                               maximumNumberOfDigits, simulate);
    std::cout << timerToString("      HaloParticleExchange        ", haloParticleExchange, maximumNumberOfDigits,
                               haloParticleExchange + reflectParticlesAtBoundaries + migratingParticleExchange);
    std::cout << timerToString("      ReflectParticlesAtBoundaries", reflectParticlesAtBoundaries,
                               maximumNumberOfDigits,
                               haloParticleExchange + reflectParticlesAtBoundaries + migratingParticleExchange);
    std::cout << timerToString("      MigratingParticleExchange   ", migratingParticleExchange, maximumNumberOfDigits,
                               haloParticleExchange + reflectParticlesAtBoundaries + migratingParticleExchange);
    std::cout << timerToString("    ForceUpdateTotal              ", forceUpdateTotal, maximumNumberOfDigits, simulate);
    std::cout << timerToString("      Tuning                      ", forceUpdateTuning, maximumNumberOfDigits,
                               forceUpdateTotal);
    std::cout << timerToString("      ForceUpdateTuning           ", forceUpdateTuning, maximumNumberOfDigits,
                               forceUpdateTotal);
    std::cout << timerToString("      ForceUpdateNonTuning        ", forceUpdateNonTuning, maximumNumberOfDigits,
                               forceUpdateTotal);
    std::cout << timerToString("    VelocityUpdate                ", velocityUpdate, maximumNumberOfDigits, simulate);
#if MD_FLEXIBLE_MODE == MULTISITE
    std::cout << timerToString("    AngularVelocityUpdate         ", angularVelocityUpdate, maximumNumberOfDigits,
                               simulate);
#endif
    std::cout << timerToString("    Thermostat                    ", thermostat, maximumNumberOfDigits, simulate);
    std::cout << timerToString("    Vtk                           ", vtk, maximumNumberOfDigits, simulate);
    std::cout << timerToString("    LoadBalancing                 ", loadBalancing, maximumNumberOfDigits, simulate);
    std::cout << timerToString("    DistributedSynchronization    ", distributedSynchronization, maximumNumberOfDigits,
                               simulate);
#ifdef AUTOPASDM_ENABLE_TIMERS
    std::cout << timerToString("      AutoPasDM CoreSynchronization     ", autoPasDMSynchronization,
                               maximumNumberOfDigits, distributedSynchronization);
    std::cout << timerToString("        AutoPasDM UpdateContainer       ", autoPasDMUpdateContainer,
                               maximumNumberOfDigits, autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM LoadBalancing         ", autoPasDMLoadBalancing,
                               maximumNumberOfDigits, autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM BoundarySync          ", autoPasDMBoundarySynchronization,
                               maximumNumberOfDigits, autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM ResizeBox             ", autoPasDMResizeBox, maximumNumberOfDigits,
                               autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM Migration             ", autoPasDMMigration, maximumNumberOfDigits,
                               autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM AddMigratedParticles  ", autoPasDMAddMigratedParticles,
                               maximumNumberOfDigits, autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM CollectOwnedParticles ", autoPasDMOwnedParticleCollection,
                               maximumNumberOfDigits, autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM HaloExchange          ", autoPasDMHaloExchange, maximumNumberOfDigits,
                               autoPasDMSynchronization);
    std::cout << timerToString("        AutoPasDM AddHaloParticles      ", autoPasDMAddHaloParticles,
                               maximumNumberOfDigits, autoPasDMSynchronization);
#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
    std::cout << timerToString("          AutoPasDM MigrationClassify   ", autoPasDMMigrationClassification,
                               maximumNumberOfDigits, autoPasDMMigration);
    std::cout << timerToString("          AutoPasDM HaloInitialCopy     ", autoPasDMHaloInitialCopy,
                               maximumNumberOfDigits, autoPasDMHaloExchange);
    std::cout << timerToString("          AutoPasDM HaloSelection       ", autoPasDMHaloSelection,
                               maximumNumberOfDigits, autoPasDMHaloExchange);
    std::cout << timerToString("          AutoPasDM HaloPeriodicShift   ", autoPasDMHaloPeriodicShift,
                               maximumNumberOfDigits, autoPasDMHaloExchange);
    std::cout << timerToString("          AutoPasDM HaloBufferInsertion ", autoPasDMHaloBufferInsertion,
                               maximumNumberOfDigits, autoPasDMHaloExchange);
    const long autoPasDMCommunicationTotal = autoPasDMMigration + autoPasDMHaloExchange;
    std::cout << timerToString("          AutoPasDM CommunicationPack   ", autoPasDMCommunicationPack,
                               maximumNumberOfDigits, autoPasDMCommunicationTotal);
    std::cout << timerToString("          AutoPasDM CountExchange       ", autoPasDMCommunicationCountExchange,
                               maximumNumberOfDigits, autoPasDMCommunicationTotal);
    std::cout << timerToString("          AutoPasDM PayloadWait         ", autoPasDMCommunicationPayloadWait,
                               maximumNumberOfDigits, autoPasDMCommunicationTotal);
    std::cout << timerToString("          AutoPasDM CommunicationUnpack ", autoPasDMCommunicationUnpack,
                               maximumNumberOfDigits, autoPasDMCommunicationTotal);
#endif
#endif
    std::cout << timerToString("  One iteration                 ", simulate / static_cast<long>(_iteration),
                               maximumNumberOfDigits, total);

    std::cout << timerToString("Total wall-clock time             ", wallClockTime, maximumNumberOfDigits, total);
    std::cout << "\n";

    std::cout << "Tuning iterations                  : " << _numTuningIterations << " / " << _iteration << " = "
              << (static_cast<double>(_numTuningIterations) / static_cast<double>(_iteration) * 100.) << "%"
              << "\n";

    auto mfups = static_cast<double>(_autoPasDMContainer->getLocalNumberOfOwnedParticles() * _iteration) * 1e-6 /
                 (static_cast<double>(forceUpdateTotal) * 1e-9);  // 1e-9 for ns to s, 1e-6 for M in MFUPs
    std::cout << "MFUPs/sec                          : " << mfups << "\n";
#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
    std::cout << "AutoPasDM communication counters (sum over ranks):\n";
    std::cout << "  Migration particles sent        : " << migrationParticlesSent << "\n";
    std::cout << "  Migration particles received    : " << migrationParticlesReceived << "\n";
    std::cout << "  Halo particles sent             : " << haloParticlesSent << "\n";
    std::cout << "  Halo particles received         : " << haloParticlesReceived << "\n";
    std::cout << "  Payload bytes sent              : " << communicationBytesSent << "\n";
    std::cout << "  Payload bytes received          : " << communicationBytesReceived << "\n";
#endif
#ifdef AUTOPAS_ENABLE_DYNAMIC_CONTAINERS
    std::cout << "Mean Rebuild Frequency               : " << _autoPasDMContainer->getMeanLocalRebuildFrequency()
              << "\n";
#endif
#ifdef AUTOPAS_ENABLE_ENERGY_MEASUREMENTS
    std::cout << "Total Energy Consumed (in Joules)    : " << totalEnergy << "\n";
#endif
  }
}

bool Simulation::needsMoreIterations() const {
  return _iteration < _configuration.iterations.value or _numTuningPhasesCompleted < _configuration.tuningPhases.value;
}

void Simulation::checkNumParticles(size_t expectedNumParticlesGlobal, size_t numParticlesCurrentlyMigratingLocal,
                                   int lineNumber) {
  if (std::all_of(_configuration.boundaryOption.value.begin(), _configuration.boundaryOption.value.end(),
                  [](const auto &boundary) {
                    return boundary == options::BoundaryTypeOption::periodic or
                           boundary == options::BoundaryTypeOption::reflective;
                  })) {
    const auto numParticlesNowTotal =
        _autoPasDMContainer->globalSum(_autoPasDMContainer->getLocalNumberOfOwnedParticles());
    const auto numParticlesMigratingTotal = _autoPasDMContainer->globalSum(numParticlesCurrentlyMigratingLocal);
    if (expectedNumParticlesGlobal != numParticlesNowTotal + numParticlesMigratingTotal) {
      const auto myRank = _autoPasDMContainer->rank();
      std::stringstream ss;
      // clang-format off
      ss << "Rank " << myRank << " Line " << lineNumber
         << ": Particles Lost! All Boundaries are periodic but the number of particles changed:"
         << "Expected        : " << expectedNumParticlesGlobal
         << "Actual          : " << (numParticlesNowTotal + numParticlesMigratingTotal)
         << "  in containers : " << numParticlesNowTotal
         << "  migrating     : " << numParticlesMigratingTotal
         << std::endl;
      // clang-format on
      throw std::runtime_error(ss.str());
    }
  }
}

void Simulation::loadParticles() {
  const auto numParticlesInConfigLocally = _configuration.particles.size();

  // Object generators describe replicated configuration input. Only rank 0 is a logical
  // source for these particles. Checkpoint loading is different: when checkpoint pieces
  // match the current rank count, every rank contributes its own piece.
  size_t numParticlesInConfigGlobal{};
  if (_configuration.checkpointfile.value.empty()) {
    _autoPasDMContainer->addParticlesFromRoot(_configuration.particles);
    numParticlesInConfigGlobal =
        _autoPasDMContainer->globalSum(_autoPasDMContainer->isRoot() ? numParticlesInConfigLocally : size_t{0});
  } else {
    _autoPasDMContainer->addDistributedParticles(_configuration.particles);
    numParticlesInConfigGlobal = _autoPasDMContainer->globalSum(numParticlesInConfigLocally);
  }
  _configuration.flushParticles();

  // Output and sanity checks
  const auto rank = _autoPasDMContainer->rank();
  const auto numberOfRanks = _autoPasDMContainer->numberOfRanks();
  const size_t numParticlesLocally = _autoPasDMContainer->getLocalNumberOfOwnedParticles();
  std::cout << "Number of particles at initialization "
            // align outputs based on the max number of ranks
            << "on rank " << std::setw(std::to_string(numberOfRanks).length()) << std::right << rank << ": "
            << numParticlesLocally << "\n";

  const auto numParticlesAddedGlobal = _autoPasDMContainer->globalSum(numParticlesLocally);

  if (_autoPasDMContainer->isRoot()) {
    std::cout << "Number of particles at initialization globally" << std::setw(std::to_string(numberOfRanks).length())
              << ""
              << ": " << numParticlesAddedGlobal << "\n";

    if (numParticlesAddedGlobal != numParticlesInConfigGlobal) {
      throw std::runtime_error(
          "Simulation::loadParticles(): "
          "Not all particles from the configuration file could be added to AutoPasDM!\n"
          "Configuration : " +
          std::to_string(numParticlesInConfigGlobal) +
          "\n"
          "Added globally: " +
          std::to_string(numParticlesAddedGlobal));
    }
  }
}

template <class ReturnType, class FunctionType>
ReturnType Simulation::applyWithChosenFunctor(FunctionType f) {
  const double cutoff = _configuration.cutoff.value;
  auto &particlePropertiesLibrary = *_configuration.getParticlePropertiesLibrary();
  switch (_configuration.functorOption.value) {
    case MDFlexConfig::FunctorOption::lj12_6: {
#if defined(MD_FLEXIBLE_FUNCTOR_AUTOVEC)
      return f(LJFunctorTypeAutovec{cutoff, particlePropertiesLibrary});
#else
      throw std::runtime_error(
          "MD-Flexible was not compiled with support for LJFunctor AutoVec. Activate it via `cmake "
          "-DMD_FLEXIBLE_FUNCTOR_AUTOVEC=ON`.");
#endif
    }
    case MDFlexConfig::FunctorOption::lj12_6_AVX: {
#if defined(MD_FLEXIBLE_FUNCTOR_AVX) && defined(__AVX__)
      return f(LJFunctorTypeAVX{cutoff, particlePropertiesLibrary});
#else
      throw std::runtime_error(
          "MD-Flexible was not compiled with support for LJFunctor AVX. Activate it via `cmake "
          "-DMD_FLEXIBLE_FUNCTOR_AVX=ON`.");
#endif
    }
    case MDFlexConfig::FunctorOption::lj12_6_SVE: {
#if defined(MD_FLEXIBLE_FUNCTOR_SVE) && defined(__ARM_FEATURE_SVE)
      return f(LJFunctorTypeSVE{cutoff, particlePropertiesLibrary});
#else
      throw std::runtime_error(
          "MD-Flexible was not compiled with support for LJFunctor SVE. Activate it via `cmake "
          "-DMD_FLEXIBLE_FUNCTOR_SVE=ON`.");
#endif
    }
    case MDFlexConfig::FunctorOption::lj12_6_HWY: {
#if defined(MD_FLEXIBLE_FUNCTOR_HWY)
      return f(LJFunctorTypeHWY{cutoff, std::ref(particlePropertiesLibrary)});
#else
      throw std::runtime_error(
          "MD-Flexible was not compiled with support for LJFunctor HWY. Activate it via `cmake "
          "-DMD_FLEXIBLE_FUNCTOR_HWY=ON`.");
#endif
    }
    default: {
      throw std::runtime_error("Unknown pairwise functor choice" +
                               std::to_string(static_cast<int>(_configuration.functorOption.value)));
    }
  }
}

template <class ReturnType, class FunctionType>
ReturnType Simulation::applyWithChosenFunctor3B(FunctionType f) {
  const double cutoff = _configuration.cutoff.value;
  auto &particlePropertiesLibrary = *_configuration.getParticlePropertiesLibrary();
  switch (_configuration.functorOption3B.value) {
    case MDFlexConfig::FunctorOption3B::at: {
#if defined(MD_FLEXIBLE_FUNCTOR_ATM_AUTOVEC)
      return f(ATMFunctor{cutoff, particlePropertiesLibrary});
#else
      throw std::runtime_error(
          "MD-Flexible was not compiled with support for AxilrodTellerMuto Functor. Activate it via `cmake "
          "-DMD_FLEXIBLE_FUNCTOR_ATM_AUTOVEC=ON`.");
#endif
    }
    default: {
      throw std::runtime_error("Unknown triwise functor choice" +
                               std::to_string(static_cast<int>(_configuration.functorOption3B.value)));
    }
  }
}
