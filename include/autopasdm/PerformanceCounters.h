#pragma once

#include <cstdint>

#if defined(AUTOPASDM_ENABLE_DETAILED_TIMERS) && !defined(AUTOPASDM_ENABLE_TIMERS)
#define AUTOPASDM_ENABLE_TIMERS
#endif

#ifdef AUTOPASDM_ENABLE_TIMERS
#include <autopas/utils/Timer.h>
#endif

namespace autopasdm::performance {

/**
 * Snapshot of the local AutoPasDM performance counters.
 *
 * Timer values are accumulated nanoseconds on the calling MPI rank. Counter
 * values are local counts and can be reduced by the application after the
 * measurement phase. Fields that are not enabled at compile time remain zero.
 */
struct PerformanceCountersSnapshot {
  std::uint64_t synchronizationNs{};
  std::uint64_t updateContainerNs{};
  std::uint64_t loadBalancingNs{};
  std::uint64_t boundarySynchronizationNs{};
  std::uint64_t resizeBoxNs{};
  std::uint64_t migrationNs{};
  std::uint64_t addMigratedParticlesNs{};
  std::uint64_t ownedParticleCollectionNs{};
  std::uint64_t haloExchangeNs{};
  std::uint64_t addHaloParticlesNs{};

  std::uint64_t migrationClassificationNs{};
  std::uint64_t haloInitialCopyNs{};
  std::uint64_t haloSelectionNs{};
  std::uint64_t haloPeriodicShiftNs{};
  std::uint64_t haloBufferInsertionNs{};
  std::uint64_t communicationPackNs{};
  std::uint64_t communicationCountExchangeNs{};
  std::uint64_t communicationPayloadWaitNs{};
  std::uint64_t communicationUnpackNs{};

  std::uint64_t migrationParticlesSent{};
  std::uint64_t migrationParticlesReceived{};
  std::uint64_t haloParticlesSent{};
  std::uint64_t haloParticlesReceived{};
  std::uint64_t communicationBytesSent{};
  std::uint64_t communicationBytesReceived{};
};

#ifdef AUTOPASDM_ENABLE_TIMERS

/**
 * RAII helper for exception-safe scoped timing with autopas::utils::Timer.
 */
class ScopedTimer {
 public:
  /**
   * Start the supplied timer when it is non-null.
   *
   * @param timer Timer to measure with, or nullptr to disable this scope.
   */
  explicit ScopedTimer(autopas::utils::Timer *timer) : _timer(timer) {
    if (_timer != nullptr) {
      _timer->start();
    }
  }

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;
  ScopedTimer(ScopedTimer &&) = delete;
  ScopedTimer &operator=(ScopedTimer &&) = delete;

  /** Stop the timer when leaving the measured scope. */
  ~ScopedTimer() {
    if (_timer != nullptr) {
      _timer->stop();
    }
  }

 private:
  autopas::utils::Timer *_timer;
};

/**
 * Optional timer and counter storage shared by one AutoPasDM instance.
 */
struct PerformanceCounters {
  autopas::utils::Timer synchronization;
  autopas::utils::Timer updateContainer;
  autopas::utils::Timer loadBalancing;
  autopas::utils::Timer boundarySynchronization;
  autopas::utils::Timer resizeBox;
  autopas::utils::Timer migration;
  autopas::utils::Timer addMigratedParticles;
  autopas::utils::Timer ownedParticleCollection;
  autopas::utils::Timer haloExchange;
  autopas::utils::Timer addHaloParticles;

#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
  autopas::utils::Timer migrationClassification;
  autopas::utils::Timer haloInitialCopy;
  autopas::utils::Timer haloSelection;
  autopas::utils::Timer haloPeriodicShift;
  autopas::utils::Timer haloBufferInsertion;
  autopas::utils::Timer communicationPack;
  autopas::utils::Timer communicationCountExchange;
  autopas::utils::Timer communicationPayloadWait;
  autopas::utils::Timer communicationUnpack;

  std::uint64_t migrationParticlesSent{};
  std::uint64_t migrationParticlesReceived{};
  std::uint64_t haloParticlesSent{};
  std::uint64_t haloParticlesReceived{};
  std::uint64_t communicationBytesSent{};
  std::uint64_t communicationBytesReceived{};
#endif

  /** Get the currently accumulated performance counter values. */
  [[nodiscard]] PerformanceCountersSnapshot snapshot() const {
    PerformanceCountersSnapshot result;
    result.synchronizationNs = static_cast<std::uint64_t>(synchronization.getTotalTime());
    result.updateContainerNs = static_cast<std::uint64_t>(updateContainer.getTotalTime());
    result.loadBalancingNs = static_cast<std::uint64_t>(loadBalancing.getTotalTime());
    result.boundarySynchronizationNs = static_cast<std::uint64_t>(boundarySynchronization.getTotalTime());
    result.resizeBoxNs = static_cast<std::uint64_t>(resizeBox.getTotalTime());
    result.migrationNs = static_cast<std::uint64_t>(migration.getTotalTime());
    result.addMigratedParticlesNs = static_cast<std::uint64_t>(addMigratedParticles.getTotalTime());
    result.ownedParticleCollectionNs = static_cast<std::uint64_t>(ownedParticleCollection.getTotalTime());
    result.haloExchangeNs = static_cast<std::uint64_t>(haloExchange.getTotalTime());
    result.addHaloParticlesNs = static_cast<std::uint64_t>(addHaloParticles.getTotalTime());

#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
    result.migrationClassificationNs = static_cast<std::uint64_t>(migrationClassification.getTotalTime());
    result.haloInitialCopyNs = static_cast<std::uint64_t>(haloInitialCopy.getTotalTime());
    result.haloSelectionNs = static_cast<std::uint64_t>(haloSelection.getTotalTime());
    result.haloPeriodicShiftNs = static_cast<std::uint64_t>(haloPeriodicShift.getTotalTime());
    result.haloBufferInsertionNs = static_cast<std::uint64_t>(haloBufferInsertion.getTotalTime());
    result.communicationPackNs = static_cast<std::uint64_t>(communicationPack.getTotalTime());
    result.communicationCountExchangeNs = static_cast<std::uint64_t>(communicationCountExchange.getTotalTime());
    result.communicationPayloadWaitNs = static_cast<std::uint64_t>(communicationPayloadWait.getTotalTime());
    result.communicationUnpackNs = static_cast<std::uint64_t>(communicationUnpack.getTotalTime());

    result.migrationParticlesSent = migrationParticlesSent;
    result.migrationParticlesReceived = migrationParticlesReceived;
    result.haloParticlesSent = haloParticlesSent;
    result.haloParticlesReceived = haloParticlesReceived;
    result.communicationBytesSent = communicationBytesSent;
    result.communicationBytesReceived = communicationBytesReceived;
#endif

    return result;
  }

  /** Reset all accumulated timer values and counters. */
  void reset() {
    synchronization.reset();
    updateContainer.reset();
    loadBalancing.reset();
    boundarySynchronization.reset();
    resizeBox.reset();
    migration.reset();
    addMigratedParticles.reset();
    ownedParticleCollection.reset();
    haloExchange.reset();
    addHaloParticles.reset();

#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
    migrationClassification.reset();
    haloInitialCopy.reset();
    haloSelection.reset();
    haloPeriodicShift.reset();
    haloBufferInsertion.reset();
    communicationPack.reset();
    communicationCountExchange.reset();
    communicationPayloadWait.reset();
    communicationUnpack.reset();

    migrationParticlesSent = 0;
    migrationParticlesReceived = 0;
    haloParticlesSent = 0;
    haloParticlesReceived = 0;
    communicationBytesSent = 0;
    communicationBytesReceived = 0;
#endif
  }
};

#else

/**
 * Empty performance-counter storage used by production builds.
 *
 * All timing and counting expressions are removed by the preprocessor. The
 * object is marked [[no_unique_address]] by AutoPasDM, so it does not
 * add storage to the container either.
 */
struct PerformanceCounters {
  /** Get an all-zero performance counter snapshot. */
  [[nodiscard]] constexpr PerformanceCountersSnapshot snapshot() const { return {}; }

  /** No-op reset for builds without performance counters. */
  constexpr void reset() {}
};

#endif

}  // namespace autopasdm::performance

#define AUTOPASDM_DETAIL_CONCAT_IMPL(left, right) left##right
#define AUTOPASDM_DETAIL_CONCAT(left, right) AUTOPASDM_DETAIL_CONCAT_IMPL(left, right)

#ifdef AUTOPASDM_ENABLE_TIMERS
#define AUTOPASDM_TIMER_START(timer) (timer).start()
#define AUTOPASDM_TIMER_STOP(timer) (timer).stop()
#define AUTOPASDM_SCOPED_TIMER(timer) \
  ::autopasdm::performance::ScopedTimer AUTOPASDM_DETAIL_CONCAT(_autopasdmScopedTimer_, __LINE__) { &(timer) }
#else
#define AUTOPASDM_TIMER_START(timer)
#define AUTOPASDM_TIMER_STOP(timer)
#define AUTOPASDM_SCOPED_TIMER(timer)
#endif

#ifdef AUTOPASDM_ENABLE_DETAILED_TIMERS
#define AUTOPASDM_DETAILED_TIMER_START(performanceCountersPointer, timerMember) \
  do {                                                                          \
    if ((performanceCountersPointer) != nullptr) {                              \
      (performanceCountersPointer)->timerMember.start();                        \
    }                                                                           \
  } while (false)
#define AUTOPASDM_DETAILED_TIMER_STOP(performanceCountersPointer, timerMember) \
  do {                                                                         \
    if ((performanceCountersPointer) != nullptr) {                             \
      (performanceCountersPointer)->timerMember.stop();                        \
    }                                                                          \
  } while (false)
#define AUTOPASDM_DETAILED_SCOPED_TIMER(performanceCountersPointer, timerMember)                            \
  ::autopasdm::performance::ScopedTimer AUTOPASDM_DETAIL_CONCAT(_autopasdmDetailedScopedTimer_, __LINE__) { \
    (performanceCountersPointer) != nullptr ? &(performanceCountersPointer)->timerMember : nullptr          \
  }
#define AUTOPASDM_DETAILED_COUNTERS_ARGUMENT(value) , value
#define AUTOPASDM_DETAILED_ADD(performanceCountersPointer, counterMember, value)        \
  do {                                                                                  \
    if ((performanceCountersPointer) != nullptr) {                                      \
      (performanceCountersPointer)->counterMember += static_cast<std::uint64_t>(value); \
    }                                                                                   \
  } while (false)
#else
#define AUTOPASDM_DETAILED_COUNTERS_ARGUMENT(value)
#define AUTOPASDM_DETAILED_TIMER_START(performanceCountersPointer, timerMember)
#define AUTOPASDM_DETAILED_TIMER_STOP(performanceCountersPointer, timerMember)
#define AUTOPASDM_DETAILED_SCOPED_TIMER(performanceCountersPointer, timerMember)
#define AUTOPASDM_DETAILED_ADD(performanceCountersPointer, counterMember, value)
#endif
