#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <type_traits>

#include "distributed_autopas/PerformanceCounters.h"

TEST(PerformanceCountersTest, SnapshotStartsAtZeroAndCanBeReset) {
  dap::performance::PerformanceCounters performanceCounters;

  const auto initialSnapshot = performanceCounters.snapshot();
  EXPECT_EQ(initialSnapshot.synchronizationNs, 0);
  EXPECT_EQ(initialSnapshot.updateContainerNs, 0);
  EXPECT_EQ(initialSnapshot.communicationBytesSent, 0);

#ifdef DAP_ENABLE_TIMERS
  performanceCounters.updateContainer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  performanceCounters.updateContainer.stop();
  EXPECT_GT(performanceCounters.snapshot().updateContainerNs, 0);
#endif

#ifdef DAP_ENABLE_DETAILED_TIMERS
  performanceCounters.haloInitialCopy.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  performanceCounters.haloInitialCopy.stop();
  performanceCounters.haloBufferInsertion.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  performanceCounters.haloBufferInsertion.stop();
  EXPECT_GT(performanceCounters.snapshot().haloInitialCopyNs, 0);
  EXPECT_GT(performanceCounters.snapshot().haloBufferInsertionNs, 0);

  performanceCounters.haloParticlesSent = 7;
  performanceCounters.communicationBytesSent = 128;
  EXPECT_EQ(performanceCounters.snapshot().haloParticlesSent, 7);
  EXPECT_EQ(performanceCounters.snapshot().communicationBytesSent, 128);
#endif

  performanceCounters.reset();
  const auto resetSnapshot = performanceCounters.snapshot();
  EXPECT_EQ(resetSnapshot.updateContainerNs, 0);
  EXPECT_EQ(resetSnapshot.haloInitialCopyNs, 0);
  EXPECT_EQ(resetSnapshot.haloBufferInsertionNs, 0);
  EXPECT_EQ(resetSnapshot.haloParticlesSent, 0);
  EXPECT_EQ(resetSnapshot.communicationBytesSent, 0);
}

#ifndef DAP_ENABLE_TIMERS
TEST(PerformanceCountersTest, DisabledPerformanceCountersHaveNoStorage) {
  static_assert(std::is_empty_v<dap::performance::PerformanceCounters>);
  SUCCEED();
}
#endif
