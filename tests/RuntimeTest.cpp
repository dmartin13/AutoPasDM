#include <gtest/gtest.h>

#include <string>

#include "autopasdm/Runtime.h"

namespace {

TEST(RuntimeTest, ReportsRankSizeAndRoot) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);

  EXPECT_EQ(runtime.size(), 4);
  EXPECT_GE(runtime.rank(), 0);
  EXPECT_LT(runtime.rank(), runtime.size());
  EXPECT_EQ(runtime.isRoot(), runtime.rank() == 0);
}

TEST(RuntimeTest, BroadcastsStringFromNonzeroRoot) {
  int argc = 0;
  char **argv = nullptr;
  autopasdm::Runtime runtime(argc, argv);

  EXPECT_EQ(runtime.size(), 4);

  constexpr int rootRank = 2;
  std::string value;
  if (runtime.rank() == rootRank) {
    value = "distributed-autopas";
  }

  runtime.broadcastString(value, rootRank);

  EXPECT_EQ(value, "distributed-autopas");
}

}  // namespace
