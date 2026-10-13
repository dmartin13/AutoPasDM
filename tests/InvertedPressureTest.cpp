#include <gtest/gtest.h>

#include "autopasdm/load_balancing/InvertedPressure.h"

namespace {

TEST(InvertedPressureTest, KeepsBoundaryCenteredForEqualWork) {
  EXPECT_DOUBLE_EQ(autopasdm::load_balancing::balanceAdjacentDomains(1., 1., 0., 10., 1.), 5.);
}

TEST(InvertedPressureTest, ShrinksMoreExpensiveDomain) {
  EXPECT_DOUBLE_EQ(autopasdm::load_balancing::balanceAdjacentDomains(3., 1., 0., 10., 1.), 2.5);
  EXPECT_DOUBLE_EQ(autopasdm::load_balancing::balanceAdjacentDomains(1., 3., 0., 10., 1.), 7.5);
}

TEST(InvertedPressureTest, EnforcesMinimumWidthForLeftDomain) {
  EXPECT_DOUBLE_EQ(autopasdm::load_balancing::balanceAdjacentDomains(99., 1., 0., 10., 2.), 2.);
}

TEST(InvertedPressureTest, EnforcesMinimumWidthForRightDomain) {
  EXPECT_DOUBLE_EQ(autopasdm::load_balancing::balanceAdjacentDomains(1., 99., 0., 10., 2.), 8.);
}

}  // namespace
