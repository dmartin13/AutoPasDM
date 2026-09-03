#pragma once

namespace dap::load_balancing {

/**
 * Calculate the balanced position of the shared boundary between two adjacent domains.
 *
 * The inverted-pressure model assumes that work should be proportional to domain volume.
 * For two neighboring regular-grid cells this reduces to balancing their widths in the
 * selected dimension according to the measured work. A more expensive domain therefore
 * shrinks while the cheaper domain grows.
 *
 * The returned boundary is clamped such that both adjacent domains retain at least
 * minWidth in the selected dimension.
 *
 * @param leftWork Work measured for the domain to the left of the shared boundary.
 * @param rightWork Work measured for the domain to the right of the shared boundary.
 * @param leftMin Minimum boundary of the left domain in the selected dimension.
 * @param rightMax Maximum boundary of the right domain in the selected dimension.
 * @param minWidth Minimum allowed width of either adjacent domain.
 */
inline double balanceAdjacentDomains(double leftWork, double rightWork, double leftMin, double rightMax,
                                     double minWidth) {
  const double balancedPosition = (leftWork * leftMin + rightWork * rightMax) / (leftWork + rightWork);

  if (balancedPosition - leftMin < minWidth) {
    return leftMin + minWidth;
  }

  if (rightMax - balancedPosition < minWidth) {
    return rightMax - minWidth;
  }

  return balancedPosition;
}

}  // namespace dap::load_balancing
