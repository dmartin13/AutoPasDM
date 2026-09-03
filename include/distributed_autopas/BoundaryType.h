#pragma once

namespace dap {

/**
 * Boundary condition applied at the global simulation-box boundary.
 */
enum class BoundaryType {
  /** Particles and halos wrap to the opposite side of the global box. */
  periodic,

  /** Non-periodic communication boundary. The physical reflection force is application-defined. */
  reflective,

  /** No boundary handling. Particles leaving the global box are discarded. */
  none
};

}  // namespace dap
