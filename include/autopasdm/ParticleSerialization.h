#pragma once

namespace autopasdm {

/**
 * Serialization customization point for particles communicated by AutoPasDM.
 *
 * Applications using a custom particle type have to specialize this template and provide:
 *
 *   using Message = ...;
 *   static Message pack(const Particle &particle);
 *   static Particle unpack(const Message &message);
 *
 * Message must be trivially copyable because the current communication layer transfers
 * messages as MPI_BYTE.
 *
 * @tparam Particle AutoPas-compatible particle type to serialize.
 */
template <class Particle>
struct ParticleSerializer;

}  // namespace autopasdm
