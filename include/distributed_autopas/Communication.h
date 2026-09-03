#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

/**
 * Particles received from the two direct neighbors of one Cartesian dimension.
 *
 * @tparam Particle Particle type exchanged between ranks.
 */
template <class Particle>
struct LeftRightExchange {
  std::vector<Particle> recvFromLeft;
  std::vector<Particle> recvFromRight;
};

namespace detail {

/**
 * Serialize a particle vector into the transport representation provided by the serializer.
 *
 * @tparam Particle Particle type to serialize.
 * @tparam Serializer Serialization policy used for the particle type.
 * @param particles Particles to serialize.
 */
template <class Particle, class Serializer>
std::vector<typename Serializer::Message> packParticles(const std::vector<Particle> &particles) {
  std::vector<typename Serializer::Message> messages;
  messages.reserve(particles.size());
  for (const auto &particle : particles) {
    messages.push_back(Serializer::pack(particle));
  }
  return messages;
}

/**
 * Deserialize transport messages into particles.
 *
 * @tparam Particle Particle type to reconstruct.
 * @tparam Serializer Serialization policy used for the particle type.
 * @param messages Serialized particle messages.
 */
template <class Particle, class Serializer>
std::vector<Particle> unpackParticles(const std::vector<typename Serializer::Message> &messages) {
  std::vector<Particle> particles;
  particles.reserve(messages.size());
  for (const auto &message : messages) {
    particles.push_back(Serializer::unpack(message));
  }
  return particles;
}

/**
 * Translate the backend-independent missing-neighbor sentinel to the MPI representation.
 *
 * @param rank Neighbor rank or a negative value if no neighbor exists.
 */
inline int toMpiNeighbor(int rank) { return rank < 0 ? MPI_PROC_NULL : rank; }

}  // namespace detail

/**
 * Handle for an in-flight direct-neighbor particle exchange.
 *
 * The request owns all serialized send and receive buffers until the posted MPI
 * operations have completed. This is important because MPI_Isend may keep using
 * the send buffers after beginLeftRightExchange() returns.
 *
 * Requests are move-only. Normally they are completed explicitly with
 * finishLeftRightExchange(). The destructor waits for outstanding operations as
 * a safety net so that communication buffers are never destroyed while MPI may
 * still access them.
 *
 * @tparam Particle Particle type exchanged between ranks.
 * @tparam Serializer Serialization policy used for the particle type.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class LeftRightExchangeRequest {
 public:
  using Message = typename Serializer::Message;

  LeftRightExchangeRequest(const LeftRightExchangeRequest &) = delete;
  LeftRightExchangeRequest &operator=(const LeftRightExchangeRequest &) = delete;
  LeftRightExchangeRequest &operator=(LeftRightExchangeRequest &&) = delete;

  /**
   * Move an in-flight exchange together with all communication buffers and MPI requests.
   *
   * @param other Request whose ownership is transferred.
   */
  LeftRightExchangeRequest(LeftRightExchangeRequest &&other) noexcept
      : _sendLeftMessages(std::move(other._sendLeftMessages)),
        _sendRightMessages(std::move(other._sendRightMessages)),
        _recvFromLeftMessages(std::move(other._recvFromLeftMessages)),
        _recvFromRightMessages(std::move(other._recvFromRightMessages)),
        _requests(other._requests),
        _active(other._active) {
    other._requests.fill(MPI_REQUEST_NULL);
    other._active = false;
  }

  /**
   * Complete outstanding communication before destroying the owned buffers.
   */
  ~LeftRightExchangeRequest() { waitForCompletion(); }

  /**
   * Check whether payload communication is still in flight.
   */
  [[nodiscard]] bool active() const noexcept { return _active; }

 private:
  /**
   * Construct a request from serialized buffers and already posted MPI requests.
   *
   * @param sendLeftMessages Serialized particles sent to the preceding neighbor.
   * @param sendRightMessages Serialized particles sent to the succeeding neighbor.
   * @param recvFromLeftMessages Receive buffer for particles from the preceding neighbor.
   * @param recvFromRightMessages Receive buffer for particles from the succeeding neighbor.
   * @param requests Payload MPI requests owned by this request object.
   */
  LeftRightExchangeRequest(std::vector<Message> sendLeftMessages, std::vector<Message> sendRightMessages,
                           std::vector<Message> recvFromLeftMessages, std::vector<Message> recvFromRightMessages,
                           std::array<MPI_Request, 4> requests)
      : _sendLeftMessages(std::move(sendLeftMessages)),
        _sendRightMessages(std::move(sendRightMessages)),
        _recvFromLeftMessages(std::move(recvFromLeftMessages)),
        _recvFromRightMessages(std::move(recvFromRightMessages)),
        _requests(requests),
        _active(true) {}

  /**
   * Wait for all outstanding payload requests exactly once.
   */
  void waitForCompletion() {
    if (not _active) {
      return;
    }

    MPI_Waitall(static_cast<int>(_requests.size()), _requests.data(), MPI_STATUSES_IGNORE);
    _active = false;
  }

  std::vector<Message> _sendLeftMessages;
  std::vector<Message> _sendRightMessages;
  std::vector<Message> _recvFromLeftMessages;
  std::vector<Message> _recvFromRightMessages;
  std::array<MPI_Request, 4> _requests{};
  bool _active{false};

  template <class P, class S>
  friend LeftRightExchangeRequest<P, S> beginLeftRightExchange(MPI_Comm, int, int, const std::vector<P> &,
                                                               const std::vector<P> &, int);

  template <class P, class S>
  friend LeftRightExchange<P> finishLeftRightExchange(LeftRightExchangeRequest<P, S> &);
};

/**
 * Begin a particle exchange with the direct left and right neighbors.
 *
 * A short count handshake is completed first because the receive buffer sizes
 * must be known before posting the payload receives. The particle payload is
 * then transferred with MPI_Irecv / MPI_Isend and remains in flight when this
 * function returns. This creates the split phase needed for later overlap with
 * useful computation without changing migration or halo-exchange semantics yet.
 *
 * Serialization is delegated to ParticleSerializer<Particle>. This keeps the
 * distributed layer independent of the concrete particle representation used by
 * an application.
 *
 * @tparam Particle Particle type exchanged between ranks.
 * @tparam Serializer Serialization policy used for the particle type.
 * @param comm MPI communicator used for the exchange.
 * @param left Rank of the preceding neighbor or a negative value if no neighbor exists.
 * @param right Rank of the succeeding neighbor or a negative value if no neighbor exists.
 * @param sendLeft Particles to send to the preceding neighbor.
 * @param sendRight Particles to send to the succeeding neighbor.
 * @param tag Base MPI tag. The exchange uses this tag and the next three consecutive tags.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
LeftRightExchangeRequest<Particle, Serializer> beginLeftRightExchange(MPI_Comm comm, int left, int right,
                                                                      const std::vector<Particle> &sendLeft,
                                                                      const std::vector<Particle> &sendRight, int tag) {
  using Message = typename Serializer::Message;
  static_assert(std::is_trivially_copyable_v<Message>,
                "ParticleSerializer::Message must be trivially copyable for MPI_BYTE communication.");

  auto sendLeftMessages = detail::packParticles<Particle, Serializer>(sendLeft);
  auto sendRightMessages = detail::packParticles<Particle, Serializer>(sendRight);

  const int sendLeftCount = static_cast<int>(sendLeftMessages.size());
  const int sendRightCount = static_cast<int>(sendRightMessages.size());

  // The domain decomposition uses negative ranks to represent a missing neighbor
  // at a non-periodic global boundary. Translate that backend-independent sentinel
  // to MPI_PROC_NULL here so callers do not need MPI-specific boundary handling.
  const int mpiLeft = detail::toMpiNeighbor(left);
  const int mpiRight = detail::toMpiNeighbor(right);

  int recvFromLeftCount = 0;
  int recvFromRightCount = 0;

  // The count handshake is tiny but must finish before receive payload buffers can
  // be allocated. The potentially large particle payload is what remains in flight
  // after beginLeftRightExchange() returns.
  std::array<MPI_Request, 4> countRequests{};
  MPI_Irecv(&recvFromRightCount, 1, MPI_INT, mpiRight, tag, comm, &countRequests[0]);
  MPI_Irecv(&recvFromLeftCount, 1, MPI_INT, mpiLeft, tag + 1, comm, &countRequests[1]);
  MPI_Isend(&sendLeftCount, 1, MPI_INT, mpiLeft, tag, comm, &countRequests[2]);
  MPI_Isend(&sendRightCount, 1, MPI_INT, mpiRight, tag + 1, comm, &countRequests[3]);
  MPI_Waitall(static_cast<int>(countRequests.size()), countRequests.data(), MPI_STATUSES_IGNORE);

  std::vector<Message> recvFromLeftMessages(recvFromLeftCount);
  std::vector<Message> recvFromRightMessages(recvFromRightCount);

  std::array<MPI_Request, 4> payloadRequests{};
  MPI_Irecv(recvFromRightMessages.data(), recvFromRightCount * static_cast<int>(sizeof(Message)), MPI_BYTE, mpiRight,
            tag + 2, comm, &payloadRequests[0]);
  MPI_Irecv(recvFromLeftMessages.data(), recvFromLeftCount * static_cast<int>(sizeof(Message)), MPI_BYTE, mpiLeft,
            tag + 3, comm, &payloadRequests[1]);
  MPI_Isend(sendLeftMessages.data(), sendLeftCount * static_cast<int>(sizeof(Message)), MPI_BYTE, mpiLeft, tag + 2,
            comm, &payloadRequests[2]);
  MPI_Isend(sendRightMessages.data(), sendRightCount * static_cast<int>(sizeof(Message)), MPI_BYTE, mpiRight, tag + 3,
            comm, &payloadRequests[3]);

  return LeftRightExchangeRequest<Particle, Serializer>{std::move(sendLeftMessages), std::move(sendRightMessages),
                                                        std::move(recvFromLeftMessages),
                                                        std::move(recvFromRightMessages), payloadRequests};
}

/**
 * Complete an exchange started with beginLeftRightExchange() and deserialize the
 * received particles.
 *
 * @tparam Particle Particle type exchanged between ranks.
 * @tparam Serializer Serialization policy used for the particle type.
 * @param request In-flight request to complete.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
LeftRightExchange<Particle> finishLeftRightExchange(LeftRightExchangeRequest<Particle, Serializer> &request) {
  if (not request._active) {
    throw std::logic_error("finishLeftRightExchange(): request has already been completed.");
  }

  request.waitForCompletion();

  return LeftRightExchange<Particle>{detail::unpackParticles<Particle, Serializer>(request._recvFromLeftMessages),
                                     detail::unpackParticles<Particle, Serializer>(request._recvFromRightMessages)};
}

/**
 * Blocking reference exchange with the direct left and right neighbors.
 *
 * This intentionally retains the previous MPI_Sendrecv implementation as a
 * reference path. Timestep migration and halo exchange use the split-phase
 * non-blocking transport, while this function remains available for correctness
 * and later performance comparisons.
 *
 * @tparam Particle Particle type exchanged between ranks.
 * @tparam Serializer Serialization policy used for the particle type.
 * @param comm MPI communicator used for the exchange.
 * @param left Rank of the preceding neighbor or a negative value if no neighbor exists.
 * @param right Rank of the succeeding neighbor or a negative value if no neighbor exists.
 * @param sendLeft Particles to send to the preceding neighbor.
 * @param sendRight Particles to send to the succeeding neighbor.
 * @param tag Base MPI tag. The exchange uses this tag and the next three consecutive tags.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
LeftRightExchange<Particle> exchangeLeftRight(MPI_Comm comm, int left, int right, const std::vector<Particle> &sendLeft,
                                              const std::vector<Particle> &sendRight, int tag) {
  using Message = typename Serializer::Message;
  static_assert(std::is_trivially_copyable_v<Message>,
                "ParticleSerializer::Message must be trivially copyable for MPI_BYTE communication.");

  const auto sendLeftMessages = detail::packParticles<Particle, Serializer>(sendLeft);
  const auto sendRightMessages = detail::packParticles<Particle, Serializer>(sendRight);

  const int sendLeftCount = static_cast<int>(sendLeftMessages.size());
  const int sendRightCount = static_cast<int>(sendRightMessages.size());

  const int mpiLeft = detail::toMpiNeighbor(left);
  const int mpiRight = detail::toMpiNeighbor(right);

  int recvFromLeftCount = 0;
  int recvFromRightCount = 0;

  MPI_Sendrecv(&sendLeftCount, 1, MPI_INT, mpiLeft, tag, &recvFromRightCount, 1, MPI_INT, mpiRight, tag, comm,
               MPI_STATUS_IGNORE);

  MPI_Sendrecv(&sendRightCount, 1, MPI_INT, mpiRight, tag + 1, &recvFromLeftCount, 1, MPI_INT, mpiLeft, tag + 1, comm,
               MPI_STATUS_IGNORE);

  std::vector<Message> recvFromLeftMessages(recvFromLeftCount);
  std::vector<Message> recvFromRightMessages(recvFromRightCount);

  MPI_Sendrecv(sendLeftMessages.data(), sendLeftCount * static_cast<int>(sizeof(Message)), MPI_BYTE, mpiLeft, tag + 2,
               recvFromRightMessages.data(), recvFromRightCount * static_cast<int>(sizeof(Message)), MPI_BYTE, mpiRight,
               tag + 2, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sendRightMessages.data(), sendRightCount * static_cast<int>(sizeof(Message)), MPI_BYTE, mpiRight,
               tag + 3, recvFromLeftMessages.data(), recvFromLeftCount * static_cast<int>(sizeof(Message)), MPI_BYTE,
               mpiLeft, tag + 3, comm, MPI_STATUS_IGNORE);

  return LeftRightExchange<Particle>{detail::unpackParticles<Particle, Serializer>(recvFromLeftMessages),
                                     detail::unpackParticles<Particle, Serializer>(recvFromRightMessages)};
}

/**
 * Redistribute particles to arbitrary destination ranks.
 *
 * Each entry in particlesByDestination contains the particles that the calling rank wants to send
 * to the rank with the same index. The returned vector contains all particles sent to the calling
 * rank by all ranks, including particles routed to itself.
 *
 * This collective is primarily used for initialization and checkpoint redistribution where a
 * particle may need to move directly to any rank. Timestep migration intentionally uses the
 * cheaper direct-neighbor exchange instead.
 *
 * @tparam Particle Particle type exchanged between ranks.
 * @tparam Serializer Serialization policy used for the particle type.
 * @param comm MPI communicator used for the collective redistribution.
 * @param particlesByDestination One particle buffer per destination rank.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
std::vector<Particle> exchangeParticlesByRank(MPI_Comm comm,
                                              const std::vector<std::vector<Particle>> &particlesByDestination) {
  using Message = typename Serializer::Message;
  static_assert(std::is_trivially_copyable_v<Message>,
                "ParticleSerializer::Message must be trivially copyable for MPI_BYTE communication.");

  int numberOfRanks = 0;
  MPI_Comm_size(comm, &numberOfRanks);

  if (particlesByDestination.size() != static_cast<std::size_t>(numberOfRanks)) {
    throw std::invalid_argument("exchangeParticlesByRank(): one destination buffer per rank is required.");
  }

  std::vector<int> sendCounts(numberOfRanks, 0);
  std::vector<int> sendDisplacements(numberOfRanks, 0);
  std::vector<Message> sendMessages;

  std::size_t totalSendMessages = 0;
  for (const auto &particles : particlesByDestination) {
    totalSendMessages += particles.size();
  }
  sendMessages.reserve(totalSendMessages);

  for (int destination = 0; destination < numberOfRanks; ++destination) {
    sendDisplacements[destination] = static_cast<int>(sendMessages.size() * sizeof(Message));
    sendCounts[destination] = static_cast<int>(particlesByDestination[destination].size() * sizeof(Message));

    for (const auto &particle : particlesByDestination[destination]) {
      sendMessages.push_back(Serializer::pack(particle));
    }
  }

  std::vector<int> receiveCounts(numberOfRanks, 0);
  MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, comm);

  std::vector<int> receiveDisplacements(numberOfRanks, 0);
  int totalReceiveBytes = 0;
  for (int source = 0; source < numberOfRanks; ++source) {
    receiveDisplacements[source] = totalReceiveBytes;
    totalReceiveBytes += receiveCounts[source];
  }

  std::vector<Message> receiveMessages(static_cast<std::size_t>(totalReceiveBytes) / sizeof(Message));

  MPI_Alltoallv(sendMessages.data(), sendCounts.data(), sendDisplacements.data(), MPI_BYTE, receiveMessages.data(),
                receiveCounts.data(), receiveDisplacements.data(), MPI_BYTE, comm);

  std::vector<Particle> receivedParticles;
  receivedParticles.reserve(receiveMessages.size());
  for (const auto &message : receiveMessages) {
    receivedParticles.push_back(Serializer::unpack(message));
  }

  return receivedParticles;
}

}  // namespace dap
