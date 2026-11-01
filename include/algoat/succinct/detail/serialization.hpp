#pragma once

#include "algoat/succinct/detail/containers.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace algoat::succinct::detail {

inline constexpr uint16_t SERIAL_COOKIE_NO_RUNCONTAINER = 12346; // 0x303A
inline constexpr uint16_t SERIAL_COOKIE = 12347;                 // 0x303B
inline constexpr size_t NO_OFFSET_THRESHOLD = 4;

struct Serializer {
    static std::vector<uint8_t>
    serialize(const std::vector<uint16_t>& keys,
              const std::vector<std::shared_ptr<Container>>& containers) {
        const size_t N = containers.size();
        if (N == 0) {
            std::vector<uint8_t> out(8, 0);
            const uint32_t cookie = SERIAL_COOKIE_NO_RUNCONTAINER;
            std::memcpy(out.data(), &cookie, 4);
            const uint32_t count = 0;
            std::memcpy(out.data() + 4, &count, 4);
            return out;
        }

        bool has_run = false;
        for (const auto& c : containers) {
            if (c->type() == ContainerType::Run) {
                has_run = true;
                break;
            }
        }

        std::vector<uint8_t> out;

        if (!has_run) {
            const uint32_t cookie = SERIAL_COOKIE_NO_RUNCONTAINER;
            const uint32_t count = static_cast<uint32_t>(N);
            const size_t header_size = 8 + 4 * N + 4 * N;

            size_t total_payload_size = 0;
            for (const auto& c : containers) {
                if (c->type() == ContainerType::Array) {
                    total_payload_size += c->cardinality() * sizeof(uint16_t);
                } else if (c->type() == ContainerType::Bitset) {
                    total_payload_size += BITSET_BYTES;
                }
            }

            out.resize(header_size + total_payload_size);

            std::memcpy(out.data(), &cookie, 4);
            std::memcpy(out.data() + 4, &count, 4);

            size_t desc_offset = 8;
            size_t offset_header_pos = 8 + 4 * N;
            size_t payload_offset = header_size;

            for (size_t k = 0; k < N; ++k) {
                const uint16_t key = keys[k];
                const uint16_t card_minus_1 =
                    static_cast<uint16_t>(containers[k]->cardinality() - 1);
                std::memcpy(out.data() + desc_offset + 4 * k, &key, 2);
                std::memcpy(out.data() + desc_offset + 4 * k + 2, &card_minus_1, 2);

                const uint32_t off = static_cast<uint32_t>(payload_offset);
                std::memcpy(out.data() + offset_header_pos + 4 * k, &off, 4);

                if (containers[k]->type() == ContainerType::Array) {
                    const auto* ac = static_cast<const ArrayContainer*>(containers[k].get());
                    const size_t bytes = ac->values.size() * sizeof(uint16_t);
                    std::memcpy(out.data() + payload_offset, ac->values.data(), bytes);
                    payload_offset += bytes;
                } else if (containers[k]->type() == ContainerType::Bitset) {
                    const auto* bc = static_cast<const BitsetContainer*>(containers[k].get());
                    std::memcpy(out.data() + payload_offset, bc->words.data(), BITSET_BYTES);
                    payload_offset += BITSET_BYTES;
                }
            }
        } else {
            const uint32_t word0 = SERIAL_COOKIE | (static_cast<uint32_t>(N - 1) << 16);
            const size_t run_bitmap_bytes = (N + 7) / 8;
            const bool has_offsets = (N >= NO_OFFSET_THRESHOLD);
            const size_t header_size = 4 + run_bitmap_bytes + 4 * N + (has_offsets ? 4 * N : 0);

            size_t total_payload_size = 0;
            for (const auto& c : containers) {
                if (c->type() == ContainerType::Array) {
                    total_payload_size += c->cardinality() * sizeof(uint16_t);
                } else if (c->type() == ContainerType::Bitset) {
                    total_payload_size += BITSET_BYTES;
                } else if (c->type() == ContainerType::Run) {
                    const auto* rc = static_cast<const RunContainer*>(c.get());
                    total_payload_size += 2 + rc->runs.size() * 4;
                }
            }

            out.resize(header_size + total_payload_size);

            std::memcpy(out.data(), &word0, 4);

            uint8_t* run_bitmap = out.data() + 4;
            std::memset(run_bitmap, 0, run_bitmap_bytes);
            for (size_t k = 0; k < N; ++k) {
                if (containers[k]->type() == ContainerType::Run) {
                    run_bitmap[k / 8] |= static_cast<uint8_t>(1 << (k % 8));
                }
            }

            const size_t desc_offset = 4 + run_bitmap_bytes;
            const size_t offset_header_pos = desc_offset + 4 * N;
            size_t payload_offset = header_size;

            for (size_t k = 0; k < N; ++k) {
                const uint16_t key = keys[k];
                const uint16_t card_minus_1 =
                    static_cast<uint16_t>(containers[k]->cardinality() - 1);
                std::memcpy(out.data() + desc_offset + 4 * k, &key, 2);
                std::memcpy(out.data() + desc_offset + 4 * k + 2, &card_minus_1, 2);

                if (has_offsets) {
                    const uint32_t off = static_cast<uint32_t>(payload_offset);
                    std::memcpy(out.data() + offset_header_pos + 4 * k, &off, 4);
                }

                if (containers[k]->type() == ContainerType::Array) {
                    const auto* ac = static_cast<const ArrayContainer*>(containers[k].get());
                    const size_t bytes = ac->values.size() * sizeof(uint16_t);
                    std::memcpy(out.data() + payload_offset, ac->values.data(), bytes);
                    payload_offset += bytes;
                } else if (containers[k]->type() == ContainerType::Bitset) {
                    const auto* bc = static_cast<const BitsetContainer*>(containers[k].get());
                    std::memcpy(out.data() + payload_offset, bc->words.data(), BITSET_BYTES);
                    payload_offset += BITSET_BYTES;
                } else if (containers[k]->type() == ContainerType::Run) {
                    const auto* rc = static_cast<const RunContainer*>(containers[k].get());
                    const uint16_t n_runs = static_cast<uint16_t>(rc->runs.size());
                    std::memcpy(out.data() + payload_offset, &n_runs, 2);
                    payload_offset += 2;
                    for (const auto& r : rc->runs) {
                        std::memcpy(out.data() + payload_offset, &r.start, 2);
                        std::memcpy(out.data() + payload_offset + 2, &r.length, 2);
                        payload_offset += 4;
                    }
                }
            }
        }

        return out;
    }

    static void deserialize(std::span<const uint8_t> bytes, std::vector<uint16_t>& keys,
                            std::vector<std::shared_ptr<Container>>& containers) {
        keys.clear();
        containers.clear();

        if (bytes.size() < 4) {
            throw std::runtime_error("Corrupted Roaring bitmap: buffer too small");
        }

        uint32_t word0 = 0;
        std::memcpy(&word0, bytes.data(), 4);

        if (word0 == SERIAL_COOKIE_NO_RUNCONTAINER) {
            if (bytes.size() < 8) {
                throw std::runtime_error("Corrupted Roaring bitmap: missing container count");
            }
            uint32_t count = 0;
            std::memcpy(&count, bytes.data() + 4, 4);
            if (count == 0)
                return;

            const size_t N = count;
            const size_t header_size = 8 + 4 * N + 4 * N;
            if (bytes.size() < header_size) {
                throw std::runtime_error("Corrupted Roaring bitmap: header truncated");
            }

            keys.resize(N);
            containers.resize(N);

            std::vector<uint32_t> cardinalities(N);
            const size_t desc_offset = 8;
            for (size_t k = 0; k < N; ++k) {
                uint16_t key = 0;
                uint16_t card_minus_1 = 0;
                std::memcpy(&key, bytes.data() + desc_offset + 4 * k, 2);
                std::memcpy(&card_minus_1, bytes.data() + desc_offset + 4 * k + 2, 2);
                keys[k] = key;
                cardinalities[k] = static_cast<uint32_t>(card_minus_1) + 1;
            }

            const size_t offset_header_pos = 8 + 4 * N;
            std::vector<uint32_t> offsets(N);
            for (size_t k = 0; k < N; ++k) {
                std::memcpy(&offsets[k], bytes.data() + offset_header_pos + 4 * k, 4);
            }

            for (size_t k = 0; k < N; ++k) {
                const size_t off = offsets[k];
                const uint32_t card = cardinalities[k];

                if (card <= ARRAY_MAX_CAPACITY) {
                    const size_t req_bytes = card * sizeof(uint16_t);
                    if (off + req_bytes > bytes.size()) {
                        throw std::runtime_error(
                            "Corrupted Roaring bitmap: ArrayContainer payload truncated");
                    }
                    auto ac = std::make_shared<ArrayContainer>();
                    ac->values.resize(card);
                    std::memcpy(ac->values.data(), bytes.data() + off, req_bytes);
                    containers[k] = ac;
                } else {
                    if (off + BITSET_BYTES > bytes.size()) {
                        throw std::runtime_error(
                            "Corrupted Roaring bitmap: BitsetContainer payload truncated");
                    }
                    auto bc = std::make_shared<BitsetContainer>();
                    std::memcpy(bc->words.data(), bytes.data() + off, BITSET_BYTES);
                    bc->card_ = card;
                    containers[k] = bc;
                }
            }
        } else if ((word0 & 0xFFFF) == SERIAL_COOKIE) {
            const size_t N = static_cast<size_t>(word0 >> 16) + 1;
            const size_t run_bitmap_bytes = (N + 7) / 8;
            const bool has_offsets = (N >= NO_OFFSET_THRESHOLD);
            const size_t header_size = 4 + run_bitmap_bytes + 4 * N + (has_offsets ? 4 * N : 0);

            if (bytes.size() < header_size) {
                throw std::runtime_error("Corrupted Roaring bitmap: header truncated");
            }

            const uint8_t* run_bitmap = bytes.data() + 4;
            keys.resize(N);
            containers.resize(N);

            std::vector<uint32_t> cardinalities(N);
            const size_t desc_offset = 4 + run_bitmap_bytes;
            for (size_t k = 0; k < N; ++k) {
                uint16_t key = 0;
                uint16_t card_minus_1 = 0;
                std::memcpy(&key, bytes.data() + desc_offset + 4 * k, 2);
                std::memcpy(&card_minus_1, bytes.data() + desc_offset + 4 * k + 2, 2);
                keys[k] = key;
                cardinalities[k] = static_cast<uint32_t>(card_minus_1) + 1;
            }

            size_t payload_offset = header_size;
            const size_t offset_header_pos = desc_offset + 4 * N;

            for (size_t k = 0; k < N; ++k) {
                if (has_offsets) {
                    uint32_t off = 0;
                    std::memcpy(&off, bytes.data() + offset_header_pos + 4 * k, 4);
                    payload_offset = off;
                }

                const bool is_run = (run_bitmap[k / 8] & (1 << (k % 8))) != 0;
                const uint32_t card = cardinalities[k];

                if (is_run) {
                    if (payload_offset + 2 > bytes.size()) {
                        throw std::runtime_error(
                            "Corrupted Roaring bitmap: RunContainer header truncated");
                    }
                    uint16_t n_runs = 0;
                    std::memcpy(&n_runs, bytes.data() + payload_offset, 2);
                    payload_offset += 2;

                    const size_t req_bytes = static_cast<size_t>(n_runs) * 4;
                    if (payload_offset + req_bytes > bytes.size()) {
                        throw std::runtime_error(
                            "Corrupted Roaring bitmap: RunContainer payload truncated");
                    }

                    auto rc = std::make_shared<RunContainer>();
                    rc->runs.resize(n_runs);
                    for (size_t r = 0; r < n_runs; ++r) {
                        std::memcpy(&rc->runs[r].start, bytes.data() + payload_offset, 2);
                        std::memcpy(&rc->runs[r].length, bytes.data() + payload_offset + 2, 2);
                        payload_offset += 4;
                    }
                    containers[k] = rc;
                } else if (card <= ARRAY_MAX_CAPACITY) {
                    const size_t req_bytes = card * sizeof(uint16_t);
                    if (payload_offset + req_bytes > bytes.size()) {
                        throw std::runtime_error(
                            "Corrupted Roaring bitmap: ArrayContainer payload truncated");
                    }
                    auto ac = std::make_shared<ArrayContainer>();
                    ac->values.resize(card);
                    std::memcpy(ac->values.data(), bytes.data() + payload_offset, req_bytes);
                    payload_offset += req_bytes;
                    containers[k] = ac;
                } else {
                    if (payload_offset + BITSET_BYTES > bytes.size()) {
                        throw std::runtime_error(
                            "Corrupted Roaring bitmap: BitsetContainer payload truncated");
                    }
                    auto bc = std::make_shared<BitsetContainer>();
                    std::memcpy(bc->words.data(), bytes.data() + payload_offset, BITSET_BYTES);
                    bc->card_ = card;
                    payload_offset += BITSET_BYTES;
                    containers[k] = bc;
                }
            }
        } else {
            throw std::runtime_error("Corrupted Roaring bitmap: invalid cookie");
        }
    }
};

} // namespace algoat::succinct::detail
