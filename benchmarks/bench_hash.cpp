#include <algoat/hash/hash.hpp>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace algoat::hash;

namespace {

std::vector<std::byte> make_buffer(std::size_t size) {
    std::vector<std::byte> buf(size);
    for (std::size_t i = 0; i < size; ++i) {
        buf[i] = static_cast<std::byte>(i * 31 + 7);
    }
    return buf;
}

const HighwayKey kBenchKey = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL, 0x1122334455667788ULL,
                              0x99AABBCCDDEEFF00ULL};

} // namespace

// ─── xxHash3_64 Throughput & Latency ──────────────────────────────────────────

static void BM_xxhash3_64(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    auto buf = make_buffer(size);

    for (auto _ : state) {
        buf[0] = static_cast<std::byte>(state.iterations());
        uint64_t h = xxhash3_64(std::span<const std::byte>{buf.data(), buf.size()}, 0);
        benchmark::DoNotOptimize(h);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_xxhash3_64)->RangeMultiplier(4)->Range(16, 1 << 20);

// ─── xxHash3_128 Throughput & Latency ─────────────────────────────────────────

static void BM_xxhash3_128(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    auto buf = make_buffer(size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(buf.data());
        auto h = xxhash3_128(std::span<const std::byte>{buf.data(), buf.size()}, 0);
        benchmark::DoNotOptimize(h);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_xxhash3_128)->RangeMultiplier(4)->Range(16, 1 << 20);

// ─── WyHash Latency & Throughput ──────────────────────────────────────────────

static void BM_wyhash(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    auto buf = make_buffer(size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(buf.data());
        uint64_t h = wyhash(std::span<const std::byte>{buf.data(), buf.size()}, 0);
        benchmark::DoNotOptimize(h);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_wyhash)->RangeMultiplier(4)->Range(16, 1 << 20);

// ─── HighwayHash_64 Throughput & Latency ──────────────────────────────────────

static void BM_highway_hash_64(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    auto buf = make_buffer(size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(buf.data());
        uint64_t h = highway_hash_64(std::span<const std::byte>{buf.data(), buf.size()}, kBenchKey);
        benchmark::DoNotOptimize(h);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_highway_hash_64)->RangeMultiplier(4)->Range(16, 1 << 20);

// ─── HighwayHash_128 Throughput & Latency ─────────────────────────────────────

static void BM_highway_hash_128(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    auto buf = make_buffer(size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(buf.data());
        auto h = highway_hash_128(std::span<const std::byte>{buf.data(), buf.size()}, kBenchKey);
        benchmark::DoNotOptimize(h);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_highway_hash_128)->RangeMultiplier(4)->Range(16, 1 << 20);

// ─── HighwayHash_256 Throughput & Latency ─────────────────────────────────────

static void BM_highway_hash_256(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    auto buf = make_buffer(size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(buf.data());
        auto h = highway_hash_256(std::span<const std::byte>{buf.data(), buf.size()}, kBenchKey);
        benchmark::DoNotOptimize(h);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_highway_hash_256)->RangeMultiplier(4)->Range(16, 1 << 20);

// ─── CPO Functor Drop-in Latency (16-byte short key) ──────────────────────────

static void BM_CPO_XxHash3_16B(benchmark::State& state) {
    std::string key = "short_key_16byte";
    XxHash3Hasher<std::string> hasher;

    for (auto _ : state) {
        std::size_t h = hasher(key);
        benchmark::DoNotOptimize(h);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 16);
}
BENCHMARK(BM_CPO_XxHash3_16B);

static void BM_CPO_WyHash_16B(benchmark::State& state) {
    std::string key = "short_key_16byte";
    WyHasher<std::string> hasher;

    for (auto _ : state) {
        std::size_t h = hasher(key);
        benchmark::DoNotOptimize(h);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 16);
}
BENCHMARK(BM_CPO_WyHash_16B);
