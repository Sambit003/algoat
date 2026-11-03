#include "algoat/numerics/bigint.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <string>
#include <vector>

using namespace algoat::numerics;

namespace {

BigInt make_random_bigint(size_t limbs, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::string hex = "0x";
    for (size_t i = 0; i < limbs; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(rng()));
        hex += buf;
    }
    return BigInt(hex, 16);
}

static void BM_Schoolbook_Multiply(benchmark::State& state) {
    const size_t bits = state.range(0);
    const size_t limbs = bits / 64;
    BigInt a = make_random_bigint(limbs, 101);
    BigInt b = make_random_bigint(limbs, 202);

    for (auto _ : state) {
        BigInt res = BigInt::multiply_schoolbook(a, b);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Schoolbook_Multiply)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);

static void BM_Karatsuba_Multiply(benchmark::State& state) {
    const size_t bits = state.range(0);
    const size_t limbs = bits / 64;
    BigInt a = make_random_bigint(limbs, 101);
    BigInt b = make_random_bigint(limbs, 202);

    for (auto _ : state) {
        BigInt res = BigInt::multiply_karatsuba(a, b);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Karatsuba_Multiply)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);

static void BM_RSA2048_ModPow(benchmark::State& state) {
    std::string mod_hex = "0x"
                          "c000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000001";
    BigInt n(mod_hex, 16);
    BigInt a("123456789012345678901234567890");
    BigInt e(65537);

    for (auto _ : state) {
        BigInt res = a.mod_pow(e, n);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RSA2048_ModPow);

static void BM_RSA4096_ModPow(benchmark::State& state) {
    std::string mod_hex = "0x"
                          "c000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000001";
    BigInt n(mod_hex, 16);
    BigInt a("123456789012345678901234567890");
    BigInt e(65537);

    for (auto _ : state) {
        BigInt res = a.mod_pow(e, n);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RSA4096_ModPow);

} // namespace
