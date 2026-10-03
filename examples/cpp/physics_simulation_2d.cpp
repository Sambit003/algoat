#include <algoat/algoat.hpp>
#include <algoat/numerics/hilbert.hpp>
#include <algoat/numerics/hybrid_sfc.hpp>
#include <algoat/numerics/morton.hpp>
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
#include <span>
#include <vector>

// Realistic Use Case: A physics simulation generating millions of 2D coordinates (Complex Numbers)
// We need to spatially sort them for a Bounding Volume Hierarchy or spatial grid lookups.
int main() {
    std::cout << "--- Algoat Realistic Physics Simulation Example ---\n";
    constexpr size_t NUM_PARTICLES = 1'000'000;

    std::vector<std::complex<float>> particle_positions(NUM_PARTICLES);

    // 1. Generate realistic 2D particle cluster data
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist_x(0.0f, 50.0f);
    std::normal_distribution<float> dist_y(0.0f, 50.0f);

    for (size_t i = 0; i < NUM_PARTICLES; ++i) {
        particle_positions[i] = {dist_x(gen), dist_y(gen)};
    }

    std::cout << "Generated " << NUM_PARTICLES << " 2D particles.\n";
    std::cout << "Sorting particles by Morton Z-order for spatial locality...\n";

    // 2. Sort the particles using Algoat's specialized numerics backend
    auto start_time = std::chrono::high_resolution_clock::now();

    // Algoat recognizes std::complex<float> and bypasses standard lexicographical O(N log N)
    // sorting. It routes to the O(N) 16-bit Radix Sort over Morton Z-order curves.
    algoat::numerics::sort_complex_morton(std::span<std::complex<float>>{particle_positions});

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Successfully sorted 1 Million particles in " << duration_ms.count() << " ms.\n";

    // Demonstrate other curves
    std::cout << "Sorting particles by continuous Hilbert curve for maximum cache locality...\n";
    start_time = std::chrono::high_resolution_clock::now();
    algoat::numerics::sort_complex_hilbert(std::span<std::complex<float>>{particle_positions});
    end_time = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Successfully sorted 1 Million particles with Hilbert in " << duration_ms.count()
              << " ms.\n";

    std::cout << "Sorting particles by Hybrid Morton-Hilbert curve for balanced performance...\n";
    start_time = std::chrono::high_resolution_clock::now();
    algoat::numerics::sort_complex_hybrid(std::span<std::complex<float>>{particle_positions});
    end_time = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Successfully sorted 1 Million particles with Hybrid in " << duration_ms.count()
              << " ms.\n";

    std::cout << "Spatial clustering is now highly optimized for subsequent physics processing!\n";

    return 0;
}
