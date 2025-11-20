#include "../src/algorithms.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <cstring>

// Defaults
const int SAMPLES = 500;
const int N_MIN = 1;
const int N_MAX = 2000;
const int SKIP_TRIPLE_LOOP_ABOVE = 500;
const int SKIP_STRASSEN_ABOVE = 1024; // Skip Strassen for very large matrices to save time
const int STRASSEN_THRESHOLD = 64;
const std::vector<int> FRIEVALD_KS = {1, 10, 20};

struct Record {
    int sample_id;
    int n;
    std::string algorithm;
    double time_seconds;
    int k; // for frievald
};

double measure_time(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int main(int argc, char* argv[]) {
    bool append_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--append") == 0) {
            append_mode = true;
        }
    }

    std::mt19937 rng(90);
    std::uniform_int_distribution<int> dist_n(N_MIN, N_MAX);

    std::ofstream csv_file;
    if (append_mode) {
        csv_file.open("experiments/results/runtime.csv", std::ios::app);
        std::cout << "Appending to experiments/results/runtime.csv" << std::endl;
    } else {
        csv_file.open("experiments/results/runtime.csv");
        csv_file << "sample_id,n,algorithm,seconds,k\n";
        std::cout << "Overwriting experiments/results/runtime.csv" << std::endl;
    }

    std::cout << "Starting C++ Runtime Benchmark..." << std::endl;

    for (int s = 0; s < SAMPLES; ++s) {
        int n = dist_n(rng);
        // Generate matrices
        Matrix A = Matrix::random(n, n, rng);
        Matrix B = Matrix::random(n, n, rng);
        Matrix C_correct(n, n); // Placeholder

        // We need a correct C for Frievald to not exit early (to measure full verification time)
        // We use Strassen to compute it efficiently.
        // However, we also want to benchmark Strassen, so we can do that first.

        // 1. Benchmark Strassen
        if (n <= SKIP_STRASSEN_ABOVE) {
            double t_strassen = measure_time([&]() {
                C_correct = matmul_strassen(A, B, STRASSEN_THRESHOLD);
            });
            csv_file << s << "," << n << ",strassen," << std::fixed << std::setprecision(9) << t_strassen << ",\n";
        } else {
            // Skip Strassen, use dummy C for Frievald
            // We don't need C_correct to be correct if we force Frievald iterations
            // But we should initialize it to correct size
            // C_correct is already n x n (initialized to 0)
        }

        // 2. Benchmark Triple Loop
        if (n <= SKIP_TRIPLE_LOOP_ABOVE) {
            double t_triple = measure_time([&]() {
                matmul_triple_loop(A, B);
            });
            csv_file << s << "," << n << ",triple_loop," << t_triple << ",\n";
        }

        // 3. Benchmark Frievald
        // If we skipped Strassen, C_correct is all zeros. Frievald will detect error.
        // We must force iterations to measure runtime.
        bool force = (n > SKIP_STRASSEN_ABOVE);
        
        for (int k : FRIEVALD_KS) {
            double t_frievald = measure_time([&]() {
                frievald_verify(A, B, C_correct, k, rng, force);
            });
            csv_file << s << "," << n << ",frievald," << t_frievald << "," << k << "\n";
        }

        if ((s + 1) % 1 == 0) {
            std::cout << "Sample " << (s + 1) << "/" << SAMPLES << " (n=" << n << ")" << std::endl;
        }
    }

    std::cout << "Benchmark complete. Results saved to experiments/results/runtime.csv" << std::endl;
    csv_file.close();

    return 0;
}
