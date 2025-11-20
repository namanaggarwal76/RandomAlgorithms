#include "../src/algorithms.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cstring>

const int TRIALS = 1000;
const int N = 200; // Fixed size for error benchmark
const int MAX_ITERS = 50;
const std::vector<std::string> ERROR_MODES = {"random_element", "random_row", "random_col"};

int main(int argc, char* argv[]) {
    bool append_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--append") == 0) {
            append_mode = true;
        }
    }

    std::mt19937 rng(42);
    std::ofstream csv_file;
    if (append_mode) {
        csv_file.open("experiments/results/cpp_error_raw.csv", std::ios::app);
        std::cout << "Appending to experiments/results/cpp_error_raw.csv" << std::endl;
    } else {
        csv_file.open("experiments/results/cpp_error_raw.csv");
        csv_file << "trial,n,error_mode,iterations_to_detect\n";
        std::cout << "Overwriting experiments/results/cpp_error_raw.csv" << std::endl;
    }

    std::cout << "Starting C++ Error Benchmark (Detection Curve)..." << std::endl;

    for (const auto& mode : ERROR_MODES) {
        std::cout << "Mode: " << mode << std::endl;
        for (int t = 0; t < TRIALS; ++t) {
            Matrix A = Matrix::random(N, N, rng);
            Matrix B = Matrix::random(N, N, rng);
            Matrix C = matmul_strassen(A, B, 64); // Correct C
            
            Matrix C_err = inject_error(C, mode, rng);

            int detected_at = -1;
            for (int k = 1; k <= MAX_ITERS; ++k) {
                // Run 1 iteration of Frievald
                // We can reuse the frievald_verify function if we pass k=1, 
                // but we need to know if it returned false (detected error).
                // frievald_verify returns true if verified (no error detected), false if error detected.
                bool verified = frievald_verify(A, B, C_err, 1, rng);
                if (!verified) {
                    detected_at = k;
                    break;
                }
            }
            
            // If never detected (should be rare for valid error modes), detected_at remains -1
            csv_file << t << "," << N << "," << mode << "," << detected_at << "\n";
        }
    }

    std::cout << "Error benchmark complete. Results saved to experiments/results/cpp_error_raw.csv" << std::endl;
    csv_file.close();
    return 0;
}
