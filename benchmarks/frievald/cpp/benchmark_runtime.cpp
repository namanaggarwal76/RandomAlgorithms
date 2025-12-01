#include "algorithms.hpp" // Include the Frievald algorithm header
#include <iostream> // Used for standard input/output
#include <fstream> // Used for file input/output
#include <vector> // Used for dynamic arrays
#include <chrono> // Used for timing measurements
#include <random> // Used for random number generation
#include <string> // Used for string manipulation
#include <iomanip> // Used for output formatting
#include <cstring> // Used for C-style string manipulation
#include <functional> // Used for function objects
#include <filesystem> // Used for filesystem operations

using namespace std;
namespace fs = std::filesystem;

// Defaults
const int SAMPLES = 10; // Reduced samples per file as we iterate over files
const int SKIP_NAIVE_ABOVE = 500; // Skip naive O(N^3) for large N
const vector<int> FRIEVALD_KS = {5}; // Fixed k for scaling experiment

/**
 * @brief Measures the execution time of a function.
 * 
 * @param func The function to measure.
 * @return double Execution time in seconds.
 */
double measure_time(function<void()> func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    return diff.count();
}

/**
 * @brief Reads a matrix from a file.
 * 
 * @param file Input file stream.
 * @param n Dimension of the matrix.
 * @return Matrix The read matrix.
 */
Matrix read_matrix(ifstream& file, int n) {
    Matrix M(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> M(i, j);
        }
    }
    return M;
}

void process_file(const string& filepath, ofstream& csv_file, mt19937& rng) {
    ifstream infile(filepath);
    if (!infile) {
        cerr << "Error opening " << filepath << endl;
        return;
    }

    int n;
    infile >> n;

    Matrix A = read_matrix(infile, n);
    Matrix B = read_matrix(infile, n);
    Matrix C_correct(n, n);

    // 1. Benchmark Naive (Triple Loop) - Baseline
    if (n <= SKIP_NAIVE_ABOVE) {
        double t_naive = measure_time([&]() {
            matmul_triple_loop(A, B);
        });
        // Log naive
        csv_file << filepath << "," << n << ",naive," << fixed << setprecision(9) << t_naive << ",0\n";
    }

    // 2. Benchmark Frievald (Fixed k)
    // We need a C to verify. For benchmarking runtime, we can use a dummy C or the correct one.
    // Using correct one is safer to avoid early exit if we had logic for that (though verify runs full k usually).
    // Let's compute C using Strassen if needed, or just use a zero matrix if we force iterations.
    // frievald_verify has a 'force_iterations' flag.
    Matrix C_dummy(n, n); // Zero matrix, likely incorrect, but we force iterations.
    
    for (int k : FRIEVALD_KS) {
        // Run multiple times to get average/stable reading
        for(int s=0; s<SAMPLES; ++s) {
             double t_frievald = measure_time([&]() {
                frievald_verify(A, B, C_dummy, k, rng, true); // force_iterations = true
            });
            csv_file << filepath << "," << n << ",frievald," << t_frievald << "," << k << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    string dataset_dir = "datasets/frievald";
    string out_csv = "results/frievald/runtime.csv";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dataset-dir") == 0 && i+1 < argc) {
            dataset_dir = argv[++i];
        } else if (strcmp(argv[i], "--out-csv") == 0 && i+1 < argc) {
            out_csv = argv[++i];
        }
    }

    // Ensure output directory exists
    fs::path out_path(out_csv);
    if (out_path.has_parent_path()) {
        fs::create_directories(out_path.parent_path());
    }

    mt19937 rng(90);
    ofstream csv_file(out_csv);
    csv_file << "file,n,algorithm,seconds,k\n";

    cout << "Starting Frievald Benchmark..." << endl;

    for (const auto& entry : fs::directory_iterator(dataset_dir)) {
        if (entry.path().extension() == ".txt") {
            cout << "Processing " << entry.path() << "..." << endl;
            process_file(entry.path().string(), csv_file, rng);
        }
    }

    cout << "Benchmark complete. Results saved to " << out_csv << endl;
    csv_file.close();

    return 0;
}
