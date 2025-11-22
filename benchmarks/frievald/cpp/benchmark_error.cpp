#include "algorithms.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cstring>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

const int TRIALS = 1000;
const int N = 500; // Fixed size for error benchmark
const vector<int> K_VALUES = {1, 2, 5, 10, 20};
const vector<string> ERROR_MODES = {"random_element"}; // Focus on one mode or multiple? Prompt says "incorrect C'".

int main(int argc, char* argv[]) {
    string out_csv = "results/frievald/error.csv";
    bool append_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--append") == 0) {
            append_mode = true;
        } else if (strcmp(argv[i], "--out-csv") == 0 && i + 1 < argc) {
            out_csv = argv[++i];
        }
    }

    // Ensure output directory exists
    fs::path out_path(out_csv);
    if (out_path.has_parent_path()) {
        fs::create_directories(out_path.parent_path());
    }

    mt19937 rng(42);
    ofstream csv_file;
    if (append_mode) {
        csv_file.open(out_csv, ios::app);
        cout << "Appending to " << out_csv << endl;
    } else {
        csv_file.open(out_csv);
        csv_file << "n,k,error_mode,false_accept_count,total_trials,false_accept_rate\n";
        cout << "Overwriting " << out_csv << endl;
    }

    cout << "Starting C++ Error Benchmark (Error vs k)..." << endl;

    // Generate fixed matrices A, B and correct C once
    Matrix A = Matrix::random(N, N, rng);
    Matrix B = Matrix::random(N, N, rng);
    // We need correct C to inject error. 
    // Since N=500, Strassen is fine.
    Matrix C_correct = matmul_strassen(A, B, 64);

    for (const auto& mode : ERROR_MODES) {
        cout << "Mode: " << mode << endl;
        Matrix C_err = inject_error(C_correct, mode, rng);
        
        for (int k : K_VALUES) {
            int false_accepts = 0;
            for (int t = 0; t < TRIALS; ++t) {
                // frievald_verify returns true if it thinks it's correct.
                // Since C_err is incorrect, true means false accept.
                bool result = frievald_verify(A, B, C_err, k, rng);
                if (result) {
                    false_accepts++;
                }
            }
            double rate = (double)false_accepts / TRIALS;
            csv_file << N << "," << k << "," << mode << "," << false_accepts << "," << TRIALS << "," << rate << "\n";
            cout << "  k=" << k << ": " << false_accepts << "/" << TRIALS << " false accepts (" << rate << ")" << endl;
        }
    }

    cout << "Error benchmark complete. Results saved to " << out_csv << endl;
    csv_file.close();
    return 0;
}
