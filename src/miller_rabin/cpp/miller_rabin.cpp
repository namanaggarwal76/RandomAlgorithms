#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;

// Use __int128 for safe modular multiplication of 64-bit numbers
typedef __int128_t int128;

// Modular multiplication: (a * b) % n
uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t n) {
    return (uint64_t)((int128)a * b % n);
}

// Modular exponentiation: (base^exp) % n
uint64_t power(uint64_t base, uint64_t exp, uint64_t n) {
    uint64_t res = 1;
    base %= n;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul_mod(res, base, n);
        base = mul_mod(base, base, n);
        exp /= 2;
    }
    return res;
}

// Miller-Rabin Primality Test
// Returns true if n is probably prime, false if composite
bool miller_rabin(uint64_t n, int k, mt19937_64& rng) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 as 2^r * d
    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }

    uniform_int_distribution<uint64_t> dist(2, n - 2);

    for (int i = 0; i < k; ++i) {
        uint64_t a = dist(rng);
        uint64_t x = power(a, d, n);

        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (int j = 0; j < r - 1; ++j) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    string input_file;
    string out_csv;
    int k = 5;
    int seed = 42;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--input-file" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--out-csv" && i + 1 < argc) out_csv = argv[++i];
        else if (arg == "--k" && i + 1 < argc) k = stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = stoi(argv[++i]);
    }

    if (input_file.empty() || out_csv.empty()) {
        cerr << "Usage: " << argv[0] << " --input-file <file> --out-csv <file> [--k <rounds>] [--seed <seed>]" << endl;
        return 1;
    }

    ifstream infile(input_file);
    if (!infile) {
        cerr << "Error opening input file: " << input_file << endl;
        return 1;
    }

    // Check if CSV exists to write header
    bool file_exists = ifstream(out_csv).good();
    ofstream outfile(out_csv, ios::app);
    if (!file_exists) {
        outfile << "n,k,is_probable_prime,time_ns\n";
    }

    mt19937_64 rng(seed);
    uint64_t n;
    while (infile >> n) {
        auto start = chrono::high_resolution_clock::now();
        bool is_prime = miller_rabin(n, k, rng);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

        outfile << n << "," << k << "," << (is_prime ? 1 : 0) << "," << duration << "\n";
    }

    return 0;
}
