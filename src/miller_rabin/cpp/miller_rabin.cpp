/**
 * @file miller_rabin.cpp
 * @brief Implementation of the Miller-Rabin primality test in C++.
 */

#include <iostream>     // Used for standard I/O operations.
#include <vector>       // Used for storing lists of numbers (if needed).
#include <string>       // Used for string manipulation.
#include <random>       // Used for random number generation (std::mt19937_64).
#include <fstream>      // Used for file I/O.
#include <chrono>       // Used for high-resolution timing.
#include <iomanip>      // Used for formatting output (std::setprecision).
#include <algorithm>    // Used for standard algorithms.

using namespace std;

// Use __int128 for safe modular multiplication of 64-bit numbers
// This prevents overflow when multiplying two 64-bit integers before modulo.
typedef __int128_t int128;

/**
 * @brief Performs modular multiplication (a * b) % n safely.
 * 
 * Uses 128-bit integers to prevent overflow during the multiplication of two 64-bit numbers.
 * 
 * @param a First operand.
 * @param b Second operand.
 * @param n Modulus.
 * @return uint64_t Result of (a * b) % n.
 */
uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t n) {
    return (uint64_t)((int128)a * b % n);
}

/**
 * @brief Performs modular exponentiation (base^exp) % n.
 * 
 * Uses the method of repeated squaring (binary exponentiation) for O(log exp) complexity.
 * 
 * @param base The base.
 * @param exp The exponent.
 * @param n The modulus.
 * @return uint64_t Result of (base^exp) % n.
 */
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

/**
 * @brief Miller-Rabin Primality Test.
 * 
 * Probabilistically checks if a number n is prime.
 * 
 * @param n The number to test.
 * @param k The number of iterations (witnesses) to check.
 * @param rng Random number generator engine.
 * @return true If n is likely prime.
 * @return false If n is definitely composite.
 */
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



/**
 * @brief Counts witnesses and liars for a composite number n.
 * 
 * Iterates through all possible bases a in [2, n-2].
 * A "witness" is a base that reveals n is composite.
 * A "liar" is a base that falsely suggests n might be prime.
 * 
 * @param n The composite number to analyze.
 * @return pair<uint64_t, uint64_t> Pair of {witnesses, liars}.
 */
pair<uint64_t, uint64_t> count_witnesses_all(uint64_t n) {
    if (n < 4) return {0, 0}; // 2 and 3 are prime, loop won't run

    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }

    uint64_t witnesses = 0;
    uint64_t liars = 0;

    for (uint64_t a = 2; a <= n - 2; ++a) {
        uint64_t x = power(a, d, n);
        if (x == 1 || x == n - 1) {
            liars++;
            continue;
        }

        bool composite = true;
        for (int j = 0; j < r - 1; ++j) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        
        if (composite) witnesses++;
        else liars++;
    }
    return {witnesses, liars};
}

int main(int argc, char* argv[]) {
    string input_file;
    string out_csv;
    string mode = "benchmark"; // benchmark, analysis
    int k = 5;
    int seed = 42;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--input-file" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--out-csv" && i + 1 < argc) out_csv = argv[++i];
        else if (arg == "--k" && i + 1 < argc) k = stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = stoi(argv[++i]);
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    }

    if (input_file.empty() || out_csv.empty()) {
        cerr << "Usage: " << argv[0] << " --input-file <file> --out-csv <file> [--mode <benchmark|analysis>] [--k <rounds>] [--seed <seed>]" << endl;
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
    
    if (mode == "benchmark") {
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

            outfile << n << "," << k << "," << is_prime << "," << duration << "\n";
        }
    } else if (mode == "analysis") {
        if (!file_exists) {
            outfile << "n,witnesses,liars,total_bases\n";
        }
        
        uint64_t n;
        while (infile >> n) {
            pair<uint64_t, uint64_t> counts = count_witnesses_all(n);
            outfile << n << "," << counts.first << "," << counts.second << "," << (counts.first + counts.second) << "\n";
        }
    } else {
        cerr << "Unknown mode: " << mode << endl;
        return 1;
    }

    return 0;
}
