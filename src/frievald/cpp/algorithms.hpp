#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include "matrix.hpp"

using namespace std;

// Standard O(N^3) multiplication
Matrix matmul_triple_loop(const Matrix& A, const Matrix& B);

// Strassen's Algorithm O(N^2.8)
Matrix matmul_strassen(const Matrix& A, const Matrix& B, int threshold = 64);

// Frievald's Verification O(k * N^2)
// if force_iterations is true, it continues for k iterations even if error is detected (for benchmarking)
bool frievald_verify(const Matrix& A, const Matrix& B, const Matrix& C, int k, mt19937& rng, bool force_iterations = false);

// Error injection
Matrix inject_error(const Matrix& C, const string& mode, mt19937& rng);

#endif // ALGORITHMS_HPP
