#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

/**
 * @file algorithms.hpp
 * @brief Header file for Matrix Multiplication and Verification algorithms.
 */

#include "matrix.hpp"

using namespace std;

/**
 * @brief Standard Matrix Multiplication using triple loops.
 * 
 * Computes C = A * B using the naive O(N^3) algorithm.
 * 
 * @param A First matrix.
 * @param B Second matrix.
 * @return Matrix The product matrix C.
 */
Matrix matmul_triple_loop(const Matrix& A, const Matrix& B);

/**
 * @brief Strassen's Matrix Multiplication Algorithm.
 * 
 * Computes C = A * B using Strassen's divide and conquer approach.
 * Complexity: O(N^log2(7)) approx O(N^2.81).
 * 
 * @param A First matrix.
 * @param B Second matrix.
 * @param threshold Size below which standard multiplication is used (default 64).
 * @return Matrix The product matrix C.
 */
Matrix matmul_strassen(const Matrix& A, const Matrix& B, int threshold = 64);

/**
 * @brief Frievald's Randomized Verification Algorithm.
 * 
 * Verifies if A * B = C with high probability in O(k * N^2) time.
 * 
 * @param A First matrix.
 * @param B Second matrix.
 * @param C Product matrix to verify.
 * @param k Number of iterations (random vectors to test).
 * @param rng Random number generator.
 * @param force_iterations If true, runs all k iterations even if error is found (for benchmarking).
 * @return true If A * B == C (with probability >= 1 - 2^-k).
 * @return false If A * B != C (always correct).
 */
bool frievald_verify(const Matrix& A, const Matrix& B, const Matrix& C, int k, mt19937& rng, bool force_iterations = false);

/**
 * @brief Injects errors into a matrix for testing verification algorithms.
 * 
 * @param C The original matrix.
 * @param mode Type of error ("single", "row", "random").
 * @param rng Random number generator.
 * @return Matrix The modified matrix with injected errors.
 */
Matrix inject_error(const Matrix& C, const string& mode, mt19937& rng);

#endif // ALGORITHMS_HPP
