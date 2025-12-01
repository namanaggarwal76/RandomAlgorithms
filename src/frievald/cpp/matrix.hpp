#ifndef MATRIX_HPP
#define MATRIX_HPP

/**
 * @file matrix.hpp
 * @brief Header file for Matrix class definition.
 */

#include <vector>       // Used for storing matrix data in a contiguous block.
#include <random>       // Used for generating random matrices.
#include <iostream>     // Used for I/O operations.
#include <stdexcept>    // Used for exception handling.
#include <cmath>        // Used for mathematical operations.
#include <iomanip>      // Used for output formatting.

using namespace std;

/**
 * @brief A simple dense Matrix class.
 * 
 * Stores matrix data in a 1D vector for cache efficiency.
 */
class Matrix {
public:
    size_t n_rows; /**< Number of rows */
    size_t n_cols; /**< Number of columns */
    vector<double> data; /**< Flattened data storage */

    /**
     * @brief Constructs a matrix of size r x c initialized with val.
     * @param r Number of rows.
     * @param c Number of columns.
     * @param val Initial value for all elements.
     */
    Matrix(size_t r, size_t c, double val = 0.0) : n_rows(r), n_cols(c), data(r * c, val) {}

    /**
     * @brief Access element at (r, c).
     * @param r Row index.
     * @param c Column index.
     * @return double& Reference to the element.
     */
    double& operator()(size_t r, size_t c) {
        return data[r * n_cols + c];
    }

    /**
     * @brief Access element at (r, c) (const version).
     * @param r Row index.
     * @param c Column index.
     * @return const double& Const reference to the element.
     */
    const double& operator()(size_t r, size_t c) const {
        return data[r * n_cols + c];
    }

    size_t rows() const { return n_rows; }
    size_t cols() const { return n_cols; }

    /**
     * @brief Generates a random matrix.
     * @param r Number of rows.
     * @param c Number of columns.
     * @param rng Random number generator.
     * @param low Lower bound for random values.
     * @param high Upper bound for random values.
     * @return Matrix A matrix filled with random values.
     */
    static Matrix random(size_t r, size_t c, mt19937& rng, double low = -10.0, double high = 10.0) {
        uniform_real_distribution<double> dist(low, high);
        Matrix m(r, c);
        for (auto& val : m.data) {
            val = dist(rng);
        }
        return m;
    }
};

#endif // MATRIX_HPP
