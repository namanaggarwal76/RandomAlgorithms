#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <iomanip>

class Matrix {
public:
    size_t n_rows;
    size_t n_cols;
    std::vector<double> data;

    Matrix(size_t r, size_t c, double val = 0.0) : n_rows(r), n_cols(c), data(r * c, val) {}

    double& operator()(size_t r, size_t c) {
        return data[r * n_cols + c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data[r * n_cols + c];
    }

    size_t rows() const { return n_rows; }
    size_t cols() const { return n_cols; }

    static Matrix random(size_t r, size_t c, std::mt19937& rng, double low = -10.0, double high = 10.0) {
        std::uniform_real_distribution<double> dist(low, high);
        Matrix m(r, c);
        for (auto& val : m.data) {
            val = dist(rng);
        }
        return m;
    }
};

#endif // MATRIX_HPP
