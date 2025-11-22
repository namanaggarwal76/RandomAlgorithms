#include "algorithms.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

// --- Helper Functions ---

// Add two matrices
Matrix add(const Matrix& A, const Matrix& B) {
    size_t n = A.rows();
    size_t m = A.cols();
    Matrix C(n, m);
    for (size_t i = 0; i < n * m; ++i) {
        C.data[i] = A.data[i] + B.data[i];
    }
    return C;
}

// Subtract two matrices
Matrix sub(const Matrix& A, const Matrix& B) {
    size_t n = A.rows();
    size_t m = A.cols();
    Matrix C(n, m);
    for (size_t i = 0; i < n * m; ++i) {
        C.data[i] = A.data[i] - B.data[i];
    }
    return C;
}

// Get next power of 2
size_t next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Pad matrix to size new_n x new_n
Matrix pad_matrix(const Matrix& A, size_t new_n) {
    size_t n = A.rows();
    Matrix P(new_n, new_n); // Initialized to 0
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            P(i, j) = A(i, j);
        }
    }
    return P;
}

// Unpad matrix to original size
Matrix unpad_matrix(const Matrix& P, size_t n) {
    Matrix A(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = P(i, j);
        }
    }
    return A;
}

// --- Triple Loop ---

Matrix matmul_triple_loop(const Matrix& A, const Matrix& B) {
    size_t n = A.rows();
    size_t m = A.cols();
    size_t p = B.cols();
    
    // Assuming A is n x m and B is m x p
    // Result is n x p
    Matrix C(n, p);

    // Simple cache-friendly optimization: transpose B? 
    // For strict "triple loop" definition we usually just do ijk.
    // But to be "considerably faster" than Python, even naive C++ is enough.
    // Let's stick to standard ijk for correctness and simplicity first.
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < m; ++k) {
            double r = A(i, k);
            for (size_t j = 0; j < p; ++j) {
                C(i, j) += r * B(k, j);
            }
        }
    }
    return C;
}

// --- Strassen ---

Matrix strassen_recursive(const Matrix& A, const Matrix& B, int threshold) {
    size_t n = A.rows();

    if (n <= (size_t)threshold) {
        return matmul_triple_loop(A, B);
    }

    size_t k = n / 2;

    // Submatrices
    Matrix A11(k, k), A12(k, k), A21(k, k), A22(k, k);
    Matrix B11(k, k), B12(k, k), B21(k, k), B22(k, k);

    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            A11(i, j) = A(i, j);
            A12(i, j) = A(i, j + k);
            A21(i, j) = A(i + k, j);
            A22(i, j) = A(i + k, j + k);

            B11(i, j) = B(i, j);
            B12(i, j) = B(i, j + k);
            B21(i, j) = B(i + k, j);
            B22(i, j) = B(i + k, j + k);
        }
    }

    // 7 Products
    // M1 = (A11 + A22)(B11 + B22)
    Matrix M1 = strassen_recursive(add(A11, A22), add(B11, B22), threshold);
    // M2 = (A21 + A22)B11
    Matrix M2 = strassen_recursive(add(A21, A22), B11, threshold);
    // M3 = A11(B12 - B22)
    Matrix M3 = strassen_recursive(A11, sub(B12, B22), threshold);
    // M4 = A22(B21 - B11)
    Matrix M4 = strassen_recursive(A22, sub(B21, B11), threshold);
    // M5 = (A11 + A12)B22
    Matrix M5 = strassen_recursive(add(A11, A12), B22, threshold);
    // M6 = (A21 - A11)(B11 + B12)
    Matrix M6 = strassen_recursive(sub(A21, A11), add(B11, B12), threshold);
    // M7 = (A12 - A22)(B21 + B22)
    Matrix M7 = strassen_recursive(sub(A12, A22), add(B21, B22), threshold);

    // Reconstruct C
    // C11 = M1 + M4 - M5 + M7
    Matrix C11 = add(sub(add(M1, M4), M5), M7);
    // C12 = M3 + M5
    Matrix C12 = add(M3, M5);
    // C21 = M2 + M4
    Matrix C21 = add(M2, M4);
    // C22 = M1 - M2 + M3 + M6
    // Note: The implementation below uses add/sub helpers which create new Matrix objects by value.
    // This is inefficient for large matrices due to allocation overhead.
    // For production code, use in-place operations or a library like BLAS/Eigen.
    Matrix C22 = add(add(sub(M1, M2), M3), M6);

    Matrix C(n, n);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            C(i, j) = C11(i, j);
            C(i, j + k) = C12(i, j);
            C(i + k, j) = C21(i, j);
            C(i + k, j + k) = C22(i, j);
        }
    }

    return C;
}

Matrix matmul_strassen(const Matrix& A, const Matrix& B, int threshold) {
    size_t n = A.rows();
    // Check if square and power of 2
    size_t m = next_power_of_2(n);
    
    if (m == n) {
        return strassen_recursive(A, B, threshold);
    } else {
        Matrix Ap = pad_matrix(A, m);
        Matrix Bp = pad_matrix(B, m);
        Matrix Cp = strassen_recursive(Ap, Bp, threshold);
        return unpad_matrix(Cp, n);
    }
}

// --- Frievald ---

bool frievald_verify(const Matrix& A, const Matrix& B, const Matrix& C, int k, mt19937& rng, bool force_iterations) {
    size_t n = A.rows();
    uniform_int_distribution<int> dist(0, 1); // Random 0 or 1
    bool all_correct = true;

    for (int iter = 0; iter < k; ++iter) {
        // Generate random vector r
        Matrix r(n, 1);
        for (size_t i = 0; i < n; ++i) {
            r(i, 0) = dist(rng);
        }

        // Compute Br = B * r
        Matrix Br = matmul_triple_loop(B, r);
        // Compute ABr = A * (Br)
        Matrix ABr = matmul_triple_loop(A, Br);
        // Compute Cr = C * r
        Matrix Cr = matmul_triple_loop(C, r);

        // Check if ABr == Cr
        for (size_t i = 0; i < n; ++i) {
            if (abs(ABr(i, 0) - Cr(i, 0)) > 1e-9) {
                if (!force_iterations) {
                    return false;
                }
                all_correct = false;
            }
        }
    }
    return all_correct;
}

// --- Error Injection ---

Matrix inject_error(const Matrix& C, const string& mode, mt19937& rng) {
    Matrix C_err = C;
    size_t n = C.rows();
    
    if (mode == "none") {
        return C_err;
    } else if (mode == "random_element") {
        uniform_int_distribution<size_t> dist_idx(0, n - 1);
        size_t r = dist_idx(rng);
        size_t c = dist_idx(rng);
        C_err(r, c) += 10.0; // Significant error
    } else if (mode == "random_row") {
        uniform_int_distribution<size_t> dist_idx(0, n - 1);
        size_t r = dist_idx(rng);
        for (size_t j = 0; j < n; ++j) {
            C_err(r, j) += 5.0;
        }
    } else if (mode == "random_col") {
        uniform_int_distribution<size_t> dist_idx(0, n - 1);
        size_t c = dist_idx(rng);
        for (size_t i = 0; i < n; ++i) {
            C_err(i, c) += 5.0;
        }
    }
    return C_err;
}
