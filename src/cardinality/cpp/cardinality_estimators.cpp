/**
 * @file cardinality_estimators.cpp
 * @brief Implementation of Cardinality Estimation algorithms.
 */

#include "cardinality_estimators.hpp"

#include <cmath>    // Used for mathematical functions (pow, log).
#include <limits>   // Used for numeric limits (if needed).

namespace cardinality {

/**
 * @brief Constructs a LogLog estimator.
 * @param precision Number of bits to use for bucket indexing.
 */
LogLog::LogLog(uint32_t precision)
    : precision_(precision),
      m_(1ull << precision),
      mask_((1ull << (64 - precision)) - 1),
      registers_(m_, 0) {}

/**
 * @brief Adds a value to the LogLog estimator.
 * 
 * Hashes the value, determines the bucket index, and updates the register
 * if the new rank is greater than the current rank.
 * 
 * @param value The value to add.
 */
void LogLog::add(uint64_t value) {
    uint64_t h = hash64(value);
    uint32_t idx = static_cast<uint32_t>(h >> (64 - precision_));
    uint64_t w = h & mask_;
    uint32_t rank = rho(w, 64 - precision_);
    if (rank > registers_[idx]) {
        registers_[idx] = static_cast<uint8_t>(rank);
    }
}

/**
 * @brief Estimates the cardinality using the LogLog formula.
 * 
 * E = alpha * m * 2^(average_rank)
 * Includes small range correction.
 * 
 * @return double The estimated cardinality.
 */
double LogLog::estimate() const {
    double sum = 0.0;
    for (auto r : registers_) {
        sum += r;
    }
    double avg_rank = sum / static_cast<double>(m_);
    constexpr double alpha = 0.39701; // Constant for LogLog
    double est = alpha * static_cast<double>(m_) * std::pow(2.0, avg_rank);
    
    // Small range correction
    size_t zeros = 0;
    for (auto r : registers_) {
        if (r == 0) zeros++;
    }
    if (zeros && est < 2.5 * m_) {
        return m_ * std::log(static_cast<double>(m_) / zeros);
    }
    return est;
}

size_t LogLog::memory_usage_bytes() const {
    return registers_.size();
}

/**
 * @brief Constructs a HyperLogLog estimator.
 * @param precision Number of bits to use for bucket indexing.
 */
HyperLogLog::HyperLogLog(uint32_t precision)
    : precision_(precision),
      m_(1ull << precision),
      mask_((1ull << (64 - precision)) - 1),
      registers_(m_, 0) {}

/**
 * @brief Adds a value to the HyperLogLog estimator.
 * @param value The value to add.
 */
void HyperLogLog::add(uint64_t value) {
    uint64_t h = hash64(value);
    uint32_t idx = static_cast<uint32_t>(h >> (64 - precision_));
    uint64_t w = h & mask_;
    uint32_t rank = rho(w, 64 - precision_);
    if (rank > registers_[idx]) {
        registers_[idx] = static_cast<uint8_t>(rank);
    }
}

/**
 * @brief Computes the alpha constant for HyperLogLog based on m.
 * @return double The alpha constant.
 */
double HyperLogLog::alpha() const {
    switch (m_) {
        case 16: return 0.673;
        case 32: return 0.697;
        case 64: return 0.709;
        default:
            return 0.7213 / (1.0 + 1.079 / static_cast<double>(m_));
    }
}

/**
 * @brief Computes the raw HyperLogLog estimate (harmonic mean).
 * @return double The raw estimate.
 */
double HyperLogLog::raw_estimate() const {
    double denominator = 0.0;
    for (auto r : registers_) {
        denominator += std::pow(2.0, -static_cast<int>(r));
    }
    return alpha() * (static_cast<double>(m_) * static_cast<double>(m_)) / denominator;
}

/**
 * @brief Estimates the cardinality using the HyperLogLog formula.
 * 
 * Includes corrections for small and large ranges.
 * 
 * @return double The estimated cardinality.
 */
double HyperLogLog::estimate() const {
    double raw = raw_estimate();
    size_t zeros = 0;
    for (auto r : registers_) {
        if (r == 0) zeros++;
    }
    // Small range correction
    if (raw <= 5 * static_cast<double>(m_) / 2.0 && zeros) {
        return m_ * std::log(static_cast<double>(m_) / zeros);
    }
    // Large range correction (for 32-bit hashes, though we use 64-bit here, logic remains similar)
    if (raw > (static_cast<double>(1ull << 32)) / 30.0) {
        return -(1ull << 32) * std::log(1.0 - raw / (1ull << 32));
    }
    return raw;
}

size_t HyperLogLog::memory_usage_bytes() const {
    return registers_.size();
}

HyperLogLogPlusPlus::HyperLogLogPlusPlus(uint32_t precision,
                                         double sparse_threshold_factor)
    : HyperLogLog(precision),
      dense_(false),
      threshold_(static_cast<size_t>(sparse_threshold_factor * m_)),
      sparse_threshold_factor_(sparse_threshold_factor) {}

void HyperLogLogPlusPlus::add(uint64_t value) {
    uint64_t h = hash64(value);
    uint32_t idx = static_cast<uint32_t>(h >> (64 - precision_));
    uint64_t w = h & mask_;
    uint32_t rank = rho(w, 64 - precision_);

    if (!dense_) {
        auto& ref = sparse_[idx];
        if (rank > ref) {
            ref = static_cast<uint8_t>(rank);
        }
        if (sparse_.size() >= threshold_) {
            convert_to_dense();
        }
    } else {
        if (rank > registers_[idx]) {
            registers_[idx] = static_cast<uint8_t>(rank);
        }
    }
}

void HyperLogLogPlusPlus::convert_to_dense() {
    for (const auto& entry : sparse_) {
        uint32_t idx = entry.first;
        uint8_t rank = entry.second;
        if (rank > registers_[idx]) {
            registers_[idx] = rank;
        }
    }
    sparse_.clear();
    dense_ = true;
}

double HyperLogLogPlusPlus::bias(double raw_estimate) const {
    static const std::pair<double, double> table[] = {
        {1000.0, 60.0},
        {2000.0, 100.0},
        {5000.0, 200.0},
        {10000.0, 320.0},
        {20000.0, 430.0},
        {40000.0, 520.0},
        {80000.0, 600.0},
    };
    if (raw_estimate <= table[0].first) {
        return table[0].second;
    }
    for (size_t i = 0; i + 1 < sizeof(table)/sizeof(table[0]); ++i) {
        auto [x0, y0] = table[i];
        auto [x1, y1] = table[i + 1];
        if (raw_estimate <= x1) {
            double ratio = (raw_estimate - x0) / (x1 - x0);
            return y0 + ratio * (y1 - y0);
        }
    }
    return table[sizeof(table)/sizeof(table[0]) - 1].second;
}

double HyperLogLogPlusPlus::estimate() const {
    if (!dense_) {
        double v = static_cast<double>(m_ - sparse_.size());
        if (v <= 0) {
            const_cast<HyperLogLogPlusPlus*>(this)->convert_to_dense();
        } else {
            return m_ * std::log(static_cast<double>(m_) / v);
        }
    }

    double raw = raw_estimate();
    double estimator = raw;
    if (raw <= 5.0 * static_cast<double>(m_)) {
        estimator = raw - bias(raw);
    }

    size_t zeros = 0;
    for (auto r : registers_) {
        if (r == 0) zeros++;
    }
    if (estimator <= 5.0 * static_cast<double>(m_) && zeros) {
        estimator = m_ * std::log(static_cast<double>(m_) / zeros);
    }
    if (estimator > (static_cast<double>(1ull << 32)) / 30.0) {
        estimator = -(1ull << 32) * std::log(1.0 - estimator / (1ull << 32));
    }
    return estimator;
}

size_t HyperLogLogPlusPlus::memory_usage_bytes() const {
    if (!dense_) {
        size_t bits = sparse_.size() * (precision_ + 6);
        return bits / 8 + 64;
    }
    return registers_.size();
}

}  // namespace cardinality
