#pragma once

/**
 * @file cardinality_estimators.hpp
 * @brief Header file for Cardinality Estimation algorithms (LogLog, HyperLogLog, PCSA).
 */

#include <cstdint>          // Used for fixed-width integer types (uint64_t, uint32_t).
#include <unordered_map>    // Used for hash maps (if needed in future extensions).
#include <vector>           // Used for dynamic arrays (registers).

namespace cardinality {

/**
 * @brief Deterministic 64-bit hash function (SplitMix64).
 * 
 * A fast, high-quality hash function used to map input values to 64-bit integers.
 * 
 * @param value The input integer.
 * @return uint64_t The hashed value.
 */
inline uint64_t hash64(uint64_t value) {
    uint64_t x = value + 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= (x >> 31);
    return x;
}

/**
 * @brief Computes the position of the first 1-bit (rank) from the left.
 * 
 * Used in LogLog and HyperLogLog to estimate cardinality based on bit patterns.
 * 
 * @param w The 64-bit integer to check.
 * @param max_bits The maximum number of bits to consider.
 * @return uint32_t The position of the first 1-bit (1-indexed).
 */
inline uint32_t rho(uint64_t w, uint32_t max_bits) {
    if (w == 0) {
        return max_bits + 1;
    }
    // __builtin_clzll returns the number of leading zeros in a 64-bit integer.
    uint32_t leading = 64 - static_cast<uint32_t>(__builtin_clzll(w));
    return max_bits - leading + 1;
}

/**
 * @brief Abstract base class for cardinality estimators.
 */
class CardinalityEstimator {
public:
    virtual ~CardinalityEstimator() = default;
    
    /**
     * @brief Adds a value to the estimator.
     * @param value The value to add.
     */
    virtual void add(uint64_t value) = 0;
    
    /**
     * @brief Estimates the cardinality of the stream seen so far.
     * @return double The estimated cardinality.
     */
    virtual double estimate() const = 0;
    
    /**
     * @brief Returns the memory usage of the estimator in bytes.
     * @return size_t Memory usage in bytes.
     */
    virtual size_t memory_usage_bytes() const = 0;
};

/**
 * @brief LogLog Cardinality Estimator.
 * 
 * Uses the LogLog algorithm (Durand-Flajolet, 2003).
 * Space complexity: O(m * log(log(N))).
 */
class LogLog : public CardinalityEstimator {
public:
    /**
     * @brief Constructs a LogLog estimator.
     * @param precision Number of bits to use for bucket indexing (m = 2^precision).
     */
    explicit LogLog(uint32_t precision = 14);
    void add(uint64_t value) override;
    double estimate() const override;
    size_t memory_usage_bytes() const override;

private:
    uint32_t precision_;
    uint64_t m_;
    uint64_t mask_;
    std::vector<uint8_t> registers_;
};

/**
 * @brief HyperLogLog Cardinality Estimator.
 * 
 * Uses the HyperLogLog algorithm (Flajolet et al., 2007).
 * Improves upon LogLog by using harmonic mean to reduce variance.
 */
class HyperLogLog : public CardinalityEstimator {
public:
    /**
     * @brief Constructs a HyperLogLog estimator.
     * @param precision Number of bits to use for bucket indexing (m = 2^precision).
     */
    explicit HyperLogLog(uint32_t precision = 14);
    void add(uint64_t value) override;
    double estimate() const override;
    size_t memory_usage_bytes() const override;

protected:
    double raw_estimate() const;
    double alpha() const;

    uint32_t precision_;
    uint64_t m_;
    uint64_t mask_;
    std::vector<uint8_t> registers_;
};

class HyperLogLogPlusPlus : public HyperLogLog {
public:
    explicit HyperLogLogPlusPlus(uint32_t precision = 14,
                                 double sparse_threshold_factor = 0.25);
    void add(uint64_t value) override;
    double estimate() const override;
    size_t memory_usage_bytes() const override;

private:
    void convert_to_dense();
    double bias(double raw_estimate) const;

    bool dense_;
    std::unordered_map<uint32_t, uint8_t> sparse_;
    size_t threshold_;
    double sparse_threshold_factor_;
};

}  // namespace cardinality
