#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace cardinality {

// Deterministic 64-bit hash (SplitMix64)
inline uint64_t hash64(uint64_t value) {
    uint64_t x = value + 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= (x >> 31);
    return x;
}

inline uint32_t rho(uint64_t w, uint32_t max_bits) {
    if (w == 0) {
        return max_bits + 1;
    }
    uint32_t leading = 64 - static_cast<uint32_t>(__builtin_clzll(w));
    return max_bits - leading + 1;
}

class CardinalityEstimator {
public:
    virtual ~CardinalityEstimator() = default;
    virtual void add(uint64_t value) = 0;
    virtual double estimate() const = 0;
    virtual size_t memory_usage_bytes() const = 0;
};

class LogLog : public CardinalityEstimator {
public:
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

class HyperLogLog : public CardinalityEstimator {
public:
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
