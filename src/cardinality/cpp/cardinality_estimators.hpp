#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cardinality {

uint64_t hash64(uint64_t value, uint64_t seed);
uint32_t leading_zeroes(uint64_t value);

class CardinalityEstimator {
public:
    explicit CardinalityEstimator(uint64_t seed = 0) : seed_(seed) {}
    virtual ~CardinalityEstimator() = default;

    virtual void add(uint64_t value) = 0;
    virtual double estimate() const = 0;
    virtual size_t memory_usage_bytes() const = 0;

protected:
    uint64_t seed_;
};

class LogLog : public CardinalityEstimator {
public:
    explicit LogLog(uint32_t precision = 14, uint64_t seed = 0);

    void add(uint64_t value) override;
    double estimate() const override;
    size_t memory_usage_bytes() const override;

private:
    uint32_t precision_;
    uint64_t m_;
    std::vector<uint8_t> registers_;
};

class HyperLogLog : public CardinalityEstimator {
public:
    explicit HyperLogLog(uint32_t precision = 14, uint64_t seed = 0, bool allocate_registers = true);

    void add(uint64_t value) override;
    double estimate() const override;
    size_t memory_usage_bytes() const override;

protected:
    double raw_estimate() const;
    double alpha() const;
    uint8_t compute_rank(uint64_t hash) const;
    double bias(double raw_estimate) const;

    uint32_t precision_;
    uint64_t m_;
    std::vector<uint8_t> registers_;
};

class HyperLogLogPlusPlus : public HyperLogLog {
public:
    explicit HyperLogLogPlusPlus(uint32_t precision = 14,
                                 double sparse_threshold_factor = 0.25,
                                 uint64_t seed = 0);

    void add(uint64_t value) override;
    double estimate() const override;
    size_t memory_usage_bytes() const override;

private:
    void convert_to_dense();
    double bias(double raw_estimate) const;

    bool dense_;
    std::vector<uint64_t> sparse_list_;
    size_t threshold_;
};

}  // namespace cardinality
