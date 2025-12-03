#include "cardinality_estimators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace cardinality {
namespace {

constexpr uint32_t kMinPrecision = 4;
constexpr uint32_t kMaxPrecision = 18;

uint32_t normalize_precision(uint32_t precision) {
    if (precision < kMinPrecision || precision > kMaxPrecision) {
        throw std::invalid_argument("precision must be in the range [4, 18]");
    }
    return precision;
}

uint8_t rank_from_hash(uint64_t hash, uint32_t precision) {
    const uint64_t shifted = hash << precision;
    uint32_t rank = leading_zeroes(shifted) + 1;
    const uint32_t max_rank = 64u - precision + 1u;
    if (rank > max_rank) {
        rank = max_rank;
    }
    return static_cast<uint8_t>(rank);
}

}  // namespace

uint64_t hash64(uint64_t value, uint64_t seed) {
    uint64_t x = value + seed + 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= (x >> 31);
    return x;
}

uint32_t leading_zeroes(uint64_t value) {
    if (value == 0) {
        return 64;
    }
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_clzll(value));
#elif defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanReverse64(&index, value);
    return 63u - index;
#else
    uint32_t count = 0;
    uint64_t mask = 1ull << 63;
    while ((value & mask) == 0) {
        ++count;
        mask >>= 1;
    }
    return count;
#endif
}

// ------------------------------ LogLog --------------------------------------

LogLog::LogLog(uint32_t precision, uint64_t seed)
    : CardinalityEstimator(seed),
      precision_(normalize_precision(precision)),
      m_(1ull << precision_),
      registers_(m_, 0) {}

void LogLog::add(uint64_t value) {
    const uint64_t hashed = hash64(value, seed_);
    const uint32_t idx = static_cast<uint32_t>(hashed >> (64 - precision_));
    const uint8_t rank = rank_from_hash(hashed, precision_);
    if (rank > registers_[idx]) {
        registers_[idx] = rank;
    }
}

double LogLog::estimate() const {
    double sum = 0.0;
    for (uint8_t reg : registers_) {
        sum += reg;
    }
    const double avg_rank = sum / static_cast<double>(m_);
    constexpr double alpha = 0.39701;
    const double raw = alpha * static_cast<double>(m_) * std::pow(2.0, avg_rank);

    size_t zeros = 0;
    for (uint8_t reg : registers_) {
        if (reg == 0) {
            ++zeros;
        }
    }
    if (zeros > 0 && raw < 2.5 * static_cast<double>(m_)) {
        return m_ * std::log(static_cast<double>(m_) / static_cast<double>(zeros));
    }
    return raw;
}

size_t LogLog::memory_usage_bytes() const {
    return registers_.size() * sizeof(uint8_t);
}

// --------------------------- HyperLogLog ------------------------------------

HyperLogLog::HyperLogLog(uint32_t precision, uint64_t seed, bool allocate_registers)
    : CardinalityEstimator(seed),
      precision_(normalize_precision(precision)),
      m_(1ull << precision_) {
    if (allocate_registers) {
        registers_.assign(m_, 0);
    }
}

void HyperLogLog::add(uint64_t value) {
    if (registers_.empty()) {
        registers_.assign(m_, 0);
    }
    const uint64_t hashed = hash64(value, seed_);
    const uint32_t idx = static_cast<uint32_t>(hashed >> (64 - precision_));
    const uint8_t rank = compute_rank(hashed);
    if (rank > registers_[idx]) {
        registers_[idx] = rank;
    }
}

double HyperLogLog::alpha() const {
    switch (m_) {
        case 16:
            return 0.673;
        case 32:
            return 0.697;
        case 64:
            return 0.709;
        default:
            return 0.7213 / (1.0 + 1.079 / static_cast<double>(m_));
    }
}

double HyperLogLog::bias(double raw_estimate) const {
    if (precision_ != 14) {
        return 0.0;
    }
    static const std::array<std::pair<double, double>, 12> kBiasTable = {{
        {1000.0, 60.0},  {2000.0, 100.0}, {3000.0, 140.0}, {5000.0, 200.0},
        {10000.0, 320.0}, {20000.0, 430.0}, {30000.0, 480.0}, {40000.0, 520.0},
        {50000.0, 560.0}, {60000.0, 580.0}, {70000.0, 595.0}, {80000.0, 600.0},
    }};

    if (raw_estimate <= kBiasTable.front().first) {
        return kBiasTable.front().second;
    }
    for (size_t i = 0; i + 1 < kBiasTable.size(); ++i) {
        const auto [x0, y0] = kBiasTable[i];
        const auto [x1, y1] = kBiasTable[i + 1];
        if (raw_estimate <= x1) {
            const double t = (raw_estimate - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    return kBiasTable.back().second;
}

uint8_t HyperLogLog::compute_rank(uint64_t hash) const {
    return rank_from_hash(hash, precision_);
}

double HyperLogLog::raw_estimate() const {
    double sum = 0.0;
    for (uint8_t reg : registers_) {
        sum += std::pow(2.0, -static_cast<int>(reg));
    }
    return alpha() * (static_cast<double>(m_) * static_cast<double>(m_)) / sum;
}

double HyperLogLog::estimate() const {
    const double raw = raw_estimate();
    double estimator = raw;
    if (precision_ == 14 && raw <= 5.0 * static_cast<double>(m_)) {
        estimator = std::max(0.0, raw - bias(raw));
    }
    size_t zeros = 0;
    for (uint8_t reg : registers_) {
        if (reg == 0) {
            ++zeros;
        }
    }

    if (estimator <= 5.0 * static_cast<double>(m_) && zeros > 0) {
        return m_ * std::log(static_cast<double>(m_) / static_cast<double>(zeros));
    }
    const double two_to_32 = static_cast<double>(1ull << 32);
    if (estimator > two_to_32 / 30.0) {
        return -two_to_32 * std::log(1.0 - estimator / two_to_32);
    }
    return estimator;
}

size_t HyperLogLog::memory_usage_bytes() const {
    return registers_.size() * sizeof(uint8_t);
}

// ------------------------ HyperLogLogPlusPlus --------------------------------

HyperLogLogPlusPlus::HyperLogLogPlusPlus(uint32_t precision,
                                         double sparse_threshold_factor,
                                         uint64_t seed)
    : HyperLogLog(precision, seed, false),
      dense_(false),
      threshold_(1) {
    const double scaled =
        sparse_threshold_factor <= 0.0
            ? static_cast<double>(m_)
            : sparse_threshold_factor * static_cast<double>(m_);
    threshold_ = static_cast<size_t>(std::max<double>(1.0, std::min<double>(scaled, static_cast<double>(m_))));
}

void HyperLogLogPlusPlus::add(uint64_t value) {
    const uint64_t hashed = hash64(value, seed_);
    if (!dense_) {
        auto it = std::lower_bound(sparse_list_.begin(), sparse_list_.end(), hashed);
        if (it == sparse_list_.end() || *it != hashed) {
            sparse_list_.insert(it, hashed);
            if (sparse_list_.size() >= threshold_) {
                convert_to_dense();
                return;
            } else {
                return;
            }
        } else {
            return;
        }
    }
    HyperLogLog::add(value);
}

void HyperLogLogPlusPlus::convert_to_dense() {
    if (dense_) {
        return;
    }
    if (registers_.size() != m_) {
        registers_.assign(m_, 0);
    }
    for (uint64_t hash : sparse_list_) {
        const uint32_t idx = static_cast<uint32_t>(hash >> (64 - precision_));
        const uint8_t rank = compute_rank(hash);
        if (rank > registers_[idx]) {
            registers_[idx] = rank;
        }
    }
    sparse_list_.clear();
    sparse_list_.shrink_to_fit();
    dense_ = true;
}

double HyperLogLogPlusPlus::bias(double raw_estimate) const {
    if (precision_ != 14) {
        return 0.0;
    }
    static const std::array<std::pair<double, double>, 12> kBiasTable = {{
        {1000.0, 60.0},  {2000.0, 100.0}, {3000.0, 140.0}, {5000.0, 200.0},
        {10000.0, 320.0}, {20000.0, 430.0}, {30000.0, 480.0}, {40000.0, 520.0},
        {50000.0, 560.0}, {60000.0, 580.0}, {70000.0, 595.0}, {80000.0, 600.0},
    }};

    if (raw_estimate <= kBiasTable.front().first) {
        return kBiasTable.front().second;
    }
    for (size_t i = 0; i + 1 < kBiasTable.size(); ++i) {
        const auto [x0, y0] = kBiasTable[i];
        const auto [x1, y1] = kBiasTable[i + 1];
        if (raw_estimate <= x1) {
            const double t = (raw_estimate - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    return kBiasTable.back().second;
}

double HyperLogLogPlusPlus::estimate() const {
    if (!dense_) {
        return static_cast<double>(sparse_list_.size());
    }

    const double raw = raw_estimate();
    double estimator = raw;
    if (raw <= 5.0 * static_cast<double>(m_) && precision_ == 14) {
        estimator = std::max(0.0, raw - bias(raw));
    }

    size_t zeros = 0;
    for (uint8_t reg : registers_) {
        if (reg == 0) {
            ++zeros;
        }
    }
    if (zeros > 0 && estimator < 2.5 * static_cast<double>(m_)) {
        return m_ * std::log(static_cast<double>(m_) / static_cast<double>(zeros));
    }

    const double two_to_32 = static_cast<double>(1ull << 32);
    if (estimator > two_to_32 / 30.0) {
        return -two_to_32 * std::log(1.0 - estimator / two_to_32);
    }
    return estimator;
}

size_t HyperLogLogPlusPlus::memory_usage_bytes() const {
    if (!dense_) {
        return sparse_list_.size() * sizeof(uint64_t);
    }
    return registers_.size() * sizeof(uint8_t);
}

}  // namespace cardinality
