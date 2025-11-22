#include "cardinality_estimators.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <cctype>

namespace fs = std::filesystem;
using namespace cardinality;

struct Args {
    fs::path dataset_dir = "datasets/cardinality";
    fs::path out_dir = "results/cardinality";
    uint32_t precision = 14;
    int reps = 3;
};

struct RawSample {
    size_t checkpoint;
    size_t stream_index;
    size_t true_cardinality;
    double estimate;
    double relative_error;
};

struct EstimatorResult {
    std::vector<RawSample> samples;
    double final_estimate = 0.0;
    double throughput = 0.0;
    double elapsed_seconds = 0.0;
    double final_error = 0.0;
    double avg_checkpoint_error = 0.0;
    double max_checkpoint_error = 0.0;
    size_t memory_bytes = 0;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset-dir" && i + 1 < argc) {
            args.dataset_dir = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            args.out_dir = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            args.precision = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--reps" && i + 1 < argc) {
            args.reps = std::stoi(argv[++i]);
        }
    }
    return args;
}

std::unique_ptr<CardinalityEstimator> create_estimator(const std::string& name, uint32_t precision) {
    if (name == "ll") {
        return std::make_unique<LogLog>(precision);
    }
    if (name == "hll") {
        return std::make_unique<HyperLogLog>(precision);
    }
    if (name == "hll++") {
        return std::make_unique<HyperLogLogPlusPlus>(precision);
    }
    throw std::runtime_error("Unknown algorithm: " + name);
}

std::vector<int64_t> load_numbers(const fs::path& path, std::vector<size_t>& unique_progression) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open dataset: " + path.string());
    std::vector<int64_t> numbers;
    numbers.reserve(1'000'000);
    unique_progression.reserve(1'000'000);
    std::unordered_set<int64_t> seen;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        int64_t value = std::stoll(line);
        numbers.push_back(value);
        seen.insert(value);
        unique_progression.push_back(seen.size());
    }
    return numbers;
}

std::vector<size_t> generate_checkpoints(size_t max_cardinality, size_t target_points = 10) {
    if (max_cardinality == 0) return {0};
    if (max_cardinality <= target_points) {
        std::vector<size_t> cp;
        for (size_t i = 1; i <= max_cardinality; ++i) cp.push_back(i);
        return cp;
    }
    std::vector<size_t> checkpoints;
    if (max_cardinality <= 5000) {
        size_t step = std::max<size_t>(1, max_cardinality / target_points);
        for (size_t cp = step; cp < max_cardinality; cp += step) {
            checkpoints.push_back(cp);
        }
    } else {
        double start = std::max(10.0, max_cardinality / (target_points * 10.0));
        double ratio = std::pow(static_cast<double>(max_cardinality) / start, 1.0 / (target_points - 1));
        double value = start;
        for (size_t i = 0; i < target_points - 1; ++i) {
            checkpoints.push_back(static_cast<size_t>(std::max(1.0, std::round(value))));
            value *= ratio;
        }
    }
    checkpoints.push_back(max_cardinality);
    return checkpoints;
}

EstimatorResult run_estimator(const std::string& name,
                              uint32_t precision,
                              const std::vector<int64_t>& numbers,
                              const std::vector<size_t>& unique_progression,
                              const std::vector<size_t>& checkpoints) {
    auto estimator = create_estimator(name, precision);
    EstimatorResult result;
    result.samples.reserve(checkpoints.size());

    size_t checkpoint_idx = 0;
    size_t checkpoint_target = checkpoints[checkpoint_idx];
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numbers.size(); ++i) {
        uint64_t hashed_input = static_cast<uint64_t>(numbers[i]);
        estimator->add(hashed_input);
        size_t true_card = unique_progression[i];
        if (true_card >= checkpoint_target) {
            double estimate = estimator->estimate();
            double rel_err = true_card ? std::abs(estimate - true_card) / true_card : 0.0;
            result.samples.push_back({checkpoint_target, i + 1, true_card, estimate, rel_err});
            checkpoint_idx++;
            if (checkpoint_idx == checkpoints.size()) break;
            checkpoint_target = checkpoints[checkpoint_idx];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    if (result.elapsed_seconds <= 0) result.elapsed_seconds = 1e-9;
    result.throughput = numbers.size() / result.elapsed_seconds;
    double final_estimate = estimator->estimate();
    double final_true = unique_progression.back();
    result.final_estimate = final_estimate;
    result.final_error = final_true ? std::abs(final_estimate - final_true) / final_true : 0.0;
    result.memory_bytes = estimator->memory_usage_bytes();

    double sum_error = 0.0;
    double max_error = 0.0;
    for (const auto& sample : result.samples) {
        sum_error += sample.relative_error;
        if (sample.relative_error > max_error) max_error = sample.relative_error;
    }
    if (!result.samples.empty()) {
        result.avg_checkpoint_error = sum_error / result.samples.size();
        result.max_checkpoint_error = max_error;
    } else {
        result.avg_checkpoint_error = result.final_error;
        result.max_checkpoint_error = result.final_error;
    }

    return result;
}

std::pair<std::string, size_t> parse_metadata(const fs::path& path) {
    std::string stem = path.stem().string();
    size_t last_underscore = stem.find_last_of('_');
    if (last_underscore == std::string::npos) {
        return {stem, 0};
    }
    std::string suffix = stem.substr(last_underscore + 1);
    bool numeric = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
    if (numeric) {
        size_t declared = std::stoull(suffix);
        std::string distribution = stem.substr(0, last_underscore);
        return {distribution, declared};
    }
    return {stem, 0};
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::create_directories(args.out_dir);

        std::ofstream raw_csv(args.out_dir / "cardinality_raw.csv");
        std::ofstream summary_csv(args.out_dir / "cardinality_summary.csv");

        raw_csv << "dataset,distribution,dataset_size,precision,rep,algorithm,checkpoint,stream_index,"
                   "true_cardinality,estimated_cardinality,relative_error\n";
        summary_csv << "dataset,distribution,dataset_size,precision,rep,algorithm,stream_length,"
                       "true_cardinality,final_estimate,relative_error,memory_bytes,throughput_ops,"
                       "elapsed_seconds,avg_checkpoint_error,max_checkpoint_error\n";

        std::vector<std::string> algorithms = {"ll", "hll", "hll++"};
        std::vector<fs::path> dataset_files;
        for (const auto& entry : fs::directory_iterator(args.dataset_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                dataset_files.push_back(entry.path());
            }
        }
        std::sort(dataset_files.begin(), dataset_files.end());
        if (dataset_files.empty()) {
            std::cerr << "No dataset files found in " << args.dataset_dir << "\n";
            return 1;
        }

        for (const auto& dataset_path : dataset_files) {
            std::cout << "[cardinality] Processing " << dataset_path << "...\n";
            std::vector<size_t> unique_progression;
            auto numbers = load_numbers(dataset_path, unique_progression);
            if (numbers.empty()) continue;

            auto [distribution, declared_size] = parse_metadata(dataset_path);
            size_t dataset_size = declared_size ? declared_size : numbers.size();
            auto checkpoints = generate_checkpoints(unique_progression.back());

            for (int rep = 0; rep < args.reps; ++rep) {
                for (const auto& algo : algorithms) {
                    auto result =
                        run_estimator(algo, args.precision, numbers, unique_progression, checkpoints);
                    for (const auto& sample : result.samples) {
                        raw_csv << dataset_path.filename().string() << ","
                                << distribution << ","
                                << dataset_size << ","
                                << args.precision << ","
                                << rep << ","
                                << algo << ","
                                << sample.checkpoint << ","
                                << sample.stream_index << ","
                                << sample.true_cardinality << ","
                                << sample.estimate << ","
                                << sample.relative_error << "\n";
                    }
                    summary_csv << dataset_path.filename().string() << ","
                                << distribution << ","
                                << dataset_size << ","
                                << args.precision << ","
                                << rep << ","
                                << algo << ","
                                << numbers.size() << ","
                                << unique_progression.back() << ","
                                << result.final_estimate << ","
                                << result.final_error << ","
                                << result.memory_bytes << ","
                                << result.throughput << ","
                                << result.elapsed_seconds << ","
                                << result.avg_checkpoint_error << ","
                                << result.max_checkpoint_error << "\n";
                }
            }
        }

        std::cout << "[cardinality] Results saved to " << args.out_dir << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}