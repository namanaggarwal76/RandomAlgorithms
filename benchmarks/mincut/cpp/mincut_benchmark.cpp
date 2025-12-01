#include "mincut.hpp" // Include the mincut algorithm header

#include <algorithm> // Used for algorithms like sort, min, etc.
#include <chrono> // Used for timing measurements
#include <filesystem> // Used for filesystem operations
#include <fstream> // Used for file input/output
#include <iostream> // Used for standard input/output
#include <random> // Used for random number generation
#include <string> // Used for string manipulation
#include <vector> // Used for dynamic arrays
#include <map> // Used for key-value storage

namespace fs = std::filesystem;

/**
 * @brief Structure to hold benchmark results.
 */
struct Result {
    std::string algorithm;
    std::string dataset;
    int vertices;
    int edges;
    int cut_size;
    double duration_sec;
};

/**
 * @brief Writes benchmark results to a CSV file.
 * 
 * @param out_path Path to the output CSV file.
 * @param results Vector of benchmark results.
 */
void write_results(const std::string& out_path, const std::vector<Result>& results) {
    std::ofstream f(out_path);
    f << "algorithm,dataset,vertices,edges,cut_size,duration_sec\n";
    for (const auto& r : results) {
        f << r.algorithm << ","
          << r.dataset << ","
          << r.vertices << ","
          << r.edges << ","
          << r.cut_size << ","
          << r.duration_sec << "\n";
    }
}

int main(int argc, char** argv) {
    std::string dataset_dir = "datasets/mincut";
    std::string out_dir = "results/mincut";
    int reps = 5; // Number of repetitions per algorithm per graph

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset-dir" && i + 1 < argc) dataset_dir = argv[++i];
        else if (arg == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (arg == "--reps" && i + 1 < argc) reps = std::stoi(argv[++i]);
    }

    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    std::vector<Result> results;
    std::mt19937 rng(std::random_device{}());

    for (const auto& entry : fs::directory_iterator(dataset_dir)) {
        if (entry.path().extension() == ".txt") {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();
            
            std::cout << "Processing " << filename << "..." << std::endl;

            try {
                Graph g = load_graph(filepath);
                
                // Benchmark Karger
                for (int i = 0; i < reps; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();
                    int cut = kargerMinCut(g, rng);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = end - start;
                    
                    results.push_back({
                        "Karger",
                        filename,
                        g.V,
                        g.E,
                        cut,
                        diff.count()
                    });
                }

                // Benchmark Karger-Stein
                for (int i = 0; i < reps; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();
                    int cut = kargerSteinMinCut(g, rng);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = end - start;
                    
                    results.push_back({
                        "Karger-Stein",
                        filename,
                        g.V,
                        g.E,
                        cut,
                        diff.count()
                    });
                }

            } catch (const std::exception& e) {
                std::cerr << "Error processing " << filename << ": " << e.what() << std::endl;
            }
        }
    }

    write_results(out_dir + "/mincut_results.csv", results);
    std::cout << "Results written to " << out_dir << "/mincut_results.csv" << std::endl;

    return 0;
}
