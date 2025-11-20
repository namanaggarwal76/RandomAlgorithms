#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Counters {
    unsigned long long comparisons = 0;
    unsigned long long swaps = 0;
};

static void swap_if_needed(int &a, int &b, Counters &c) {
    if (&a == &b) return;
    std::swap(a, b);
    c.swaps++;
}

static int partition(std::vector<int> &arr, int L, int R, std::mt19937 &rng, Counters &c) {
    std::uniform_int_distribution<int> dist(L, R);
    int pivot_index = dist(rng);
    swap_if_needed(arr[pivot_index], arr[R], c);
    int pivot = arr[R];

    int i = L; // place for the next smaller-or-equal element
    for (int j = L; j < R; ++j) {
        c.comparisons++;
        if (arr[j] <= pivot) {
            swap_if_needed(arr[i], arr[j], c);
            ++i;
        }
    }
    swap_if_needed(arr[i], arr[R], c);
    return i;
}

static void quicksort(std::vector<int> &arr, int L, int R, std::mt19937 &rng, Counters &c) {
    if (L >= R) return;
    int p = partition(arr, L, R, rng, c);
    quicksort(arr, L, p - 1, rng, c);
    quicksort(arr, p + 1, R, rng, c);
}

struct Args {
    std::string input_file;
    std::string out_csv;
    long long seed = 0;
    long long rep_id = 0;
};

static bool parse_args(int argc, char **argv, Args &args, std::string &err) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                std::ostringstream oss;
                oss << "Missing value for " << name;
                err = oss.str();
                return {};
            }
            return std::string(argv[++i]);
        };
        if (a == "--input-file") {
            args.input_file = next("--input-file");
            if (!err.empty()) return false;
        } else if (a == "--seed") {
            std::string v = next("--seed");
            if (!err.empty()) return false;
            try { args.seed = std::stoll(v); } catch (...) { err = "--seed must be an integer"; return false; }
        } else if (a == "--rep") {
            std::string v = next("--rep");
            if (!err.empty()) return false;
            try { args.rep_id = std::stoll(v); } catch (...) { err = "--rep must be an integer"; return false; }
        } else if (a == "--out-csv") {
            args.out_csv = next("--out-csv");
            if (!err.empty()) return false;
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: ./random_qsort --input-file <path> --seed <int> --rep <rep_id> --out-csv <path>\n";
            return false;
        } else {
            err = std::string("Unknown argument: ") + a;
            return false;
        }
    }

    if (args.input_file.empty() || args.out_csv.empty()) {
        err = "Required args: --input-file, --seed, --rep, --out-csv";
        return false;
    }
    return true;
}

static std::string utc_iso_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);

    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

static std::string infer_category_from_path(const std::string &path_str) {
    try {
        fs::path p(path_str);
        auto parent = p.parent_path();
        if (!parent.empty()) {
            return parent.filename().string();
        }
    } catch (...) {
        // ignore
    }
    return std::string("unknown");
}

static bool is_file_nonempty(const fs::path &p) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    auto sz = fs::file_size(p, ec);
    if (ec) return false;
    return sz > 0;
}

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Args args;
    std::string err;
    if (!parse_args(argc, argv, args, err)) {
        if (!err.empty()) std::cerr << "Error: " << err << "\n";
        return err.empty() ? 0 : 1;
    }

    // Validate input file
    if (!fs::exists(args.input_file)) {
        std::cerr << "Error: input file not found: " << args.input_file << "\n";
        return 2;
    }

    // Read numbers (supports integers, floats, doubles - converts to int)
    std::vector<int> data;
    {
        std::ifstream fin(args.input_file);
        if (!fin) {
            std::cerr << "Error: failed to open input file: " << args.input_file << "\n";
            return 3;
        }
        double x;
        while (fin >> x) {
            data.push_back(static_cast<int>(x));
        }
        if (data.empty()) {
            std::cerr << "Error: input file contains no numbers or could not be parsed.\n";
            return 4;
        }
        if (!fin.eof() && fin.fail()) {
            std::cerr << "Error: bad format while reading input numbers.\n";
            return 5;
        }
    }

    std::vector<int> arr = data; // copy for quicksort
    std::vector<int> arr_sorted = data; // copy for std::sort verification

    Counters counters;
    std::mt19937 rng(static_cast<uint32_t>(args.seed));

    auto t0 = std::chrono::high_resolution_clock::now();
    std::clock_t cpu0 = std::clock();
    quicksort(arr, 0, static_cast<int>(arr.size()) - 1, rng, counters);
    std::clock_t cpu1 = std::clock();
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    long long cpu_ms = static_cast<long long>((cpu1 - cpu0) * 1000.0 / CLOCKS_PER_SEC);

    std::sort(arr_sorted.begin(), arr_sorted.end());
    int correct = (arr == arr_sorted) ? 1 : 0;

    std::string timestamp = utc_iso_timestamp();
    std::string category = infer_category_from_path(args.input_file);
    long long n = static_cast<long long>(data.size());

    const std::string header = "timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,cpu_ms,comparisons,swaps,correct";

    // Ensure output directory exists (fail with friendly msg if not)
    fs::path out_path(args.out_csv);
    if (!out_path.parent_path().empty() && !fs::exists(out_path.parent_path())) {
        std::cerr << "Error: output directory does not exist: " << out_path.parent_path().string() << "\n";
        return 6;
    }

    bool need_header = !is_file_nonempty(out_path);

    // Compose CSV row
    std::ostringstream row;
    row << timestamp << ","
        << category << ","
        << args.input_file << ","
        << n << ","
        << args.seed << ","
        << args.rep_id << ","
        << elapsed_ms << ","
        << cpu_ms << ","
        << counters.comparisons << ","
        << counters.swaps << ","
        << correct;

    // Append to CSV file
    {
        std::ofstream fout(args.out_csv, std::ios::app);
        if (!fout) {
            std::cerr << "Error: cannot open out CSV for appending: " << args.out_csv << "\n";
            return 7;
        }
        if (need_header) fout << header << '\n';
        fout << row.str() << '\n';
    }

    // Emit to stdout: print header only if file was empty, and always the row
    if (need_header) {
        std::cout << header << "\n";
    }
    std::cout << row.str() << "\n";

    return 0;
}
