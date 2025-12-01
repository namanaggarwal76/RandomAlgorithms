/**
 * File: random_qsort.cpp
 * Description: Implementation of randomized quicksort with comprehensive performance metrics.
 *              Measures time complexity, comparisons, swaps, recursion depth, and space complexity.
 */

#include <algorithm>      // Used for std::sort (verification baseline)
#include <chrono>         // Used for high-resolution timing measurements
#include <cstdint>        // Used for fixed-width integer types
#include <cstdio>         // Used for C-style I/O operations
#include <ctime>          // Used for time conversion functions
#include <filesystem>     // Used for file system operations and path handling
#include <fstream>        // Used for file I/O (reading input data, writing CSV)
#include <iomanip>        // Used for I/O manipulators (formatting timestamps)
#include <iostream>       // Used for standard I/O streams (console output)
#include <random>         // Used for random number generation (pivot selection)
#include <sstream>        // Used for string stream operations (formatting output)
#include <string>         // Used for string operations
#include <vector>         // Used for dynamic array storage of input data

using namespace std;
namespace fs = std::filesystem;

/**
 * Counters: Structure to track performance metrics during quicksort execution.
 * 
 * Members:
 *   comparisons: Total number of element comparisons performed
 *   swaps: Total number of element swaps performed
 *   max_recursion_depth: Maximum depth of recursive calls (historical)
 *   bad_split_count: Number of unbalanced partitions (>90% on one side)
 *   max_stack_depth: Maximum call stack depth (space complexity metric)
 *   estimated_stack_bytes: Estimated peak stack memory usage in bytes
 */
struct Counters {
    unsigned long long comparisons = 0;
    unsigned long long swaps = 0;
    int max_recursion_depth = 0;
    int bad_split_count = 0;
    int max_stack_depth = 0;  // Track maximum call stack depth for space complexity
    unsigned long long estimated_stack_bytes = 0;  // Estimated stack memory usage
};

/**
 * Swaps two elements if they are different (not the same memory location).
 * Increments the swap counter.
 * 
 * Args:
 *     a (int&): Reference to first element
 *     b (int&): Reference to second element
 *     c (Counters&): Reference to performance counters
 */
static void swap_if_needed(int &a, int &b, Counters &c) {
    if (&a == &b) return;  // Skip if same element
    swap(a, b);
    c.swaps++;
}

/**
 * Partitions the array around a randomly selected pivot element.
 * Uses the Lomuto partition scheme with randomized pivot selection.
 * 
 * Args:
 *     arr (vector<int>&): Array to partition
 *     L (int): Left boundary of partition range (inclusive)
 *     R (int): Right boundary of partition range (inclusive)
 *     rng (mt19937&): Random number generator for pivot selection
 *     c (Counters&): Reference to performance counters
 *     
 * Returns:
 *     int: Final position of the pivot element after partitioning
 */
static int partition(vector<int> &arr, int L, int R, mt19937 &rng, Counters &c) {
    // Randomly select pivot index from range [L, R]
    uniform_int_distribution<int> dist(L, R);
    int pivot_index = dist(rng);
    
    // Move pivot to the end for partitioning
    swap_if_needed(arr[pivot_index], arr[R], c);
    int pivot = arr[R];

    int i = L; // Position for the next smaller-or-equal element
    
    // Partition: move elements <= pivot to the left
    for (int j = L; j < R; ++j) {
        c.comparisons++;
        if (arr[j] <= pivot) {
            swap_if_needed(arr[i], arr[j], c);
            ++i;
        }
    }
    
    // Place pivot in its final sorted position
    swap_if_needed(arr[i], arr[R], c);
    return i;
}

/**
 * Recursive randomized quicksort implementation with performance tracking.
 * 
 * Algorithm:
 *   1. Select random pivot and partition array
 *   2. Recursively sort left and right subarrays
 *   3. Track space complexity metrics (stack depth and memory)
 * 
 * Args:
 *     arr (vector<int>&): Array to sort
 *     L (int): Left boundary of sort range (inclusive)
 *     R (int): Right boundary of sort range (inclusive)
 *     rng (mt19937&): Random number generator for pivot selection
 *     c (Counters&): Reference to performance counters
 *     depth (int): Current recursion depth (for tracking space complexity)
 */
static void quicksort(vector<int> &arr, int L, int R, mt19937 &rng, Counters &c, int depth) {
    // Update maximum recursion depth reached
    c.max_recursion_depth = max(c.max_recursion_depth, depth);
    c.max_stack_depth = max(c.max_stack_depth, depth);
    
    // Estimate stack memory usage: each recursive call frame uses roughly
    // 4 bytes (L) + 4 bytes (R) + 8 bytes (counters ptr) + 4 bytes (depth) 
    // + overhead for rng reference and return address (~32 bytes per frame conservatively)
    const int BYTES_PER_FRAME = 64;  // Conservative estimate
    unsigned long long estimated_bytes = static_cast<unsigned long long>(depth + 1) * BYTES_PER_FRAME;
    c.estimated_stack_bytes = max(c.estimated_stack_bytes, estimated_bytes);
    
    // Base case: single element or empty range
    if (L >= R) return;
    
    // Partition around random pivot
    int p = partition(arr, L, R, rng, c);
    
    // Check for bad split (> 90% elements on one side)
    // This helps identify pathological behavior
    int len = R - L + 1;
    int left_len = p - L;
    int right_len = R - p;
    if (len > 10) { // Only count for reasonable segment sizes
        double ratio = (double)max(left_len, right_len) / len;
        if (ratio > 0.9) {
            c.bad_split_count++;
        }
    }

    // Recursively sort left and right partitions
    quicksort(arr, L, p - 1, rng, c, depth + 1);
    quicksort(arr, p + 1, R, rng, c, depth + 1);
}

/**
 * Args: Structure to hold command-line arguments.
 * 
 * Members:
 *   input_file: Path to input data file
 *   out_csv: Path to output CSV file for results
 *   seed: Random seed for reproducible pivot selection
 *   rep_id: Replication ID for multiple runs
 */
struct Args {
    string input_file;
    string out_csv;
    long long seed = 0;
    long long rep_id = 0;
};

/**
 * Parses command-line arguments.
 * 
 * Args:
 *     argc (int): Argument count
 *     argv (char**): Argument values
 *     args (Args&): Output structure to populate with parsed arguments
 *     err (string&): Error message output (if parsing fails)
 *     
 * Returns:
 *     bool: True if parsing succeeded, false otherwise
 */
static bool parse_args(int argc, char **argv, Args &args, string &err) {
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        // Lambda to get next argument value
        auto next = [&](const char *name) -> string {
            if (i + 1 >= argc) {
                ostringstream oss;
                oss << "Missing value for " << name;
                err = oss.str();
                return {};
            }
            return string(argv[++i]);
        };
        if (a == "--input-file") {
            args.input_file = next("--input-file");
            if (!err.empty()) return false;
        } else if (a == "--seed") {
            string v = next("--seed");
            if (!err.empty()) return false;
            try { args.seed = stoll(v); } catch (...) { err = "--seed must be an integer"; return false; }
        } else if (a == "--rep") {
            string v = next("--rep");
            if (!err.empty()) return false;
            try { args.rep_id = stoll(v); } catch (...) { err = "--rep must be an integer"; return false; }
        } else if (a == "--out-csv") {
            args.out_csv = next("--out-csv");
            if (!err.empty()) return false;
        } else if (a == "-h" || a == "--help") {
            cout << "Usage: ./random_qsort --input-file <path> --seed <int> --rep <rep_id> --out-csv <path>\n";
            return false;
        } else {
            err = string("Unknown argument: ") + a;
            return false;
        }
    }

    if (args.input_file.empty() || args.out_csv.empty()) {
        err = "Required args: --input-file, --seed, --rep, --out-csv";
        return false;
    }
    return true;
}

/**
 * Returns current UTC time in ISO 8601 format.
 * 
 * Returns:
 *     string: Timestamp in format "YYYY-MM-DDTHH:MM:SSZ"
 */
static string utc_iso_timestamp() {
    using namespace chrono;
    auto now = system_clock::now();
    time_t t = system_clock::to_time_t(now);

    tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);  // Windows-specific thread-safe version
#else
    gmtime_r(&t, &tm);  // POSIX thread-safe version
#endif
    ostringstream oss;
    oss << put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

/**
 * Infers the data category from the input file path.
 * Extracts the parent directory name (e.g., "random", "sorted", "duplicates").
 * 
 * Args:
 *     path_str (const string&): Path to input file
 *     
 * Returns:
 *     string: Category name or "unknown" if extraction fails
 */
static string infer_category_from_path(const string &path_str) {
    try {
        fs::path p(path_str);
        auto parent = p.parent_path();
        if (!parent.empty()) {
            return parent.filename().string();
        }
    } catch (...) {
        // Silently handle filesystem errors
    }
    return string("unknown");
}

/**
 * Checks if a file exists and has non-zero size.
 * 
 * Args:
 *     p (const fs::path&): Path to file
 *     
 * Returns:
 *     bool: True if file exists and is non-empty, false otherwise
 */
static bool is_file_nonempty(const fs::path &p) {
    error_code ec;
    if (!fs::exists(p, ec)) return false;
    auto sz = fs::file_size(p, ec);
    if (ec) return false;
    return sz > 0;
}

/**
 * Main function: Runs randomized quicksort benchmark.
 * 
 * Workflow:
 *   1. Parse command-line arguments
 *   2. Read input data from file
 *   3. Execute randomized quicksort with performance tracking
 *   4. Verify correctness against std::sort
 *   5. Write results to CSV file
 *   
 * Returns:
 *     int: Exit code (0 = success, non-zero = error)
 */
int main(int argc, char **argv) {
    // Optimize I/O performance
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Parse command-line arguments
    Args args;
    string err;
    if (!parse_args(argc, argv, args, err)) {
        if (!err.empty()) cerr << "Error: " << err << "\n";
        return err.empty() ? 0 : 1;
    }

    // Validate input file
    if (!fs::exists(args.input_file)) {
        cerr << "Error: input file not found: " << args.input_file << "\n";
        return 2;
    }

    // Read input numbers (supports integers, floats, doubles - converts to int)
    vector<int> data;
    {
        ifstream fin(args.input_file);
        if (!fin) {
            cerr << "Error: failed to open input file: " << args.input_file << "\n";
            return 3;
        }
        // Read all numbers from file
        double x;
        while (fin >> x) {
            data.push_back(static_cast<int>(x));
        }
        if (data.empty()) {
            cerr << "Error: input file contains no numbers or could not be parsed.\n";
            return 4;
        }
        if (!fin.eof() && fin.fail()) {
            cerr << "Error: bad format while reading input numbers.\n";
            return 5;
        }
    }

    // Create copies: one for quicksort, one for verification
    vector<int> arr = data;        // Copy for randomized quicksort
    vector<int> arr_sorted = data; // Copy for std::sort verification

    // Initialize performance counters and random number generator
    Counters counters;
    mt19937 rng(static_cast<uint32_t>(args.seed));

    // Execute randomized quicksort and measure time
    auto t0 = chrono::high_resolution_clock::now();
    quicksort(arr, 0, static_cast<int>(arr.size()) - 1, rng, counters, 0);
    auto t1 = chrono::high_resolution_clock::now();

    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

    // Measure std::sort time for comparison baseline
    auto t2 = chrono::high_resolution_clock::now();
    sort(arr_sorted.begin(), arr_sorted.end());
    auto t3 = chrono::high_resolution_clock::now();
    auto std_sort_ms = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();

    // Verify correctness: compare with std::sort result
    int correct = (arr == arr_sorted) ? 1 : 0;

    // Gather metadata for result record
    string timestamp = utc_iso_timestamp();
    string category = infer_category_from_path(args.input_file);
    long long n = static_cast<long long>(data.size());

    // CSV header definition
    const string header = "timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,comparisons,swaps,correct,std_sort_ms,recursion_depth,bad_split_count,max_stack_depth,estimated_stack_bytes";

    // Ensure output directory exists
    fs::path out_path(args.out_csv);
    if (!out_path.parent_path().empty() && !fs::exists(out_path.parent_path())) {
        cerr << "Error: output directory does not exist: " << out_path.parent_path().string() << "\n";
        return 6;
    }

    // Determine if CSV header needs to be written
    bool need_header = !is_file_nonempty(out_path);

    // Compose CSV result row with all metrics
    ostringstream row;
    row << timestamp << ","
        << category << ","
        << args.input_file << ","
        << n << ","
        << args.seed << ","
        << args.rep_id << ","
        << elapsed_ms << ","
        << counters.comparisons << ","
        << counters.swaps << ","
        << correct << ","
        << std_sort_ms << ","
        << counters.max_recursion_depth << ","
        << counters.bad_split_count << ","
        << counters.max_stack_depth << ","
        << counters.estimated_stack_bytes;

    // Write results to CSV file
    {
        ofstream fout(args.out_csv, ios::app);
        if (!fout) {
            cerr << "Error: cannot open out CSV for appending: " << args.out_csv << "\n";
            return 7;
        }
        if (need_header) fout << header << '\n';
        fout << row.str() << '\n';
    }

    // Emit to stdout: print header only if file was empty, and always the data row
    if (need_header) {
        cout << header << "\n";
    }
    cout << row.str() << "\n";

    return 0;
}
