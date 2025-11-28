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

using namespace std;
namespace fs = std::filesystem;

struct Counters {
    unsigned long long comparisons = 0;
    unsigned long long swaps = 0;
    int max_recursion_depth = 0;
    int bad_split_count = 0;
};

static void swap_if_needed(int &a, int &b, Counters &c) {
    if (&a == &b) return;
    swap(a, b);
    c.swaps++;
}

static int partition(vector<int> &arr, int L, int R, mt19937 &rng, Counters &c) {
    uniform_int_distribution<int> dist(L, R);
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

static void quicksort(vector<int> &arr, int L, int R, mt19937 &rng, Counters &c, int depth) {
    c.max_recursion_depth = max(c.max_recursion_depth, depth);
    if (L >= R) return;
    
    int p = partition(arr, L, R, rng, c);
    
    // Check for bad split (e.g., > 90% on one side)
    int len = R - L + 1;
    int left_len = p - L;
    int right_len = R - p;
    if (len > 10) { // Only count for reasonable segment sizes
        double ratio = (double)max(left_len, right_len) / len;
        if (ratio > 0.9) {
            c.bad_split_count++;
        }
    }

    quicksort(arr, L, p - 1, rng, c, depth + 1);
    quicksort(arr, p + 1, R, rng, c, depth + 1);
}

struct Args {
    string input_file;
    string out_csv;
    long long seed = 0;
    long long rep_id = 0;
};

static bool parse_args(int argc, char **argv, Args &args, string &err) {
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
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

static string utc_iso_timestamp() {
    using namespace chrono;
    auto now = system_clock::now();
    time_t t = system_clock::to_time_t(now);

    tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    ostringstream oss;
    oss << put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

static string infer_category_from_path(const string &path_str) {
    try {
        fs::path p(path_str);
        auto parent = p.parent_path();
        if (!parent.empty()) {
            return parent.filename().string();
        }
    } catch (...) {
        // ignore
    }
    return string("unknown");
}

static bool is_file_nonempty(const fs::path &p) {
    error_code ec;
    if (!fs::exists(p, ec)) return false;
    auto sz = fs::file_size(p, ec);
    if (ec) return false;
    return sz > 0;
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

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

    // Read numbers (supports integers, floats, doubles - converts to int)
    vector<int> data;
    {
        ifstream fin(args.input_file);
        if (!fin) {
            cerr << "Error: failed to open input file: " << args.input_file << "\n";
            return 3;
        }
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

    vector<int> arr = data; // copy for quicksort
    vector<int> arr_sorted = data; // copy for std::sort verification

    Counters counters;
    mt19937 rng(static_cast<uint32_t>(args.seed));

    auto t0 = chrono::high_resolution_clock::now();
    quicksort(arr, 0, static_cast<int>(arr.size()) - 1, rng, counters, 0);
    auto t1 = chrono::high_resolution_clock::now();

    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

    // Measure std::sort time
    auto t2 = chrono::high_resolution_clock::now();
    sort(arr_sorted.begin(), arr_sorted.end());
    auto t3 = chrono::high_resolution_clock::now();
    auto std_sort_ms = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();

    int correct = (arr == arr_sorted) ? 1 : 0;

    string timestamp = utc_iso_timestamp();
    string category = infer_category_from_path(args.input_file);
    long long n = static_cast<long long>(data.size());

    const string header = "timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,comparisons,swaps,correct,std_sort_ms,recursion_depth,bad_split_count";

    // Ensure output directory exists (fail with friendly msg if not)
    fs::path out_path(args.out_csv);
    if (!out_path.parent_path().empty() && !fs::exists(out_path.parent_path())) {
        cerr << "Error: output directory does not exist: " << out_path.parent_path().string() << "\n";
        return 6;
    }

    bool need_header = !is_file_nonempty(out_path);

    // Compose CSV row
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
        << counters.bad_split_count;

    // Append to CSV file
    {
        ofstream fout(args.out_csv, ios::app);
        if (!fout) {
            cerr << "Error: cannot open out CSV for appending: " << args.out_csv << "\n";
            return 7;
        }
        if (need_header) fout << header << '\n';
        fout << row.str() << '\n';
    }

    // Emit to stdout: print header only if file was empty, and always the row
    if (need_header) {
        cout << header << "\n";
    }
    cout << row.str() << "\n";

    return 0;
}
