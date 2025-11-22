CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
INCLUDES = -I src/frievald/cpp

BIN_DIR = bin
SRC_FRIEVALD = src/frievald/cpp
BENCH_FRIEVALD = benchmarks/frievald/cpp
SRC_QSORT = src/qsort/cpp

# Targets
all: $(BIN_DIR)/frievald_benchmark_runtime $(BIN_DIR)/frievald_benchmark_error $(BIN_DIR)/random_qsort $(BIN_DIR)/miller_rabin

$(BIN_DIR)/frievald_benchmark_runtime: $(BENCH_FRIEVALD)/benchmark_runtime.cpp $(SRC_FRIEVALD)/algorithms.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BIN_DIR)/frievald_benchmark_error: $(BENCH_FRIEVALD)/benchmark_error.cpp $(SRC_FRIEVALD)/algorithms.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BIN_DIR)/random_qsort: $(SRC_QSORT)/random_qsort.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/miller_rabin: src/miller_rabin/cpp/miller_rabin.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
