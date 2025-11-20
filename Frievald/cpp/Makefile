CXX = g++
CXXFLAGS = -std=c++17 -Wall -I src

SRC_DIR = src
BENCH_DIR = benchmarks
BIN_DIR = bin

# Targets
all: $(BIN_DIR)/benchmark_runtime $(BIN_DIR)/benchmark_error

$(BIN_DIR)/benchmark_runtime: $(BENCH_DIR)/benchmark_runtime.cpp $(SRC_DIR)/algorithms.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/benchmark_error: $(BENCH_DIR)/benchmark_error.cpp $(SRC_DIR)/algorithms.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf $(BIN_DIR)

run_benchmarks: all
	./$(BIN_DIR)/benchmark_runtime
	./$(BIN_DIR)/benchmark_error
	python3 process_results.py

.PHONY: all clean
