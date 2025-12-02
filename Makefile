CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
INCLUDES = -I src/frievald/cpp -I src/cardinality/cpp -I src/mincut/cpp

BIN_DIR = bin
SRC_QSORT = src/qsort/cpp
SRC_CARD = src/cardinality/cpp
SRC_MINCUT = src/mincut/cpp

# Targets
all: $(BIN_DIR)/random_qsort $(BIN_DIR)/miller_rabin $(BIN_DIR)/cardinality_benchmark $(BIN_DIR)/mincut_benchmark

$(BIN_DIR)/random_qsort: $(SRC_QSORT)/random_qsort.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BIN_DIR)/random_qsort: $(SRC_QSORT)/random_qsort.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/miller_rabin: src/miller_rabin/cpp/miller_rabin.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/cardinality_benchmark: $(SRC_CARD)/cardinality_estimators.cpp benchmarks/cardinality/cpp/cardinality_benchmark.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BIN_DIR)/mincut_benchmark: $(SRC_MINCUT)/mincut.cpp benchmarks/mincut/cpp/mincut_benchmark.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
