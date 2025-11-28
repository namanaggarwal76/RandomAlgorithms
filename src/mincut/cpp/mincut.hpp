#pragma once

#include <vector>
#include <utility>
#include <string>
#include <random>

struct Edge {
    int u, v;
};

struct Graph {
    int V; // Number of vertices
    int E; // Number of edges
    std::vector<Edge> edges;
};

// Load graph from file (handles remapping vertices to 0..V-1)
Graph load_graph(const std::string& filename);

// Single run of Karger's contraction algorithm
// Returns the cut size found
int kargerMinCut(Graph g, std::mt19937& rng);

// Single run of Karger-Stein algorithm
// Returns the cut size found
int kargerSteinMinCut(Graph g, std::mt19937& rng);
