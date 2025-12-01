/**
 * @file mincut.cpp
 * @brief Implementation of Karger's and Karger-Stein Minimum Cut algorithms.
 */

#include "mincut.hpp"
#include <fstream>      // Used for file input operations (reading graph files).
#include <iostream>     // Used for standard I/O (debugging, error messages).
#include <sstream>      // Used for parsing lines from input files.
#include <unordered_map>// Used for mapping arbitrary vertex IDs to contiguous 0..V-1 IDs.
#include <algorithm>    // Used for std::min, std::iota.
#include <cmath>        // Used for std::ceil, std::sqrt.
#include <numeric>      // Used for std::iota (filling DSU parent array).

/**
 * @brief Disjoint Set Union (DSU) data structure.
 * 
 * Used to efficiently manage connected components during edge contraction.
 * Supports union-find operations with path compression.
 */
struct DSU {
    std::vector<int> parent; /**< Stores the parent of each node. parent[i] == i means i is a root. */
    int components;          /**< Tracks the number of active components (vertices) remaining. */

    /**
     * @brief Constructs a DSU with n elements.
     * @param n Number of elements.
     */
    DSU(int n) : parent(n), components(n) {
        std::iota(parent.begin(), parent.end(), 0); // Initialize parent[i] = i
    }

    /**
     * @brief Finds the representative (root) of the set containing i.
     * Applies path compression.
     * @param i Element to find.
     * @return int Root of the set.
     */
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }

    /**
     * @brief Unites the sets containing i and j.
     * @param i First element.
     * @param j Second element.
     * @return true If i and j were in different sets and are now merged.
     * @return false If i and j were already in the same set.
     */
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            components--;
            return true;
        }
        return false;
    }
};

Graph load_graph(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    Graph g;
    std::vector<Edge> edges;
    std::unordered_map<int, int> id_map;
    int next_id = 0;

    std::string line;
    int u_in, v_in;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::stringstream ss(line);
        if (!(ss >> u_in >> v_in)) continue;

        if (id_map.find(u_in) == id_map.end()) id_map[u_in] = next_id++;
        if (id_map.find(v_in) == id_map.end()) id_map[v_in] = next_id++;
        
        if (u_in != v_in) { // Ignore self-loops in input
            edges.push_back({id_map[u_in], id_map[v_in]});
        }
    }

    g.V = next_id;
    g.edges = edges;
    g.E = edges.size();
    return g;
}

/**
 * @brief Contracts the graph g down to k vertices.
 * 
 * This is a helper function used by both Karger and Karger-Stein.
 * It randomly selects edges and contracts them (merges endpoints) using DSU
 * until the number of components equals k.
 * 
 * @param g The input graph.
 * @param k The target number of vertices.
 * @param rng Random number generator.
 * @return Graph A new graph with k vertices and preserved edges (excluding self-loops).
 */
Graph contract(const Graph& g, int k, std::mt19937& rng) {
    if (g.V <= k) return g;

    DSU dsu(g.V);
    int current_vertices = g.V;
    
    // We need to pick edges randomly. 
    // To do this efficiently without removing from vector repeatedly:
    // We can shuffle the edges or pick random indices. 
    // Since we need to contract (V - k) times, and E might be large,
    // picking random indices is better if E >> V.
    // However, we need to avoid picking self-loops repeatedly.
    
    // For Karger-Stein, graphs get small, so shuffling might be okay.
    // Let's use a local copy of edges indices to shuffle/pick from?
    // Or just Fisher-Yates style swap-to-end.
    
    std::vector<Edge> active_edges = g.edges;
    
    while (current_vertices > k && !active_edges.empty()) {
        // Pick a random edge
        std::uniform_int_distribution<> dis(0, active_edges.size() - 1);
        int idx = dis(rng);
        
        Edge e = active_edges[idx];
        
        // Swap with last to remove in O(1)
        active_edges[idx] = active_edges.back();
        active_edges.pop_back();
        
        if (dsu.unite(e.u, e.v)) {
            current_vertices--;
        }
    }
    
    // Construct new graph
    // Map old components to new vertex IDs 0..k-1
    std::vector<int> component_map(g.V, -1);
    int new_id_counter = 0;
    
    Graph new_g;
    new_g.V = current_vertices;
    
    // We iterate over ALL edges of the input graph 'g'.
    // Find new endpoints using DSU.
    // If new_u != new_v, add to new graph.
    
    for (const auto& e : g.edges) {
        int root_u = dsu.find(e.u);
        int root_v = dsu.find(e.v);
        
        if (root_u != root_v) {
            if (component_map[root_u] == -1) component_map[root_u] = new_id_counter++;
            if (component_map[root_v] == -1) component_map[root_v] = new_id_counter++;
            
            new_g.edges.push_back({component_map[root_u], component_map[root_v]});
        }
    }
    new_g.E = new_g.edges.size();
    return new_g;
}

int kargerMinCut(const Graph& g, std::mt19937& rng) {
    Graph contracted = contract(g, 2, rng);
    return contracted.E;
}

int kargerSteinMinCut(const Graph& g, std::mt19937& rng) {
    if (g.V <= 6) {
        return kargerMinCut(g, rng);
    }

    int t = std::ceil(1.0 + g.V / std::sqrt(2.0));

    Graph g1 = contract(g, t, rng);
    int res1 = kargerSteinMinCut(g1, rng);

    Graph g2 = contract(g, t, rng); // Independent contraction
    int res2 = kargerSteinMinCut(g2, rng);

    return std::min(res1, res2);
}
