#pragma once

/**
 * @file mincut.hpp
 * @brief Header file for Karger's and Karger-Stein Minimum Cut algorithms.
 * 
 * This file defines the data structures (Edge, Graph) and function prototypes
 * for the randomized min-cut algorithms.
 */

#include <vector>   // Used for storing lists of edges and graph structures dynamically.
#include <utility>  // Used for std::pair (if needed) and other utility functions.
#include <string>   // Used for handling filenames and string manipulation.
#include <random>   // Used for std::mt19937 random number generator for randomized algorithms.

/**
 * @brief Represents an undirected edge in the graph.
 */
struct Edge {
    int u; /**< First endpoint of the edge */
    int v; /**< Second endpoint of the edge */
};

/**
 * @brief Represents a graph using an edge list.
 */
struct Graph {
    int V; /**< Number of vertices in the graph */
    int E; /**< Number of edges in the graph */
    std::vector<Edge> edges; /**< List of all edges in the graph */
};

/**
 * @brief Loads a graph from a text file.
 * 
 * Reads a graph from a file where each line represents an edge "u v".
 * Vertices are remapped to a 0-based contiguous range [0, V-1].
 * 
 * @param filename The path to the input graph file.
 * @return Graph The constructed Graph object.
 * @throws std::runtime_error If the file cannot be opened.
 */
Graph load_graph(const std::string& filename);

/**
 * @brief Executes a single run of Karger's Contraction Algorithm.
 * 
 * Repeatedly contracts random edges until only 2 vertices remain.
 * The number of edges between the remaining 2 vertices is the cut size.
 * 
 * @param g The input graph.
 * @param rng The random number generator engine.
 * @return int The size of the cut found (number of crossing edges).
 */
int kargerMinCut(const Graph& g, std::mt19937& rng);

/**
 * @brief Executes a single run of the Karger-Stein Recursive Algorithm.
 * 
 * Uses the recursive approach:
 * 1. If V <= 6, run Karger's algorithm.
 * 2. Otherwise, contract to t = ceil(1 + V/sqrt(2)) vertices twice independently.
 * 3. Return the minimum result of the two recursive calls.
 * 
 * @param g The input graph.
 * @param rng The random number generator engine.
 * @return int The size of the cut found.
 */
int kargerSteinMinCut(const Graph& g, std::mt19937& rng);
