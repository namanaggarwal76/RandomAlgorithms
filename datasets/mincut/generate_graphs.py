#!/usr/bin/env python3
"""
File: generate_graphs.py
Description: Generate synthetic graph datasets for MinCut benchmarks.
Generates:
1. Random Dense Graphs (Erdos-Renyi)
2. Planted Cut Graphs (Two dense clusters connected by few edges)
3. Cycles (Min cut is always 2)
"""
import argparse     # Used for parsing command-line arguments.
import random       # Used for random number generation.
from pathlib import Path # Used for filesystem path operations.
import networkx as nx # Used for graph generation algorithms.

def ensure_dir(path):
    """
    Ensures that a directory exists.
    
    Args:
        path (Path): The directory path.
    """
    path.mkdir(parents=True, exist_ok=True)

def write_graph(path, G):
    """
    Writes a graph to a file as an edge list.
    
    Args:
        path (Path): The output file path.
        G (nx.Graph): The NetworkX graph object.
    """
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

def gen_random_dense(n, p=0.5):
    """
    Generates a random Erdos-Renyi graph G(n, p).
    
    Args:
        n (int): Number of nodes.
        p (float): Probability of edge creation.
        
    Returns:
        nx.Graph: The generated graph.
    """
    return nx.erdos_renyi_graph(n, p)

def gen_planted_cut(n, inter_cluster_edges=5):
    """
    Generates a graph with a planted minimum cut.
    
    Creates two dense clusters of size n/2, connected by a small number
    of 'inter_cluster_edges'. The min-cut should be exactly these edges.
    Intra-cluster probability is high (0.5).
    
    Args:
        n (int): Total number of nodes.
        inter_cluster_edges (int): Number of edges connecting the two clusters.
        
    Returns:
        nx.Graph: The generated graph.
    """
    n1 = n // 2
    n2 = n - n1
    G1 = nx.erdos_renyi_graph(n1, 0.5)
    G2 = nx.erdos_renyi_graph(n2, 0.5)
    
    # Relabel G2 nodes to not conflict with G1
    mapping = {i: i + n1 for i in range(n2)}
    G2 = nx.relabel_nodes(G2, mapping)
    
    G = nx.compose(G1, G2)
    
    # Add bridging edges
    for _ in range(inter_cluster_edges):
        u = random.randint(0, n1 - 1)
        v = random.randint(n1, n - 1)
        G.add_edge(u, v)
        
    return G

def gen_cycle(n):
    """
    Generates a cycle graph C_n.
    The min-cut of a cycle graph is always 2.
    
    Args:
        n (int): Number of nodes.
        
    Returns:
        nx.Graph: The generated cycle graph.
    """
    return nx.cycle_graph(n)

def main():
    parser = argparse.ArgumentParser(description="Generate MinCut datasets.")
    parser.add_argument("--out-dir", default="datasets/mincut", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    
    # 1. Random Graphs (Erdos-Renyi) - Dense range for good plots
    print("Generating Random Graphs...")
    # Generating a dense range of sizes for smooth plotting
    sizes = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
    for n in sizes:
        # p decreases slightly as n grows to keep edge count reasonable for laptop
        p = max(0.1, 5.0 / n) 
        G = gen_random_dense(n, p=p)
        write_graph(out_dir / f"random_{n}.txt", G)

    # 2. Planted Cut (The "interesting" case for MinCut)
    print("Generating Planted Cut Graphs...")
    for n in sizes:
        G = gen_planted_cut(n, inter_cluster_edges=3)
        write_graph(out_dir / f"planted_{n}.txt", G)

    # 3. Cycles (Min cut = 2)
    print("Generating Cycle Graphs...")
    for n in sizes:
        G = gen_cycle(n)
        write_graph(out_dir / f"cycle_{n}.txt", G)

    print(f"Datasets generated in {out_dir}")

if __name__ == "__main__":
    main()
