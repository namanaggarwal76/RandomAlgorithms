#!/usr/bin/env python3
"""
Generate synthetic graph datasets for MinCut benchmarks.
Generates:
1. Random Dense Graphs (Erdos-Renyi)
2. Planted Cut Graphs (Two dense clusters connected by few edges)
3. Cycles (Min cut is always 2)
"""
import argparse
import random
from pathlib import Path
import networkx as nx

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def write_graph(path, G):
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

def gen_random_dense(n, p=0.5):
    return nx.erdos_renyi_graph(n, p)

def gen_planted_cut(n, inter_cluster_edges=5):
    """
    Two clusters of size n/2, connected by 'inter_cluster_edges'.
    Intra-cluster probability is high (0.5).
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
