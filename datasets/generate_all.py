import argparse
import random
import os
from pathlib import Path
import math

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

# --- QuickSort Generators ---

def gen_qsort_uniform(n):
    return [random.randint(-1000000, 1000000) for _ in range(n)]

def gen_qsort_sorted(n):
    return list(range(n))

def gen_qsort_reverse_sorted(n):
    return list(range(n, 0, -1))

def gen_qsort_duplicates(n):
    unique = [random.randint(-1000, 1000) for _ in range(10)]
    return [random.choice(unique) for _ in range(n)]

def generate_qsort_datasets(out_dir):
    print("Generating QuickSort datasets...")
    sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    categories = {
        "random": gen_qsort_uniform,
        "sorted": gen_qsort_sorted,
        "reverse_sorted": gen_qsort_reverse_sorted,
        "duplicates": gen_qsort_duplicates
    }
    
    base_dir = out_dir / "qsort"
    
    for cat, func in categories.items():
        cat_dir = base_dir / cat
        ensure_dir(cat_dir)
        for size in sizes:
            filepath = cat_dir / f"{cat}_{size}.txt"
            if filepath.exists():
                continue
            print(f"  Generating {filepath}...")
            data = func(size)
            with open(filepath, "w") as f:
                for num in data:
                    f.write(f"{num}\n")

# --- Frievald Generators ---

def generate_frievald_datasets(out_dir):
    print("Generating Frievald datasets...")
    # We generate pairs of matrices A and B, and store them in a file.
    # Format:
    # n
    # A_row_1
    # ...
    # A_row_n
    # B_row_1
    # ...
    # B_row_n
    
    sizes = [100, 200, 400, 800, 1200, 1500] # Sizes for benchmarking
    base_dir = out_dir / "frievald"
    ensure_dir(base_dir)
    
    for n in sizes:
        filepath = base_dir / f"matrix_{n}.txt"
        if filepath.exists():
            continue
        print(f"  Generating {filepath}...")
        with open(filepath, "w") as f:
            f.write(f"{n}\n")
            # Generate A
            for _ in range(n):
                row = [str(random.randint(0, 10)) for _ in range(n)]
                f.write(" ".join(row) + "\n")
            # Generate B
            for _ in range(n):
                row = [str(random.randint(0, 10)) for _ in range(n)]
                f.write(" ".join(row) + "\n")

# --- Miller-Rabin Generators ---

from sympy import isprime

def is_prime_trial(n):
    return isprime(n)

def generate_mr_datasets(out_dir):
    print("Generating Miller-Rabin datasets...")
    base_dir = out_dir / "miller_rabin"
    ensure_dir(base_dir)
    
    # 1. Primes (various bit sizes)
    # 2. Composites (various bit sizes)
    # 3. Carmichael numbers (hardcoded small list + generated if possible, but hardcoded is safer for correctness)
    
    bit_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    # Primes
    primes_file = base_dir / "primes.txt"
    if not primes_file.exists():
        print(f"  Generating {primes_file}...")
        with open(primes_file, "w") as f:
            for bits in bit_sizes:
                count = 0
                while count < 20:
                    # Generate random odd number
                    lower = 1 << (bits - 1)
                    upper = (1 << bits) - 1
                    num = random.randint(lower, upper) | 1
                    if is_prime_trial(num):
                        f.write(f"{num}\n")
                        count += 1

    # Composites
    composites_file = base_dir / "composites.txt"
    if not composites_file.exists():
        print(f"  Generating {composites_file}...")
        with open(composites_file, "w") as f:
            for bits in bit_sizes:
                count = 0
                while count < 20:
                    lower = 1 << (bits - 1)
                    upper = (1 << bits) - 1
                    num = random.randint(lower, upper) | 1
                    if not is_prime_trial(num):
                        f.write(f"{num}\n")
                        count += 1

    # Carmichael Numbers (small subset fitting in 64-bit)
    carmichael_file = base_dir / "carmichael.txt"
    if not carmichael_file.exists():
        print(f"  Generating {carmichael_file}...")
        # Known Carmichael numbers
        carmichaels = [
            561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341, 41041, 46657, 52633, 62745, 63973, 75361, 101101, 115921, 126217, 162401, 172081, 188461, 252601, 278545, 294409, 314821, 334153, 340561, 399001, 410041, 449065, 488881, 512461
        ]
        with open(carmichael_file, "w") as f:
            for num in carmichaels:
                f.write(f"{num}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate all datasets")
    parser.add_argument("--out-dir", type=str, default="datasets", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir).resolve()
    
    generate_qsort_datasets(out_dir)
    generate_frievald_datasets(out_dir)
    generate_mr_datasets(out_dir)
    print("All datasets generated.")

if __name__ == "__main__":
    main()
