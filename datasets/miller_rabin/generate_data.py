#!/usr/bin/env python3
"""
File: generate_data.py
Description: Generates datasets for testing the Miller-Rabin primality test.
             Creates files containing small primes, large primes, composites, and Carmichael numbers.
"""

import os       # Used for file system operations (creating directories, paths).
import random   # Used for generating random numbers.
from sympy import isprime, nextprime # Used for ground-truth primality checks.
import math     # Used for mathematical functions.

def ensure_dir(path):
    """
    Ensures that a directory exists.
    
    Args:
        path (str): The directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def is_carmichael(n):
    """
    Checks if a number is a Carmichael number.
    
    A Carmichael number is a composite number n such that b^(n-1) = 1 (mod n)
    for all integers b coprime to n.
    Uses Korselt's criterion: n is square-free and for all prime factors p of n, p-1 divides n-1.
    
    Args:
        n (int): The number to check.
        
    Returns:
        bool: True if n is a Carmichael number, False otherwise.
    """
    if isprime(n) or n < 2:
        return False
    
    # Korselt's criterion: n is square-free and for all prime factors p of n, p-1 divides n-1
    d = 2
    temp = n
    factors = []
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            temp //= d
            if temp % d == 0: # Not square-free
                return False
        else:
            d += 1
    if temp > 1:
        factors.append(temp)
    
    # Carmichael numbers must have at least 3 prime factors
    if len(factors) < 3:
        return False
        
    for p in factors:
        if (n - 1) % (p - 1) != 0:
            return False
    return True

def generate_datasets(out_dir):
    """
    Generates various datasets for Miller-Rabin testing.
    
    Args:
        out_dir (str): The output directory for the datasets.
    """
    ensure_dir(out_dir)
    
    print("Generating Miller-Rabin datasets...")
    
    # 1. All Primes under 50,000
    print("  Generating primes_small.txt...")
    with open(os.path.join(out_dir, "primes_small.txt"), "w") as f:
        for n in range(2, 50000):
            if isprime(n):
                f.write(f"{n}\n")

    # 2. All Composites under 50,000
    print("  Generating composites_small.txt...")
    with open(os.path.join(out_dir, "composites_small.txt"), "w") as f:
        for n in range(4, 50000):
            if not isprime(n):
                f.write(f"{n}\n")

    # 3. Carmichael Numbers under 1,000,000
    print("  Generating carmichael_small.txt...")
    carmichaels = []
    # We can iterate odd numbers. Carmichael numbers are odd.
    for n in range(3, 1000000, 2):
        if is_carmichael(n):
            carmichaels.append(n)
            
    with open(os.path.join(out_dir, "carmichael_small.txt"), "w") as f:
        for n in carmichaels:
            f.write(f"{n}\n")
            
    # 4. Large Primes (for performance benchmarking)
    print("  Generating primes_large.txt...")
    with open(os.path.join(out_dir, "primes_large.txt"), "w") as f:
        # Generate some large primes of different bit sizes
        for bits in [128, 256, 512, 1024]:
            # Generate 10 primes for each bit size
            start = 2**(bits-1)
            p = nextprime(start)
            for _ in range(10):
                f.write(f"{p}\n")
                p = nextprime(p)

    # 5. Large Composites
    print("  Generating composites_large.txt...")
    with open(os.path.join(out_dir, "composites_large.txt"), "w") as f:
        for bits in [128, 256, 512, 1024]:
            for _ in range(10):
                # Product of two primes
                p1 = nextprime(random.getrandbits(bits // 2))
                p2 = nextprime(random.getrandbits(bits // 2))
                f.write(f"{p1 * p2}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="datasets/miller_rabin")
    args = parser.parse_args()
    generate_datasets(args.out_dir)
