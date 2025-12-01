#!/usr/bin/env python3
import os
import random
from sympy import isprime, nextprime
import math

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_carmichael(n):
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
