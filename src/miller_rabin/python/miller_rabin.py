"""
File: miller_rabin.py
Description: Implementation of the Miller-Rabin primality test.
"""

import random   # Used for generating random bases 'a' for the primality test.
import sys      # Used for system-specific parameters and functions (if needed).
import time     # Used for measuring execution time (if needed).
import argparse # Used for parsing command-line arguments.

def power(base, exp, mod):
    """
    Computes (base^exp) % mod using modular exponentiation.
    
    Args:
        base (int): The base.
        exp (int): The exponent.
        mod (int): The modulus.
        
    Returns:
        int: The result of (base^exp) % mod.
    """
    return pow(base, exp, mod)

def miller_rabin(n, k, seed=None):
    """
    Performs the Miller-Rabin primality test on integer n.
    
    Args:
        n (int): The number to test for primality.
        k (int): The number of iterations (witnesses) to check.
        seed (int, optional): Seed for the random number generator for reproducibility.
        
    Returns:
        tuple: (bool, int)
            - bool: True if n is likely prime, False if n is composite.
            - int: The number of modular exponentiations performed (cost metric).
    """
    if seed is not None:
        random.seed(seed)
        
    if n < 2: return False, 0
    if n == 2 or n == 3: return True, 0
    if n % 2 == 0: return False, 0

    # Write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    modexp_count = 0
    
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = power(a, d, n)
        modexp_count += 1
        
        if x == 1 or x == n - 1:
            continue

        composite = True
        for _ in range(r - 1):
            x = power(x, 2, n)
            modexp_count += 1
            if x == n - 1:
                composite = False
                break
        
        if composite:
            return False, modexp_count
            
    return True, modexp_count

def main():
    """
    Main function to run the Miller-Rabin test from the command line.
    Reads numbers from an input file and writes results to a CSV.
    """
    parser = argparse.ArgumentParser(description="Run Miller-Rabin Primality Test")
    parser.add_argument("--input-file", required=True, help="Path to input file containing numbers to test")
    parser.add_argument("--out-csv", required=True, help="Path to output CSV file")
    parser.add_argument("--k", type=int, default=5, help="Number of iterations for the test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as f:
            numbers = [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)

    # Check if file exists to write header
    write_header = not os.path.exists(args.out_csv)
    
    with open(args.out_csv, 'a') as f:
        if write_header:
            f.write("n,k,is_probable_prime,time_ns,modexp_count\n")
            
        for n in numbers:
            start_time = time.time_ns()
            is_prime, modexp_count = miller_rabin(n, args.k, args.seed)
            end_time = time.time_ns()
            duration = end_time - start_time
            
            f.write(f"{n},{args.k},{1 if is_prime else 0},{duration},{modexp_count}\n")

if __name__ == "__main__":
    import os
    main()
