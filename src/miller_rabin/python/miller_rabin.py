import random
import sys
import time
import argparse

def power(base, exp, mod):
    return pow(base, exp, mod)

def miller_rabin(n, k, seed=None):
    if seed is not None:
        random.seed(seed)
        
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--k", type=int, default=5)
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
