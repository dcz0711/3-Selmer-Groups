"""
Selmer Group Computation for Elliptic Curves y² + Axy + By = x³

This module computes statistics on Selmer groups by:
1. Generating random (A,B) pairs with specific prime structure
2. Computing cubic residue symbols in Z[ω] (Eisenstein integers)
3. Building matrices whose nullspace gives the Selmer group
4. Aggregating results across many random instances
"""


import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# ---- shared optimized core ----
load("selmer_shared.sage")


from functools import lru_cache
import numpy as np
import random
import json
from collections import defaultdict
from pathlib import Path
import os
import argparse
import psutil
import math



def cube_root_data_uncached(p):
    """
    Return the nontrivial cube roots of unity modulo p.

    For p ≡ 1 (mod 3), the multiplicative group (Z/pZ)^× contains a unique
    subgroup of order 3. We store (z, z^2) where z has exact order 3 mod p.
    """
    #  Find a non-cube a mod p; then z = a^((p-1)/3) is a nontrivial cube root of unity.
    for a in (2, 3, 5, 6, 7, 10, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67):
        z = pow(a % p, (p - 1) // 3, p)
        if z != 1:
            z2 = (z * z) % p
            # canonicalize: choose the smaller representative as "z"
            if z2 < z:
                z, z2 = z2, z
            return (z, z2)

    # Fallback
    g = int(primitive_root(p))
    z = int(pow(g, (p - 1)//3, p))    
    z2 = int((z * z) % p)
    if z2 < z:
        z, z2 = z2, z
    return (z, z2)

def sample_geometric_exponent(p):
    """
    Sample exponent with geometric distribution matching natural density.
    
    The probability that a random integer is divisible by exactly p^k
    (but not p^(k+1)) is (1-1/p)/p^k.
    
    This samples k with that distribution by inverting the CDF.
    
    Args:
        p: Prime base
    
    Returns:
        Random exponent k ≥ 1
    """
    # u = random() is uniform on (0,1)
    # We want P(k) = (1-1/p) * (1/p)^(k-1)
    # CDF: P(X ≤ k) = 1 - (1/p)^k
    # Inverting: k = ceil(log(1-u) / log(1/p)) = ceil(-log(u) / log(p))
    # Adding 1 accounts for k ≥ 1
    return 1 + int(math.log(random.random()) / math.log(1 / p))


def generate_random_a(b_value, height, delta=0.1,  method="height"):
    """
    Generate random A near the "height" H = B^(1/3), coprime to 3.
    
    Three sampling methods:
    - "height": Sample |A| ∈ [(1-δ)H, (1+δ)H] with random sign
    - "box": Sample A ∈ [-(1+δ)H, (1+δ)H]
    - "ignore_height": Sample A ∈ [-B, B]
    
    Args:
        b_value: Value of B (determines height)
        delta: Relative width of sampling interval
        method: Sampling method
    
    Returns:
        Random integer A with gcd(A, 3) = 1
    """        
    if method == "box":
        # Symmetric interval around 0
        low = int(-(1 + delta) * height)
        high = int((1 + delta) * height)
        sign = 1
    else:  
        # "height" method (default)
        # One-sided interval with random sign
        low = int((1 - delta) * height)
        high = int((1 + delta) * height)
        sign = random.choice([1, -1])
        
    # Use random.random() for speed: converts [0,1) to integer range
    result = (low + int((high - low) * random.random())) * sign
        
    # Ensure result is coprime to 3
    if result % 3 == 0:
        result += random.choice([1,-1])
    
    return result



def generate_primes(num_of_primes):
    """
    Generate first num_of_primes primes > 3, split by residue mod 3.
    
    Args:
        num_of_primes: Number of primes to generate
    
    Returns:
        p1: List of primes ≡ 1 (mod 3)
        p2: List of primes ≡ 2 (mod 3)
    """
    return primes_first_n(num_of_primes + 2)[2:]

# ============================================================
# Factor manipulation
# ============================================================

def generate_random_a_b_pair(primes, num_factors, method="height"):
    """
    Generate a random (A,B) pair already in reduced form.
    
    Strategy:
    1. Build B from random primes with geometric exponents
    2. Sample A near height B^(1/3)
    3. Check if this curve is in reduced form (if p | A then not p^3 | B)
    4. Reject and resample if reduction occurs
    
    This is more efficient than generating arbitrary (A,B) and then
    reducing, because we avoid creating highly reducible pairs.
    
    Args:
        primes: Available primes to choose from
        num_factors: Number of distinct prime factors for B
        method: Sampling method for A
    
    Returns:
        (a, b_factorization, b) in reduced form with gcd(a,3) = 1
    """
    # Select random distinct primes for B
    selected_primes = random.sample(primes, num_factors)
    
    # Build B with geometric exponents (mimics natural distribution)
    b_factorization = [(p, sample_geometric_exponent(p)) for p in selected_primes]
    b_value = prod([p^exp for p, exp in b_factorization])
    bad_primes = [p for (p,e) in b_factorization if e >= 3]
    
    height, exact = Integer(b_value).nth_root(3, truncate_mode=True)
    
    if exact:
        # b is a perfect cube, so E admits a 9-isogeny. Reject this b
        return generate_random_a_b_pair(primes, num_factors, method)
    
    height = int(height)
    
    # Rejection sampling: keep trying until we get reduced pair
    while True:
        a = generate_random_a(b_value, height, method=method)
        
        # Test if this (a,b) is already in reduced form
        if any(a % p == 0 for p in bad_primes):
            continue

        return a, b_factorization, b_value



def compute_random_instance(primes, num_factors, height, method="height"):
    """Generate one random curve instance and compute its Selmer matrix.
    """
    # Try a few times in case local conditions force us to resample.
    for _ in range(1000):
        a, b_factorization, b = generate_random_a_b_pair(primes, num_factors, method)
        discriminant = 27 * b - a**3
        nrows, ncols, matrix_str = compute_selmer_matrix(
            B_value=b,
            B_factorization=b_factorization,
            A_value=a,
            discriminant=discriminant,
            cubic_residue_symbol=_CUBIC_RESIDUE_SYMBOL,
            string_matrix=True,
            strip_col=True,
            cutoff=12
        )
        if matrix_str is None: continue
        
        return nrows, ncols, matrix_str
    raise RuntimeError("Failed to sample a valid instance after many attempts.")


# ============================================================
# WORKER PROCESS
# Each worker computes many random instances independently
# ============================================================

def worker_process(args):
    """
    Worker process for parallel computation.
    
    Each worker:
    1. Initializes its own Sage environment (required for multiprocessing)
    2. Sets up LRU caches for expensive computations
    3. Computes num_iterations random instances
    4. Aggregates results by unique matrix configuration
    5. Returns counts to main process
    
    This design ensures:
    - No shared memory between workers (spawn method)
    - Each worker has optimal cache performance
    - Results are aggregated locally before returning
    
    Args:
        args: Tuple of (num_iterations, primes, num_factors,
                       prime_above_precomputed, method, cache_fraction)
    
    Returns:
        List of dicts with {pair, matrix, count} for each unique configuration
    """
    (
        num_iterations,
        primes,
        num_factors,
        prime_above_precomputed,  # Dict of pre-computed values
        method,
        cache_fraction,
        num_processes
    ) = args

    _init_worker_caches(cache_fraction, num_processes)
    cubic_residue_symbol = _CUBIC_RESIDUE_SYMBOL
    
    counts = defaultdict(int)

    for _ in range(num_iterations):
        # Compute Selmer matrix for one random instance
        key = compute_random_instance(primes, num_factors, method)
        counts[key] += 1
    
    # Convert to JSON-serializable format
    results = [
        {
            "pair": [int(num_rows), int(num_cols)],
            "matrix": matrix_str,
            "count": int(count)
        }
        for (num_rows, num_cols, matrix_str), count in counts.items()
    ]

    return results
    
# ============================================================
# Parallel driver
# ============================================================

def compute_prime_above_batch(primes_batch):
    """Worker function to compute prime_above for a batch of primes."""
    return {p: cube_root_data_uncached(p) for p in primes_batch}
    
def compute_single_prime(p):
    """Compute for single prime - minimal data transfer."""
    return (p, cube_root_data_uncached(p))


def run_parallel(primes, num_trials, num_factors, output_file, method="height",
                 batch_size=2000, aggregate_every=10, cache_fraction=0.5, num_processes=None, log_every=100):
    """Parallel computation across workers (fixed2 regimen, optimized core)."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    if num_processes == 1:
        return run_non_parallel(primes, num_trials, num_factors, output_file, method, cache_fraction)

    # ============================================================
    # Parallelize prime_above_cache precomputation
    # ============================================================
    primes_to_cache = [p for p in primes if p % 3 == 1]
    
    if primes_to_cache:
        print(f"Precomputing primitive roots for {len(primes_to_cache)} primes...")
        
        # Use imap for lazy iteration
        prime_above_cache = {}
        with mp.Pool(processes=num_processes) as pool:
            for p, result in pool.imap(compute_single_prime, primes_to_cache, chunksize=1000):
                prime_above_cache[p] = result
                if len(prime_above_cache) % 10000 == 0:
                    print(f"Cached {len(prime_above_cache)} values...")
                    
        print(f"Precomputation complete. Cached {len(prime_above_cache)} values.")
    else:
        prime_above_cache = {}
    
    
    # ============================================================
    # Main parallel processing
    # ============================================================
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "tmp"
    temp_dir.mkdir(exist_ok=True)
    
    # Create batches for main processing
    batches = [
        (min(batch_size, num_trials - i), primes, num_factors, 
         prime_above_cache, method, cache_fraction, num_processes)
        for i in range(0, num_trials, batch_size)
    ]
    
    # Process batches in parallel
    temp_files = []
    num_computed = 0
    computed_increment = batch_size * log_every
    with mp.Pool(processes=num_processes) as pool:
        for batch_id, result in enumerate(pool.imap_unordered(worker_process, batches, chunksize=1)):
            # Save batch result to temporary file
            temp_file = temp_dir / f"{output_path.stem}_batch{batch_id}.tmp"
            temp_files.append(temp_file)
            with open(temp_file, "w") as f:
                for rec in result:
                    f.write(json.dumps(rec) + "\n")
            
            # Periodically aggregate temporary files
            if (batch_id + 1) % aggregate_every == 0:
                aggregate_temp_files(temp_files, output_path)
                temp_files = []
            if batch_id % log_every == 0:
                if num_computed == 0:
                    print(f"[progress] beginning computations")
                else:
                    print(f"[progress] computed {num_computed} trials")
                num_computed += computed_increment
                
    # Final aggregation of remaining files
    if temp_files:
        aggregate_temp_files(temp_files, output_path)
    
    print("Finished all batches.")
    
    

def run_non_parallel(primes, num_trials, num_factors, output_file,
                     method="height", cache_fraction=0.5):
    """Single-process computation."""
     
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)     
    
    _init_worker_caches(cache_fraction)
    cubic_residue_symbol = _CUBIC_RESIDUE_SYMBOL
    
    counts = defaultdict(int)
    
    log_every = 2000
    aggregate_every = 20000
    
    for i in range(0, num_trials, log_every):
        for _ in range(min(log_every, num_trials-i)):
            key = compute_random_instance(
                primes, num_factors, method
                )
            counts[key] += 1
        
        # convert to JSON-friendly format
        result = [
            {"pair": [int(l), int(m)], "matrix": matrix, "count": int(count)}
            for (l, m, matrix), count in counts.items()
        ]
        
        with open(output_file, "w") as f:
            for rec in result:
                f.write(json.dumps(rec) + "\n")
                
        # Periodic aggregation
        if i % aggregate_every == -1:
            aggregate_main_file(out_path)
    
    aggregate_main_file(out_path)
    
    print("Finished all computations.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, required=True)
    parser.add_argument("--primes", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--method", type=str, default="height")
    parser.add_argument("--nprocesses", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    primes = primes_first_n(args.primes + 2)[2:]  # Skip 2,3
    
    if args.debug:   
        random.seed(int(100))
        import cProfile
        import pstats
    
        with cProfile.Profile() as pr:

            run_parallel(
                primes=primes,
                num_trials=args.trials,
                num_factors=args.factors,
                output_file=f"data/output_{args.factors}.jsonl",
                method=args.method,
                batch_size=2000,
                aggregate_every=10,
                cache_fraction=0.5,
                num_processes=args.nprocesses
            )

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    
    else:
        run_parallel(
            primes=primes,
            num_trials=args.trials,
            num_factors=args.factors,
            output_file=f"data/output_{args.factors}.jsonl",
            method=args.method,
            batch_size=2000,
            aggregate_every=10,
            cache_fraction=0.5,
            num_processes=args.nprocesses
        )
        

