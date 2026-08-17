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


_DEFAULT_DELTA = QQ("1/10")


def cube_root_data_uncached(p):
    """
    Return the nontrivial cube roots of unity modulo p.

    For p ≡ 1 (mod 3), the multiplicative group (Z/pZ)^× contains a unique
    subgroup of order 3. We store (z, z^2) where z has exact order 3 mod p.
    """
    p = int(p)
    if p % 3 != 1:
        raise ValueError("cube_root_data_uncached requires p congruent to 1 mod 3")

    # Find a non-cube a mod p; then z = a^((p-1)/3) is a nontrivial cube root of unity.
    for a in (2, 3, 5, 6, 7, 10, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67):
        if a % p == 0:
            continue
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
    
    Conditional on p dividing a random integer, the probability that its
    p-adic valuation is k is

        P(k) = (1 - 1/p) / p^(k-1),  k >= 1.
    
    Args:
        p: Prime base
    
    Returns:
        Random exponent k ≥ 1
    """
    # Exact geometric sampling avoids both log(0) and floating-point bias.
    p = int(p)
    exponent = 1
    while random.randrange(p) == 0:
        exponent += 1
    return exponent


def generate_random_a(b_value, height, delta=None, method="height"):
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
    # Use Sage rationals throughout.  Python's ``fractions.Fraction`` is not
    # interoperable with preparsed Sage Integer literals under Python 3.13
    # (for example, ``1 - Fraction(...)`` raises a TypeError).  Keeping the
    # default precomputed also avoids reparsing 1/10 millions of times.
    if delta is None:
        delta = _DEFAULT_DELTA
    else:
        delta = QQ(str(delta))
        if not 0 <= delta < 1:
            raise ValueError("delta must lie in [0, 1)")

    if method == "box":
        # Symmetric interval around 0
        low = int(-(1 + delta) * height)
        high = int((1 + delta) * height)
    elif method == "height":
        # "height" method (default)
        # One-sided interval with random sign
        low = int((1 - delta) * height)
        high = int((1 + delta) * height)
    elif method == "ignore_height":
        low = -int(b_value)
        high = int(b_value) + 1
    else:
        raise ValueError(
            "method must be one of 'height', 'box', or 'ignore_height'"
        )

    if high <= low:
        raise ValueError("The requested A-sampling interval is empty")
    if high - low == 1 and low % 3 == 0:
        raise ValueError(
            "The requested A-sampling interval contains no integer coprime to 3"
        )

    # Rejection sampling gives the uniform distribution on the integers in the
    # requested interval that are coprime to 3.  random.randrange works with
    # arbitrary-size integers, unlike the previous random.random conversion.
    while True:
        result = random.randrange(low, high)
        if method == "height":
            result *= random.choice([1, -1])
        if result % 3 != 0:
            return result



def generate_primes(num_of_primes):
    """
    Generate the first num_of_primes primes greater than 3.
    
    Args:
        num_of_primes: Number of primes to generate
    
    Returns:
        List of primes greater than 3.
    """
    return [int(p) for p in primes_first_n(num_of_primes + 2)[2:]]

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
    # Rejection-sample B without recursion.  Although a cubic B is rare, an
    # iterative loop cannot exhaust Python's recursion limit.
    while True:
        selected_primes = random.sample(primes, num_factors)

        # Build B with geometric exponents (mimics natural distribution)
        b_factorization = sorted(
            [(p, sample_geometric_exponent(p)) for p in selected_primes],
            key=lambda item: item[0],
        )
        b_value = prod([p^exp for p, exp in b_factorization])
        bad_primes = [p for (p,e) in b_factorization if e >= 3]

        height, exact = Integer(b_value).nth_root(3, truncate_mode=True)
        if not exact:
            break
        # B is a perfect cube, so E admits a 9-isogeny; resample B.
    
    height = int(height)
    
    # Rejection sampling: keep trying until we get reduced pair
    while True:
        a = generate_random_a(b_value, height, method=method)
        
        # Test if this (a,b) is already in reduced form
        if any(a % p == 0 for p in bad_primes):
            continue

        return a, b_factorization, b_value



def compute_random_instance(primes, num_factors, method="height"):
    """Generate one random curve instance and compute its Selmer matrix.
    """
    # Try a few times in case local conditions force us to resample.
    for _ in range(1000):
        a, b_factorization, b = generate_random_a_b_pair(primes, num_factors, method)
        discriminant = a**3 - 27*b

        # Exclude the second exceptional isogeny subfamily, as specified in
        # the experiment.  This also rejects the singular case discriminant=0.
        if is_perfect_cube(discriminant):
            continue

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
        if matrix_str is None:
            continue
        
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

    # A process may receive several batches.  Preserve its LRU caches instead
    # of rebuilding and emptying them at the beginning of every batch.
    if _CUBIC_RESIDUE_SYMBOL is None:
        _init_worker_caches(
            cache_fraction,
            num_processes,
            precomputed_root_data=prime_above_precomputed,
        )
    
    counts = defaultdict(int)

    for _ in range(num_iterations):
        # Compute Selmer matrix for one random instance
        key = compute_random_instance(primes, num_factors, method=method)
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
                 batch_size=2000, aggregate_every=10, cache_fraction=0.5,
                 num_processes=None, log_every=1, overwrite=False):
    """Parallel computation across workers (fixed2 regimen, optimized core)."""
    primes = [int(p) for p in primes]
    num_trials = int(num_trials)
    num_factors = int(num_factors)
    batch_size = int(batch_size)
    aggregate_every = int(aggregate_every)
    log_every = int(log_every)

    if num_processes is None:
        num_processes = mp.cpu_count()
    num_processes = int(num_processes)

    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")
    if method not in ("height", "box", "ignore_height"):
        raise ValueError("unknown A-sampling method")
    if not 1 <= num_factors <= len(primes):
        raise ValueError("num_factors must lie between 1 and the number of primes")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if aggregate_every < 1:
        raise ValueError("aggregate_every must be at least 1")
    if log_every < 1:
        raise ValueError("log_every must be at least 1")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    if not 0 < float(cache_fraction) <= 1:
        raise ValueError("cache_fraction must lie in (0, 1]")

    if num_processes == 1:
        return run_non_parallel(
            primes, num_trials, num_factors, output_file, method,
            cache_fraction, log_every=log_every, overwrite=overwrite,
        )

    output_path = prepare_output_file(output_file, overwrite=overwrite)

    # ============================================================
    # Parallelize prime_above_cache precomputation
    # ============================================================
    primes_to_cache = [p for p in primes if p % 3 == 1]
    
    if primes_to_cache:
        print(
            f"Precomputing primitive roots for {len(primes_to_cache)} primes...",
            flush=True,
        )
        
        # Use imap for lazy iteration
        prime_above_cache = {}
        precompute_chunksize = max(
            1, len(primes_to_cache) // (4 * num_processes)
        )
        with mp.Pool(processes=num_processes) as pool:
            for p, result in pool.imap(
                    compute_single_prime,
                    primes_to_cache,
                    chunksize=precompute_chunksize):
                prime_above_cache[p] = result
                if len(prime_above_cache) % 10000 == 0:
                    print(f"Cached {len(prime_above_cache)} values...", flush=True)
                    
        print(
            f"Precomputation complete. Cached {len(prime_above_cache)} values.",
            flush=True,
        )
    else:
        prime_above_cache = {}
    
    
    # ============================================================
    # Main parallel processing
    # ============================================================
    temp_dir = output_path.parent / "tmp"
    temp_dir.mkdir(exist_ok=True)
    
    # Create batches for main processing
    batches = [
        (min(batch_size, num_trials - i), primes, num_factors, 
         prime_above_cache, method, cache_fraction, num_processes)
        for i in range(0, num_trials, batch_size)
    ]
    total_batches = len(batches)
    print(
        f"[info] computing {num_trials} accepted trials in {total_batches} "
        f"batches with {num_processes} processes",
        flush=True,
    )
    
    # Process batches in parallel
    temp_files = []
    num_computed = 0
    with mp.Pool(processes=num_processes) as pool:
        for batch_id, result in enumerate(pool.imap_unordered(worker_process, batches, chunksize=1)):
            num_computed += sum(rec["count"] for rec in result)
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
            if (batch_id + 1) % log_every == 0 or batch_id + 1 == total_batches:
                print(
                    f"[progress] completed {batch_id + 1}/{total_batches} batches; "
                    f"computed {num_computed}/{num_trials} accepted trials",
                    flush=True,
                )
                
    # Final aggregation of remaining files
    if temp_files:
        aggregate_temp_files(temp_files, output_path)
    
    if num_computed != num_trials:
        raise RuntimeError(
            f"Internal count mismatch: computed {num_computed} of {num_trials} trials"
        )

    print("Finished all batches.", flush=True)
    
    

def run_non_parallel(primes, num_trials, num_factors, output_file,
                     method="height", cache_fraction=0.5, log_every=1,
                     overwrite=False):
    """Single-process computation."""
    primes = [int(p) for p in primes]
    num_trials = int(num_trials)
    num_factors = int(num_factors)
    log_every = int(log_every)
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")
    if method not in ("height", "box", "ignore_height"):
        raise ValueError("unknown A-sampling method")
    if not 1 <= num_factors <= len(primes):
        raise ValueError("num_factors must lie between 1 and the number of primes")
    if log_every < 1:
        raise ValueError("log_every must be at least 1")
    if not 0 < float(cache_fraction) <= 1:
        raise ValueError("cache_fraction must lie in (0, 1]")
     
    out_path = prepare_output_file(output_file, overwrite=overwrite)
    
    if _CUBIC_RESIDUE_SYMBOL is None:
        _init_worker_caches(cache_fraction)
    
    counts = defaultdict(int)
    
    serial_batch_size = 2000
    total_batches = (num_trials + serial_batch_size - 1) // serial_batch_size
    num_computed = 0
    print(
        f"[info] computing {num_trials} accepted trials in {total_batches} "
        "serial batches",
        flush=True,
    )
    
    for batch_id, i in enumerate(
            range(0, num_trials, serial_batch_size), start=1):
        current_batch_size = min(serial_batch_size, num_trials - i)
        for _ in range(current_batch_size):
            key = compute_random_instance(
                primes, num_factors, method=method
            )
            counts[key] += 1
        num_computed += current_batch_size
        
        # convert to JSON-friendly format
        result = [
            {"pair": [int(l), int(m)], "matrix": matrix, "count": int(count)}
            for (l, m, matrix), count in counts.items()
        ]
        
        write_jsonl_atomic(result, out_path)

        if batch_id % log_every == 0 or batch_id == total_batches:
            print(
                f"[progress] completed {batch_id}/{total_batches} batches; "
                f"computed {num_computed}/{num_trials} accepted trials",
                flush=True,
            )
                
    aggregate_main_file(out_path)
    
    print("Finished all computations.", flush=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, required=True)
    parser.add_argument("--primes", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=("height", "box", "ignore_height"),
        default="height",
    )
    parser.add_argument("--nprocesses", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress after this many completed batches.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    primes = generate_primes(args.primes)
    
    output_file = args.output or (
        f"data/factor_N{args.primes}_n{args.factors}.jsonl"
    )

    if args.debug:   
        random.seed(int(100))
        import cProfile
        import pstats
    
        with cProfile.Profile() as pr:

            run_parallel(
                primes=primes,
                num_trials=args.trials,
                num_factors=args.factors,
                output_file=output_file,
                method=args.method,
                batch_size=2000,
                aggregate_every=10,
                cache_fraction=0.5,
                num_processes=args.nprocesses,
                log_every=args.log_every,
                overwrite=args.overwrite,
            )

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    
    else:
        run_parallel(
            primes=primes,
            num_trials=args.trials,
            num_factors=args.factors,
            output_file=output_file,
            method=args.method,
            batch_size=2000,
            aggregate_every=10,
            cache_fraction=0.5,
            num_processes=args.nprocesses,
            log_every=args.log_every,
            overwrite=args.overwrite,
        )
        
