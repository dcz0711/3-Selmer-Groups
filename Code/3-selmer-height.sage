"""
Selmer Group Computation for Elliptic Curves y² + Axy + By = x³

This module computes statistics on Selmer matrices arising in a 3-isogeny descent
for the family of elliptic curves

    E_{A,B} : y^2 + A x y + B y = x^3.

For each admissible pair (A,B), we build a matrix over F_3 whose nullspace
encodes Selmer information. The driver enumerates many (A,B) subject to various
local conditions, and aggregates counts of identical matrices.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from math import gcd as _gcd

import numpy as np
import psutil

# Ensure consistent multiprocessing behavior across platforms.
mp.set_start_method("spawn", force=True)

# ---- shared optimized core ----
load("selmer_shared.sage")

# =============================================================================
# General use functions
# =============================================================================


def forbidden_prime_product_from_cutoff(min_prime):
    """
    Return the product of all primes < min_prime, together with 3.

    The prime 3 is always excluded explicitly, even if min_prime <= 3.
    """
    bad_primes = [p for p in prime_range(min_prime) if p != 3]
    bad_primes.append(3)
    return prod(bad_primes)

def iter_tasks(
    B_max,
    batch_size_smallB,
    batch_size_largeB,
    min_height,
    max_height,
    forbidden_prime_product,
    cache_fraction,
    num_processes,
    B_start=2,
    split_B=None,
):
    """
    Like your original iter_tasks, but uses two different batch sizes:
      - batch_size_smallB for B < split_B
      - batch_size_largeB for B >= split_B

    Default split_B is min_height^3 (the point where your code switches from
    A_lower = min_height to A_lower = 1).
    """
    if split_B is None:
        split_B = int(min_height) ** 3

    B_end = min(int(B_max), int(max_height) ** 3)  # safety; B_max should already be max_height^3
    B = int(B_start)

    while B <= B_end:
        current_batch_size = batch_size_smallB if B < split_B else batch_size_largeB
        end = min(B + int(current_batch_size), B_end + 1)

        B_values = [x for x in range(B, end) if _gcd(x, forbidden_prime_product) == 1]
        if B_values:
            yield (
                B_values,
                min_height,
                max_height,
                forbidden_prime_product,
                cache_fraction,
                B_values[0],
                B_values[-1],
                num_processes,
            )

        B = end

def count_batches(
    B_max,
    batch_size_smallB,
    batch_size_largeB,
    min_height,
    forbidden_prime_product,
    B_start=2,
    split_B=None,
):
    """
    Count how many *nonempty* batches iter_tasks_two will yield,
    using the same partition of [B_start, B_max].

    A batch is counted if it contains at least one B with gcd(B, forbidden)=1.
    """
    if split_B is None:
        split_B = int(min_height) ** 3

    B_end = int(B_max)
    B = int(B_start)
    n_batches = 0

    while B <= B_end:
        current_batch_size = batch_size_smallB if B < split_B else batch_size_largeB
        end = min(B + int(current_batch_size), B_end + 1)

        # Does this interval contain any admissible B?
        has_any = False
        for x in range(B, end):
            if _gcd(x, forbidden_prime_product) == 1:
                has_any = True
                break

        if has_any:
            n_batches += 1

        B = end

    return n_batches


# =============================================================================
# Fast discriminant screening (avoid bad primes dividing B*(A^3 - 27B))
# =============================================================================

def _passes_bad_prime_sieve_for_sign(discriminant, forbidden_prime_product):
    """
    Return True if both:
    1) the factor A^3 - 27B is not a cube, excluding the exceptional
    additional-isogeny subfamily (and the singular case),
    
    2) the discriminant of the model with A = sign*A_abs is coprime
    to the forbidden prime product, assuming gcd(B_value, forbidden_prime_product)=1.
    """
    return _gcd(discriminant, forbidden_prime_product) == 1 and (not is_perfect_cube(discriminant))


def _iterate_A_abs_with_cubes(A_start, A_stop):
    """
    Iterate over A_abs in [A_start, A_stop] yielding (A_abs, A_abs^3),
    updating cubes in O(1) per step.

    Uses:
        (A+1)^2 = A^2 + 2A + 1
        (A+1)^3 = A^3 + 3A^2 + 3A + 1
    """
    A = int(A_start)
    A2 = A * A
    A3 = A2 * A

    for _ in range(int(A_start), int(A_stop) + 1):
        yield A, A3
        # update from A to A+1
        A3 += 3 * A2 + 3 * A + 1
        A2 += 2 * A + 1
        A += 1


# =============================================================================
# Worker
# =============================================================================

def worker_process(task_args):
    """
    Worker entry point.

    Each worker processes a batch of B values independently and returns aggregated
    counts keyed by Selmer matrix signatures.

    Parameters
    ----------
    task_args : tuple
        (B_values, min_height, max_height, forbidden_prime_product, cache_fraction, 
        B_min, B_max, num_processes)

    Returns
    -------
    list[dict]
        JSON-serializable records with keys:
            "pair": [num_rows, num_cols],
            "matrix": matrix_string,
            "count": multiplicity.
    """
    (
        B_values,
        min_height,
        max_height,
        forbidden_prime_product,
        cache_fraction,
        B_min,
        B_max,
        num_processes,
    ) = task_args

    # A worker may process many tasks.  Keep its LRU caches warm between them.
    if _CUBIC_RESIDUE_SYMBOL is None:
        _init_worker_caches(cache_fraction, num_processes)
    cubic_residue_symbol = _CUBIC_RESIDUE_SYMBOL

    counts = defaultdict(int)

    for B_value in B_values:
        # Heuristic lower bound for |A| based on cube root of B (as in the original code).
        r, exact = Integer(B_value).nth_root(3, truncate_mode=True)
        
        # If B is a perfect cube, then y^2 + Axy + By = x^3 admits a 9-isogeny
        if exact:
            continue
            
        cube_root_B = int(r) + 1
        
        # B_values were prefiltered so gcd(B_value, forbidden_prime_product)=1 holds.
        pari_B_factorization = list(pari(B_value).factor())
        B_factorization = list(zip(pari_B_factorization[0], pari_B_factorization[1]))#list(factor(B_value))
        
        A_lower = 1 if cube_root_B > min_height else min_height
        bad_primes = [p for (p,e) in B_factorization if e >= 3]
        for A_abs, A_abs_cubed in _iterate_A_abs_with_cubes(A_lower, max_height):
            
            # Check we haven't already seen this curve in a different form. 
            # We can always replace A with A/p and B with B/p^3 if both are integers
            if any(A_abs % p == 0 for p in bad_primes):
                continue

            # Ensure elliptic curve has good reduction at forbidden primes
            for sign in (1, -1):
                discriminant = sign * A_abs_cubed - 27*B_value
                if not _passes_bad_prime_sieve_for_sign(
                    discriminant=discriminant,
                    forbidden_prime_product=forbidden_prime_product
                ):
                    continue

                A_value = sign * A_abs
                
                key = compute_selmer_matrix(B_value, B_factorization, A_value, discriminant, cubic_residue_symbol)
                counts[key] += 1

    return B_min, B_max, [
        {"pair": [int(r), int(c)], "matrix": m, "count": int(k)}
        for (r, c, m), k in counts.items()
    ]
    
# =============================================================================
# Parallel driver
# =============================================================================

def run_parallel(min_height, max_height, forbidden_prime_product, output_file,
                 batch_size_smallB=1000, batch_size_largeB=10, aggregate_every=5, log_every=1,
                 cache_fraction=0.5, num_processes=None, overwrite=False):
    """
    Run the computation in parallel using a process pool.

    This driver partitions the range 2 <= B <= max_height^3 into contiguous
    batches and dispatches one batch per task.

    Notes
    -----
    - B values are prefiltered so that gcd(B, forbidden_prime_product)=1.
      The remaining discriminant factor gcd(A^3 - 27B, forbidden)=1 
      is enforced inside the worker (because it depends on A).
    - Per-process arithmetic caches are initialized lazily inside worker_process
      via _init_worker_caches(cache_fraction), so the pool need not use an
      initializer.
    """
    min_height = int(min_height)
    max_height = int(max_height)
    forbidden_prime_product = int(forbidden_prime_product)
    batch_size_smallB = int(batch_size_smallB)
    batch_size_largeB = int(batch_size_largeB)
    aggregate_every = int(aggregate_every)
    log_every = int(log_every)

    if num_processes is None:
        num_processes = mp.cpu_count()
    num_processes = int(num_processes)

    if min_height < 1:
        raise ValueError("min_height must be at least 1")
    if max_height < min_height:
        raise ValueError("max_height must be at least min_height")
    if forbidden_prime_product < 1:
        raise ValueError("forbidden_prime_product must be positive")
    if batch_size_smallB < 1 or batch_size_largeB < 1:
        raise ValueError("batch sizes must be at least 1")
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
            min_height=min_height,
            max_height=max_height,
            forbidden_prime_product=forbidden_prime_product,
            output_file=output_file,
            cache_fraction=cache_fraction,
            log_every=log_every,
            overwrite=overwrite,
        )

    output_path = prepare_output_file(output_file, overwrite=overwrite)

    B_max = max_height ** 3

    total_batches = count_batches(
        B_max=B_max,
        batch_size_smallB=batch_size_smallB,  
        batch_size_largeB=batch_size_largeB,
        forbidden_prime_product=forbidden_prime_product,
        min_height=min_height
    )

    print(f"[info] total nonempty batches: {total_batches}", flush=True)

    if total_batches == 0:
        write_jsonl_atomic([], output_path)
        print("Finished all batches.", flush=True)
        return

    # Construct tasks: each task is one batch of admissible B values.
    task_iter = iter_tasks(
    B_max=B_max,
    batch_size_smallB=batch_size_smallB,   # e.g. small B: bigger batches
    batch_size_largeB=batch_size_largeB,     # e.g. large B: smaller batches
    min_height=min_height,
    max_height=max_height,
    forbidden_prime_product=forbidden_prime_product,
    cache_fraction=cache_fraction,
    num_processes=num_processes,
    B_start=2
    )


    temp_dir = output_path.parent / "tmp"
    temp_dir.mkdir(exist_ok=True)

    temp_files = []
    
    with mp.Pool(processes=num_processes) as pool:
        for task_id, (B_min, B_max_batch, result) in enumerate(
                pool.imap_unordered(worker_process, task_iter, chunksize=8),
                start=1):
            temp_path = temp_dir / f"{output_path.stem}_task{task_id}.tmp"
            temp_files.append(temp_path)

            with open(temp_path, "w") as f:
                for rec in result:
                    f.write(json.dumps(rec) + "\n")

            if task_id % aggregate_every == 0:
                aggregate_temp_files(temp_files, output_path)
                temp_files = []

            if task_id % log_every == 0 or task_id == total_batches:
                print(
                    f"[progress] completed {task_id}/{total_batches} batches"
                    f" (B in [{B_min}, {B_max_batch}])",
                    flush=True,
                )
                

    if temp_files:
        aggregate_temp_files(temp_files, output_path)

    print("Finished all batches.", flush=True)

# =============================================================================
# Serial path
# =============================================================================

def run_non_parallel(min_height, max_height, forbidden_prime_product, output_file,
                     cache_fraction=0.5, log_every=1, overwrite=False):
    """
    Serial implementation used for debugging and profiling.

    This shares the same per-process caching initialization and the same
    mathematical filters as the parallel path.
    """
    min_height = int(min_height)
    max_height = int(max_height)
    forbidden_prime_product = int(forbidden_prime_product)
    log_every = int(log_every)
    if min_height < 1:
        raise ValueError("min_height must be at least 1")
    if max_height < min_height:
        raise ValueError("max_height must be at least min_height")
    if forbidden_prime_product < 1:
        raise ValueError("forbidden_prime_product must be positive")
    if log_every < 1:
        raise ValueError("log_every must be at least 1")

    if _CUBIC_RESIDUE_SYMBOL is None:
        _init_worker_caches(cache_fraction)
    cubic_residue_symbol = _CUBIC_RESIDUE_SYMBOL

    output_path = prepare_output_file(output_file, overwrite=overwrite)

    counts = defaultdict(int)
    B_max = max_height ** 3
    block_size = 1000
    total_blocks = max(0, (B_max - 1 + block_size - 1) // block_size)

    if total_blocks == 0:
        write_jsonl_atomic([], output_path)
        print("Finished all computations.", flush=True)
        return

    for block_id, start in enumerate(
            range(2, B_max + 1, block_size), start=1):
        end = min(start + block_size, B_max + 1)
        B_values = [B for B in range(start, end) if _gcd(B, forbidden_prime_product) == 1]
        if not B_values:
            continue
            
        for B_value in B_values:
            # Heuristic lower bound for |A| based on cube root of B (as in the original code).
            r, exact = Integer(B_value).nth_root(3, truncate_mode=True)
        
            # If B is a perfect cube, then y^2 + Axy + By = x^3 admits a 9-isogeny
            if exact:
                continue
            
            pari_B_factorization = list(pari(B_value).factor())
            B_factorization = list(zip(pari_B_factorization[0], pari_B_factorization[1]))#list(factor(B_value))
            
            cube_root_B = int(r) + 1
            
            A_lower = 1 if cube_root_B > min_height else min_height
            bad_primes = [p for (p,e) in B_factorization if e >= 3]

            for A_abs, A_abs_cubed in _iterate_A_abs_with_cubes(A_lower, max_height):
                # Check we haven't already seen this curve in a different form. 
                # We can always replace A with A/p and B with B/p^3 if both are integers
                if any(A_abs % p == 0 for p in bad_primes):
                    continue

                for sign in (1, -1):
                    discriminant = sign * A_abs_cubed - 27*B_value
                    if not _passes_bad_prime_sieve_for_sign(
                        discriminant=discriminant,
                        forbidden_prime_product=forbidden_prime_product
                    ):
                        continue

                    A_value = sign * A_abs
            
                    key = compute_selmer_matrix(B_value, B_factorization, A_value, discriminant, cubic_residue_symbol)
                    counts[key] += 1

        # Write an intermediate snapshot.
        snapshot = [
            {"pair": [int(r), int(c)], "matrix": m, "count": int(k)}
            for (r, c, m), k in counts.items()
        ]
        write_jsonl_atomic(snapshot, output_path)

        if block_id % log_every == 0 or block_id == total_blocks:
            print(
                f"[progress] completed {block_id}/{total_blocks} serial blocks "
                f"(B through {end - 1})",
                flush=True,
            )

    if not output_path.exists():
        write_jsonl_atomic([], output_path)
    aggregate_main_file(output_path)
    print("Finished all computations.", flush=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_height", type=int, default=1)
    parser.add_argument("--max_height", type=int, required=True)
    parser.add_argument("--nprocesses", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress after this many completed batches.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--min_prime",
        type=int,
        default=1,
        help="Exclude all primes < min_prime (3 is always excluded).",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # The experiment only includes curves with good reduction at 3 
    # and all primes < min_prime
    forbidden_prime_product = forbidden_prime_product_from_cutoff(args.min_prime)

    output_file = args.output or (
        f"data/height_H{args.min_height}-{args.max_height}_"
        f"p{args.min_prime}.jsonl"
    )

    batch_size_smallB=10000
    batch_size_largeB=100
    log_every = args.log_every
    
    if args.debug:
        import cProfile
        import pstats

        with cProfile.Profile() as profiler:
            run_parallel(
                min_height=args.min_height,
                max_height=args.max_height,
                forbidden_prime_product=forbidden_prime_product,
                output_file=output_file,
                batch_size_smallB=batch_size_smallB, 
                batch_size_largeB=batch_size_largeB,
                aggregate_every=10,
                cache_fraction=0.5,
                log_every=log_every,
                num_processes=args.nprocesses,
                overwrite=args.overwrite,
            )

        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    else:
        run_parallel(
            min_height=args.min_height,
            max_height=args.max_height,
            forbidden_prime_product=forbidden_prime_product,
            output_file=output_file,
            batch_size_smallB=batch_size_smallB, 
            batch_size_largeB=batch_size_largeB,
            aggregate_every=10,
            cache_fraction=0.5,
            log_every=log_every,
            num_processes=args.nprocesses,
            overwrite=args.overwrite,
        )
