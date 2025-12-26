from functools import cache
from itertools import product
import numpy as np
import random
import json
from sympy import nextprime
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
from pathlib import Path
import os
import time
import psutil  
from collections import defaultdict
import argparse

# -------------------------
# Algebraic setup
# -------------------------

# Working in cyclotomic field Q(ω) where ω is a primitive 3rd root of unity
K.<ω> = CyclotomicField(3)
O = K.ring_of_integers()

# -------------------------
# Core functions
# -------------------------

def residue_map(π):
    π = O(π)
    P = O.fractional_ideal(π)
    k = K.residue_field(P)
    red = k.reduction_map()
    w = red(ω)
    e = (π.norm() - 1) // 3
    one = k(1)
    return red, w, e, one

def prime_above(p):
    """
    Return a prime above a rational prime p in O.
    The output is (a, b) such that a + bω is a generator of the ideal pi above p
    """
    if p % 3 == 2:
        return (p, 0)
    factor = O(p).factor()[0][0]
    return (int(factor[0]), int(factor[1]))

def get_prime_above(p, prime_above_cache, PRIME_MAX):
    """
      Retrieve prime above p from cache or compute if not present.
      Caches up to PRIME_MAX entries.
     """
    if p % 3 == 2:
        return (p, 0)
        
    if p in prime_above_cache:
        return prime_above_cache[p]
        
    value = prime_above(p)
    if len(prime_above_cache) < PRIME_MAX:
        prime_above_cache[p] = value
        
    return value

def cubic_residue(a, p, cr_cache, prime_above_cache, CR_MAX, PRIME_MAX):
    """
        Compute the cubic residue of a modulo p in O.
        Uses caches to speed up repeated queries.
        Returns 0,1,2 based on cubic residue class.
    """
    key = (int(a), int(p))
    if (res := cr_cache.get(key)) is not None:
        return res
        
    a = O(a)
    x, y = get_prime_above(p, prime_above_cache, PRIME_MAX)
    π = O(x + ω*y)
    
    red, w, e, one = residue_map(π)
    
    res = 0 if red(a) == 0 else (
        0 if (r := red(a) ** e) == one else
        1 if r == w else
        2
    )
    
    if len(cr_cache) < CR_MAX:
        cr_cache[key] = int(res)
    return int(res)

def clean_fac(a, b_fac):
    """
        Adjust the factorization of B relative to A.
        If a prime p^3 divides B and p divides A, reduce powers in B accordingly.
        Returns updated (A, B factorization, B).
    """
    new_b_fac = []
    new_b = 1
    new_a = a
    for p, exp in b_fac:
        if exp < 3 or a % p != 0:
            new_b_fac.append([p, exp])
            new_b *= p ** exp
            continue
        v1 = valuation(a, p)
        times = min(v1, exp // 3)
        new_a /= p ** times
        new_exp = exp - (3 * times)
        if new_exp > 0:
            new_b_fac.append([p, new_exp])
            new_b *= p ** new_exp
    return a, new_b_fac, new_b
    
def trim_matrix(M):
    """Remove the last column of a numpy matrix"""
    return np.delete(M, -1, axis=1)
    
def build_matrix(primes_to_check, primes_to_use, m, l, t, B, prime_above_cache, cr_cache, CR_MAX, PRIME_MAX):
    R = np.empty((l, m), dtype=int)
    for i in range(l):
        if i + 1 <= t:
            for j in range(m):
                if i == j:
                    q_i = primes_to_use[i][0] ** primes_to_use[i][1]
                    res = cubic_residue(B / q_i, primes_to_check[i], cr_cache, prime_above_cache, CR_MAX, PRIME_MAX)
                    R[i][j] = {0: 0, 1: 2, 2: 1}[res]
                else:
                    R[i][j] = cubic_residue(primes_to_use[j][0] ** primes_to_use[i][1],
                                            primes_to_check[i], cr_cache, prime_above_cache, CR_MAX, PRIME_MAX)
        else:
            for j in range(m):
                R[i][j] = cubic_residue(primes_to_use[j][0], primes_to_check[i], cr_cache, prime_above_cache, CR_MAX, PRIME_MAX)
    return l, m - 1, trim_matrix(R)

def solve_fac(B, B_fac, A, prime_above_cache, cr_cache, CR_MAX, PRIME_MAX):
    dp = 27 * B - A**3
    t1, t2 = [], []
    primes_to_check = []
    for p, exp in B_fac:
        if A % p == 0 and p % 3 == 1:
            primes_to_check.append(p)
            t1.append([p, exp])
        else:
            t2.append([p, exp])
    primes_to_use = t1 + t2
    m, t = len(primes_to_use), len(primes_to_check)
    for p, exp in dp.factor():
        if p % 3 == 1 and B % p != 0:
            primes_to_check.append(p)
    l = len(primes_to_check)
    return build_matrix(primes_to_check, primes_to_use, m, l, t, B, prime_above_cache, cr_cache, CR_MAX, PRIME_MAX)



# -------------------------
# Worker & parallel functions
# -------------------------

def worker(args):
    (min_H, max_H, H, cache_limits,  debug) = args
    CR_MAX, PRIME_MAX = cache_limits

    # local caches
    cr_cache = {}
    prime_above_cache = {}
    # batch-level counting
    batch_counts = defaultdict(int)
    
    for B in range(min_H, max_H):
        B_fac = Integer(B).factor()
        for A in range(-int(H^(1/3)), int(H^(1/3))):
            newA, new_b_fac, newB = clean_fac(A, B_fac)
              
            if newA != A or newB != B: continue
            
            l, m, mat = solve_fac(B, B_fac, A, prime_above_cache, cr_cache, CR_MAX, PRIME_MAX)
            compressed = ''.join(str(num) for row in mat for num in row)
              
            key = (int(l - m), int(m), compressed)
            batch_counts[key] += 1

    # convert to JSON-friendly format
    results_out = [
        {"pair": [l_minus_m, m], "matrix": matrix, "count": int(count)}
        for (l_minus_m, m, matrix), count in batch_counts.items()
    ]
    
    if debug:
        return results_out, len(cr_cache), len(prime_above_cache)
    return results_out

def main_parallel(H,
            a,
            out_file,
            batch_size=2000,
            nprocesses=min(os.cpu_count(), 32),
            cache_limits=(10_000, 10_000),
            debug=False):

    CR_MAX, PRIME_MAX = cache_limits

    batches = [
        (a*H + batch_size * i, min(a*H + batch_size * (i+1), (a+1)*H), (a+1)*H, cache_limits, debug)
        for i in range(int(H/batch_size)+1)
    ]

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_files = []

    with Pool(processes=nprocesses) as pool:
        for batch_id, result in enumerate(pool.imap_unordered(worker, batches, chunksize=1)):
            results_out = result if not debug else result[0]

            # Write each batch to a separate temp file
            temp_file = out_path.parent / f"{out_path.stem}_batch{batch_id}.tmp"
            temp_files.append(temp_file)
            with open(temp_file, "w") as f:
                for rec in results_out:
                    f.write(json.dumps(rec) + "\n")
            
            # Periodic aggregation
            if (batch_id + 1) % aggregate_every == 0:
                _aggregate_temp_files(temp_files, out_path)
                temp_files = []  # reset temp file list
                if debug:
                    print(f"[batch {batch_id}] checkpointed, cache sizes: cr={cr_size}, prime={prime_size}")

    # Final aggregation for any remaining temp files
    if temp_files:
        _aggregate_temp_files(temp_files, out_path)

    print("Finished all batches.")



def _aggregate_temp_files(temp_files, out_file):
    """Read all temp files, aggregate counts, and safely write to main file."""
    agg_counts = defaultdict(int)

    # Read all temp files
    for temp_file in temp_files:
        with open(temp_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                agg_counts[key] += rec["count"]
        os.remove(temp_file)  # remove temp file after reading

    # Merge with existing main file if it exists
    if Path(out_file).exists():
        with open(out_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                agg_counts[key] += rec["count"]

    # Write aggregated data safely
    temp_out = str(out_file) + ".agg.tmp"
    with open(temp_out, "w") as f:
        for (pair, matrix), count in agg_counts.items():
            f.write(json.dumps({"pair": list(pair), "matrix": matrix, "count": count}) + "\n")

    os.replace(temp_out, out_file)
    print(f"[INFO] Aggregated counts written to {out_file}")

# -------------------------
# Batch and cache estimation
# -------------------------
def estimate_optimal_batch_size(H, a, cache_limits=(20_000, 20_000),
                                nprocesses=None, safe_fraction=0.5,
                                target_batch_seconds=30,total_mem=None):

    if nprocesses is None:
        nprocesses = min(cpu_count(), 4)

    CR_MAX, PRIME_MAX = cache_limits
    cr_cache = {}
    prime_above_cache = {}

    small_batch_size = 10
    start_mem = psutil.Process(os.getpid()).memory_info().rss
    start_time = time.time()
    
    for B in range(a*H, a*H + small_batch_size):
        B_fac = Integer(B).factor()
        for A in range(-int(H^(1/3)), int(H^(1/3))):
              newA, new_b_fac, newB = clean_fac(A, B_fac)
              
              if newA != A or newB != B: continue
              solve_fac(B, B_fac, A, prime_above_cache, cr_cache, CR_MAX, PRIME_MAX)

    end_time = time.time()
    end_mem = psutil.Process(os.getpid()).memory_info().rss

    mem_per_item = max(end_mem - start_mem, 1e6) / small_batch_size
    time_per_item = (end_time - start_time) / small_batch_size

    if not total_mem:
        total_mem = psutil.virtual_memory().total
    safe_mem = total_mem * safe_fraction

    memory_limited_batch = int(safe_mem / mem_per_item)
    time_limited_batch = max(1, int(target_batch_seconds / time_per_item))

    optimal_batch = min(memory_limited_batch, time_limited_batch)

    print(f"[DEBUG] Memory per item: {mem_per_item / 1e6:.2f} MB")
    print(f"[DEBUG] Time per item: {time_per_item:.3f} s")
    print(f"[DEBUG] Memory-limited batch size: {memory_limited_batch}")
    print(f"[DEBUG] Time-limited batch size: {time_limited_batch}")
    print(f"[DEBUG] Chosen optimal batch size: {optimal_batch}")

    return optimal_batch

def estimate_cache_sizes(H, a, safe_fraction=0.5,total_mem=None):
    CR_MAX_TEST, PRIME_MAX_TEST = 100, 100
    cr_cache = {}
    prime_above_cache = {}

    start_mem = psutil.Process(os.getpid()).memory_info().rss
    
    for B in range(a*H, a*H + 1):
        B_fac = Integer(B).factor()
        for A in range(-int(H**(1/3)), int(H**(1/3))):
            newA, new_b_fac, newB = clean_fac(A, B_fac)
            
            if newA != A or newB != B: 
                continue
            
            solve_fac(B, B_fac, A, prime_above_cache, cr_cache, CR_MAX_TEST, PRIME_MAX_TEST)
           
    end_mem = psutil.Process(os.getpid()).memory_info().rss
    mem_used = max(end_mem - start_mem, 1e6)
    n_cr_entries = len(cr_cache) or 1
    n_prime_entries = len(prime_above_cache) or 1

    mem_per_cr = mem_used / n_cr_entries
    mem_per_prime = mem_used / n_prime_entries

    if not total_mem:
        total_mem = psutil.virtual_memory().total
    safe_mem = total_mem * safe_fraction

    cr_cache_size = int(safe_mem * 2 / 3 / mem_per_cr)
    prime_cache_size = int(safe_mem / 3 / mem_per_prime)

    print(f"[DEBUG] mem/cr_entry={mem_per_cr:.0f}, mem/prime_entry={mem_per_prime:.0f}")
    print(f"[DEBUG] Estimated CR_MAX={cr_cache_size}, PRIME_MAX={prime_cache_size}")

    return cr_cache_size, prime_cache_size


# -------------------------
# Run example
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num1", type=int, required=True)
    parser.add_argument("--num2", type=int, required=True)
    parser.add_argument("--mem", type=int, required=False)

    args = parser.parse_args()

    H = args.num1
    a = args.num2
    total_mem = args.mem


    out_file = f"data/checkpoint_{H}_{a}.jsonl"

    # Estimate batch size and cache sizes
    batch_size = estimate_optimal_batch_size(H,a,nprocesses=min(os.cpu_count(), 64),total_mem=total_mem)
    
    CR_MAX, PRIME_MAX = estimate_cache_sizes(H,a,total_mem=total_mem)
        
    print(f"Running H={H}, a={a} "
          f"batch_size={batch_size}, CR_MAX={CR_MAX}, PRIME_MAX={PRIME_MAX}")

    main_parallel(
        H,
        a,
        out_file=out_file,
        batch_size=batch_size,
        nprocesses=min(os.cpu_count(), 64),
        cache_limits=(CR_MAX, PRIME_MAX),
        debug=True
    )
       
