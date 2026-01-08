import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from functools import lru_cache
import numpy as np
import random
import json
from sympy import nextprime
from collections import defaultdict
from pathlib import Path
import os
import argparse
import psutil
import math
import time

# ------------------------------------------------------------
# Sage objects (must exist at module level)
#
#   K.<ω> = CyclotomicField(3)
#   O = K.ring_of_integers()
# ------------------------------------------------------------


# ============================================================
# Parent-side arithmetic
# ============================================================

def prime_above_uncached(p):
    """
    Compute generator for a prime above p in Z[ω].
    Used ONLY for parent-side precomputation.
    """
    if p % 3 == 2:
        return (p, 0)

    K.<ω> = CyclotomicField(3)
    O = K.ring_of_integers()
    
    factor = O(p).factor()[0][0]
    
    return (int(factor[0]), int(factor[1]))


# ============================================================
# Utilities
# ============================================================

def trim_matrix(M):
    """Drop final column of the matrix M."""
    return np.delete(M, -1, axis=1)


def random_exponent(p):
    """
        Sample a random exponent e >= 1 for prime p.
        Probability distribution: P(e = k) = (1 - 1/p) * p^{-(k-1)}
    """
    return 1 + int(math.log(random.random()) / math.log(1 / p))


def generate_primes(num_of_primes):
    """
    Generate first num_of_primes primes > 3, split by residue mod 3.
    
    Args:
        num_of_primes: Number of primes to generate
    
    Returns:
        p1: List of primes ≡ 1 (mod 3)
        p2: List of primes ≡ 2 (mod 3)
    """
    p1 = []  # primes ≡ 1 (mod 3)
    p2 = []  # primes ≡ 2 (mod 3)
    
    p = 3
    
    for _ in range(num_of_primes):
        p = nextprime(p)
        if p % 3 == 1:
            p1.append(int(p))
        else:  # p % 3 == 2 (all primes > 3 are 1 or 2 mod 3)
            p2.append(int(p))
    
    return p1, p2

# ============================================================
# Factor manipulation
# ============================================================

def strip_cube_factors(a, b_factorization):
    """
    Simplify A and B by removing common cube factors.
    
    Repeatedly applies the transformation: if p|A and p³|B, replace A → A/p and B → B/p³.
    
    Args:
        a: Integer coefficient A
        b_factorization: List of [prime, exponent] pairs representing B's factorization
        
    Returns:
        tuple: (reduced_a, reduced_b_factorization, reduced_b)
            - reduced_a: A after removing common factors
            - reduced_b_factorization: B's factorization after removing cube factors
            - reduced_b: Integer value of reduced B
    
    Example:
        If A = 12 = 2² × 3 and B = 2³ × 3³ × 5:
        - p=2: 2|12 and 2³|B, so A → 12/2 = 6, B → B/2³ = 3³ × 5
        - p=3: 3|6 and 3³|B, so A → 6/3 = 2, B → B/3³ = 5
        Final: A = 2, B = 5
    """
    reduced_b_factorization = []
    reduced_b = 1
    
    for p, exp in b_factorization:
        # Skip if we can't apply the transformation (exp < 3 or p doesn't divide A)
        if exp < 3 or a % p != 0:
            reduced_b_factorization.append([p, exp])
            reduced_b *= p^exp
            continue
        
        # Count how many times we can apply: A → A/p, B → B/p³
        p_power_in_a = valuation(a, p)
        num_reductions = min(p_power_in_a, exp // 3)
        
        # Apply the transformation num_reductions times
        a //= p^num_reductions
        remaining_exp = exp - 3 * num_reductions
        
        # Store remaining factors of this prime in B
        if remaining_exp > 0:
            reduced_b_factorization.append([p, remaining_exp])
            reduced_b *= p^remaining_exp
    
    return a, reduced_b_factorization, reduced_b



# ============================================================
# Random (A,B)
# ============================================================

def generate_random_A(B, delta=0.1, method="height"):

    if method == "box":
        height = int(B ** (1 / 3))
        return random.randint(
            int(-1 * (1 + delta) * height),
            int((1 + delta) * height),
            )
            
    if method == "ignore height":
        return random.randint(
            int(-B), int(B))
            
    else:
        height = int(B ** (1 / 3))
        return random.randint(
            int((1 - delta) * height),
            int((1 + delta) * height),
            ) * random.choice([1, -1])
            
    
            
def generate_random_A_B(primes_1_mod_3, primes_2_mod_3, num_1_mod_3, num_2_mod_3, method="height"):
    """
    Generate random coprime integers A and B with specified prime structure.
    
    Creates B from random primes (with random exponents), then generates A
    so that (A, B) is already in reduced form.
    
    Args:
        primes_1_mod_3: List of available primes ≡ 1 (mod 3)
        primes_2_mod_3: List of available primes ≡ 2 (mod 3)
        num_1_mod_3: Number of primes ≡ 1 (mod 3) to use in B
        num_2_mod_3: Number of primes ≡ 2 (mod 3) to use in B
        
    Returns:
        tuple: (A, B_factorization, B) where A is not divisible by 3
    """
    # Select random primes from each congruence class
    selected_primes = (random.sample(primes_1_mod_3, num_1_mod_3) + 
                      random.sample(primes_2_mod_3, num_2_mod_3))
    
    # Build B's factorization with random exponents
    b_fac = [(p, random_exponent(p)) for p in selected_primes]
    b = prod([p^exp for p, exp in b_fac])
    
    # Keep generating A until we get one where strip_cube_factors does not reduce B
    while True:
        a = generate_random_A(b, method=method)
        
        # Skip if A is divisible by 3
        if a % 3 == 0:
            continue
        
        # Apply cube factor stripping
        reduced_a, reduced_b_fac, reduced_b = strip_cube_factors(a, b_fac)
        
        # Accept only if B was reduced
        if reduced_b == b:
            return reduced_a, reduced_b_fac, reduced_b


# ============================================================
# Matrix construction
# ============================================================
def build_matrix(check_primes, basis_primes, t, b, cubic_residue):
    """
    Build a matrix of cubic residues for the linear algebra step.
    
    Args:
        check_primes: Primes at which to compute cubic residues (rows)
        basis_primes: Prime powers forming the basis (columns)
        t: Number of primes ≡ 1 (mod 3) dividing A
        b: The value B
        cubic_residue: Function to compute cubic residue
        
    Returns:
        tuple: (l, m - 1, trimmed_matrix)
    """
    l = len(check_primes)
    m = len(basis_primes)
    mat = np.empty((l, m), dtype=int)
    
    for i in range(l):
        if i < t:
            # Special case: primes ≡ 1 (mod 3) dividing A
            for j in range(m):
                p, exp = basis_primes[j]
                
                if i == j:
                    # Diagonal: use B/q with special mapping
                    q = p ** exp
                    res = cubic_residue(b / q, check_primes[i])
                    mat[i][j] = {0: 0, 1: 2, 2: 1}[res]
                else:
                    # Off-diagonal: standard cubic residue
                    mat[i][j] = cubic_residue(p ** exp, check_primes[i])
        else:
            # Standard case: primes from discriminant factorization
            for j in range(m):
                p, _ = basis_primes[j]
                mat[i][j] = cubic_residue(p, check_primes[i])
    
    return l, m - 1, trim_matrix(mat)



def selmer_matrix(b, b_fac, a, cubic_residue):
    """
    Construct the Selmer matrix for the elliptic curve y² + Axy + By = x³.
    
    The nullspace of this matrix (over Z/3Z) is isomorphic to the φ-Selmer group,
    where φ is the 3-isogeny. The matrix encodes cubic residue conditions that
    elements of the Selmer group must satisfy.
    
    Constructs check primes (for rows) and basis primes (for columns):
    - Check primes: primes ≡ 1 (mod 3) dividing both A and B, plus those from discriminant
    - Basis primes: primes ≡ 1 (mod 3) that divide A first, then primes ≡ 2 (mod 3)
    
    Args:
        b: The integer B from the curve equation
        b_fac: List of [prime, exponent] pairs for B
        a: The integer A from the curve equation
        cubic_residue: Function to compute cubic residue
        
    Returns:
        tuple: (num_rows, num_cols, matrix_string) from build_matrix
    """
    disc = 27 * b - a ** 3
    
    # Build check primes and basis, with primes ≡ 1 (mod 3) dividing A first
    check_primes = []
    basis_primes = []
    
    # First pass: primes ≡ 1 (mod 3) that divide A
    for p, exp in b_fac:
        if a % p == 0 and p % 3 == 1:
            check_primes.append(p)
            basis_primes.append([p, exp])
    
    t = len(check_primes)
    
    # Second pass: all other primes from B
    for p, exp in b_fac:
        if not (a % p == 0 and p % 3 == 1):
            basis_primes.append([p, exp])
    
    # Add primes from discriminant factorization
    for p, _ in disc.factor():
        if p % 3 == 1 and b % p != 0:
            check_primes.append(p)
    
    return build_matrix(check_primes, basis_primes, t, b, cubic_residue)


def compute_random_by_prime_factorization(primes_1_mod_3, primes_2_mod_3, 
                                          num_1_mod_3, num_2_mod_3, cubic_residue, method="height"):
    """
    Generate a random instance and compute its Selmer matrix.
    
    Args:
        primes_1_mod_3: List of available primes ≡ 1 (mod 3)
        primes_2_mod_3: List of available primes ≡ 2 (mod 3)
        num_1_mod_3: Number of primes ≡ 1 (mod 3) to use
        num_2_mod_3: Number of primes ≡ 2 (mod 3) to use
        cubic_residue: Function to compute cubic residue
        
    Returns:
        tuple: (num_rows, num_cols, matrix_as_string)
    """
    a, b_factorization, b = generate_random_A_B(
        primes_1_mod_3, primes_2_mod_3, num_1_mod_3, num_2_mod_3, method
    )
    
    num_rows, num_cols, matrix = selmer_matrix(b, b_factorization, a, cubic_residue)
   
    # Flatten matrix to string
    matrix_string = ''.join(str(element) for row in matrix for element in row)
    
    return int(num_rows), int(num_cols), matrix_string


# ============================================================
# Automatic cache sizing (per process)
# ============================================================

def auto_cache_limits(fraction=0.6):
    """
    Estimate cache sizes based on available RAM.
    Conservative and spawn-safe.
    """
    avail = psutil.virtual_memory().available * fraction

    # Very rough per-entry estimates
    bytes_prime = 200
    bytes_residue = 1500
    bytes_cr = 300

    PRIME_MAX = int(avail * 0.15 / bytes_prime)
    RESIDUE_MAX = int(avail * 0.25 / bytes_residue)
    CR_MAX = int(avail * 0.60 / bytes_cr)
    print("Cache Limits", CR_MAX, PRIME_MAX, RESIDUE_MAX)

    return CR_MAX, PRIME_MAX, RESIDUE_MAX


# ============================================================
# Worker
# ============================================================

def worker(args):
    (
        n_iter,
        p1,
        p2,
        num_1_mod_3,
        num_2_mod_3,
        prime_above_precomputed,
        method,
        cache_fraction,
    ) = args

    K.<ω> = CyclotomicField(3)
    O = K.ring_of_integers()
    
    CR_MAX, PRIME_MAX, RESIDUE_MAX = auto_cache_limits(cache_fraction)
    
    # ---------------- prime_above ----------------
    @lru_cache(maxsize=PRIME_MAX)
    def _prime_above_cached(p):
        # only called for values NOT in the precomputed table
        if p % 3 == 2:
            return (p, 0)
        factor = O(p).factor()[0][0]
        return (int(factor[0]), int(factor[1]))

    def prime_above(p):
        '''Return the precomputed value or compute it'''
        return prime_above_precomputed.get(p) or _prime_above_cached(p)

    # ---------------- residue_map ----------------
    @lru_cache(maxsize=RESIDUE_MAX)
    def residue_map(π):
        x, y = π
        π = O(x + ω * y)
        P = O.fractional_ideal(π)
        k = O.residue_field(P)
        red = k.reduction_map()
        return red, red(ω), (π.norm() - 1) // 3, k(1)

    # ---------------- cubic_residue ----------------
    @lru_cache(maxsize=CR_MAX)
    def cubic_residue(a, p):
        a = O(a)
        x, y = prime_above(p)
        red, w, e, one = residue_map((x, y))
        if red(a) == 0:
            return 0
        r = red(a) ** e
        return 0 if r == one else 1 if r == w else 2

    counts = defaultdict(int)

    for _ in range(n_iter):
        key = compute_random_by_prime_factorization(
            p1, p2, num_1_mod_3, num_2_mod_3, cubic_residue, method
        )
        counts[key] += 1
        
    # convert to JSON-friendly format
    results_out = [
        {"pair": [int(l), int(m)], "matrix": matrix, "count": int(count)}
        for (l, m, matrix), count in counts.items()
    ]

    return results_out

# ============================================================
# Aggregation
# ============================================================

def _aggregate_temp_files(temp_files, out_file):
    agg = defaultdict(int)

    for tf in temp_files:
        with open(tf) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                agg[key] += rec["count"]
        os.remove(tf)

    if Path(out_file).exists():
        with open(out_file) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                agg[key] += rec["count"]

    tmp = str(out_file) + ".tmp"
    with open(tmp, "w") as f:
        for (pair, mat), c in agg.items():
            f.write(json.dumps({
                "pair": list(pair),
                "matrix": mat,
                "count": c
            }) + "\n")

    os.replace(tmp, out_file)


# -------------------------
# Batch size estimation
# -------------------------

def estimate_optimal_batch_size(
    number_of_primes,
    primes,
    num_1_mod_3,
    num_2_mod_3,
    test_iterations=200,
    safe_fraction=0.5,
    target_batch_seconds=30,
    nprocesses=None,
):
    """
    Empirically estimate an optimal batch size by measuring:
      - wall-clock time per iteration
      - memory growth per iteration

    The returned batch size is the minimum of:
      - memory-limited batch size
      - time-limited batch size

    This function is:
      - spawn-safe
      - cluster-safe
      - compatible with lru_cache
      - independent of multiprocessing state

    Parameters
    ----------
    number_of_primes : int
        Number of primes used to generate p1, p2 (informational; not mutated)

    primes : tuple (p1, p2)
        Lists of primes ≡ 1 mod 3 and ≡ 2 mod 3

    prime_above_precomputed : dict
        Precomputed prime_above table (read-only)

    num_1_mod_3, num_2_mod_3 : int
        Parameters passed to computeRandomByPrimeFac

    test_iterations : int
        Number of trial iterations used for estimation

    safe_fraction : float
        Fraction of total RAM allowed for batch memory usage

    target_batch_seconds : int
        Target wall-clock runtime per batch

    Returns
    -------
    int
        Recommended batch size
    """


    if nprocesses is None:
        nprocesses = 1  # IMPORTANT: single-process measurement only

    p1, p2 = primes
    
    K.<ω> = CyclotomicField(3)
    O = K.ring_of_integers()

    def prime_above(p):
        return prime_above_uncached(p)

    def residue_map(π):
        x, y = π
        π = O(x + ω * y)
        P = O.fractional_ideal(π)
        k = O.residue_field(P)
        red = k.reduction_map()
        return red, red(ω), (π.norm() - 1) // 3, k(1)
        
    def cubic_residue(a, p):
        a = O(a)
        x, y = prime_above(p)
        red, w, e, one = residue_map((x, y))
        if red(a) == 0:
            return 0
        r = red(a) ** e
        return 0 if r == one else 1 if r == w else 2

    proc = psutil.Process(os.getpid())

    # -------------------------
    # Measure memory + time
    # -------------------------
    mem_before = proc.memory_info().rss
    t0 = time.time()

    for _ in range(test_iterations):
        compute_random_by_prime_factorization(
            p1, p2,
            num_1_mod_3,
            num_2_mod_3,
            cubic_residue
        )

    t1 = time.time()
    mem_after = proc.memory_info().rss

    # -------------------------
    # Per-iteration estimates
    # -------------------------
    elapsed = max(t1 - t0, 1e-6)
    mem_used = max(mem_after - mem_before, 1e6)

    time_per_item = elapsed / test_iterations
    mem_per_item = mem_used / test_iterations

    # -------------------------
    # System limits
    # -------------------------
    total_mem = float(psutil.virtual_memory().total)
    safe_mem = total_mem * float(safe_fraction)

    memory_limited_batch = int(safe_mem / mem_per_item)
    time_limited_batch   = max(1, int(target_batch_seconds / time_per_item))

    optimal_batch = max(1, min(memory_limited_batch, time_limited_batch))

    # -------------------------
    # Diagnostics
    # -------------------------
    print("[BATCH ESTIMATE]")
    print(f"  time/item      : {time_per_item:.4f} s")
    print(f"  mem/item       : {mem_per_item / 1e6:.2f} MB")
    print(f"  memory limit   : {memory_limited_batch}")
    print(f"  time limit     : {time_limited_batch}")
    print(f"  chosen batch   : {optimal_batch}")

    return optimal_batch


# ============================================================
# Parallel driver (periodic aggregation)
# ============================================================

def main_parallel(
    p1,
    p2,
    N,
    num_1_mod_3,
    num_2_mod_3,
    out_file,
    method="height",
    batch_size=2000,
    aggregate_every=10,
    cache_fraction=0.6,
    nprocesses=None,
):
    if nprocesses is None:
        nprocesses = mp.cpu_count()
        
    prime_above_cache = {p: prime_above_uncached(p) for p in p1}

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    batches, temp_files = [], []

    for i in range(0, N, batch_size):
        batches.append((
            min(batch_size, N - i),
            p1, p2,
            num_1_mod_3, num_2_mod_3,
            prime_above_cache,
            method,
            cache_fraction
        ))
        
    with mp.Pool(processes=nprocesses) as pool:
        for batch_id, result in enumerate(pool.imap_unordered(worker, batches, chunksize=1)):

            # Write each batch to a separate temp file
            temp_file = out_path.parent / f"{out_path.stem}_batch{batch_id}.tmp"
            temp_files.append(temp_file)
            
            with open(temp_file, "w") as f:
                for rec in result:
                    f.write(json.dumps(rec) + "\n")
            
            # Periodic aggregation
            if (batch_id + 1) % aggregate_every == 0:
                _aggregate_temp_files(temp_files, out_path)
                temp_files = []  # reset temp file list

    # Final aggregation for any remaining temp files
    if temp_files:
        _aggregate_temp_files(temp_files, out_path)

    print("Finished all batches.")



# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num1", type=int, required=True)
    parser.add_argument("--num2", type=int, required=True)
    parser.add_argument("--num3", type=int, required=True)
    parser.add_argument("--num4", type=int, required=True)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    num_1_mod_3 = args.num1
    num_2_mod_3 = args.num2
    number_of_primes = args.num3
    N = args.num4
    method = args.method

    p1, p2 = generate_primes(number_of_primes)

    batch_size = estimate_optimal_batch_size(
        number_of_primes=number_of_primes,
        primes=(p1, p2),
        num_1_mod_3=num_1_mod_3,
        num_2_mod_3=num_2_mod_3,
    )
    
    main_parallel(
        p1=p1,
        p2=p2,
        N=N,
        num_1_mod_3=num_1_mod_3,
        num_2_mod_3=num_2_mod_3,
        out_file=f"data/checkpoint_{num_1_mod_3}_{num_2_mod_3}.jsonl",
        method=method,
        batch_size=2000,
        aggregate_every=10,
        cache_fraction=0.6,
        nprocesses=min(mp.cpu_count(), 64),
    )
