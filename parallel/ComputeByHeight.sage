"""
Selmer Group Computation for Elliptic Curves y² + Axy + By = x³

This module computes statistics on Selmer groups by:
1. Generating random (A,B) pairs with specific prime structure
2. Computing cubic residue symbols in Z[ω] (Eisenstein integers)
3. Building matrices whose nullspace gives the Selmer group
4. Aggregating results across many random instances
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

import numpy as np
import psutil

# Ensure consistent multiprocessing behavior across platforms.
mp.set_start_method("spawn", force=True)

# =============================================================================
# Canonicalization of (A,B)
# =============================================================================

def strip_common_cube_factors(A_value, B_factorization):
    """
    Remove common cube factors to put (A,B) in canonical form.
    
    Applies the transformation: if p|A and p³|B, replace:
        A → A/p
        B → B/p³
    
    Repeat until no such p exists. This is important because:
    - It ensures we don't overcount equivalent curves
    - It makes the Selmer matrix well-defined
    - The transformation preserves the isomorphism class
    
    Example:
        A = 12 = 2² × 3
        B = 2³ × 3³ × 5
        
        Step 1 (p=2): 2|12 and 2³|B → A = 6, B = 3³ × 5
        Step 2 (p=3): 3|6 and 3³|B → A = 2, B = 5
        Final: A = 2, B = 5

    Parameters
    ----------
    A_value : int or Integer
        The coefficient A.
    B_factorization : list[list[int, int]]
        Factorization of B as [[p1,e1], [p2,e2], ...] with primes pi and exponents ei.

    Returns
    -------
    reduced_A : int
        Reduced A.
    reduced_B_factorization : list[list[int, int]]
        Updated factorization after stripping cube factors.
    reduced_B : int
        Reduced B (reconstructed from reduced_B_factorization).
    """
    reduced_B_factorization = []
    reduced_B = 1

    for prime, exponent in B_factorization:
        # If exponent < 3 or prime ∤ A, no cube stripping is possible for this prime.
        if exponent < 3 or (A_value % prime) != 0:
            reduced_B_factorization.append([prime, exponent])
            reduced_B *= prime ** exponent
            continue

        # How many times does p divide A?
        prime_adic_in_A = valuation(A_value, prime)

        # Apply transformation as many times as possible
        # Limited by both p^k || A and p^(3k) || B
        num_strips = min(prime_adic_in_A, exponent // 3)

        A_value //= prime ** num_strips
        remaining_exp = exponent - 3 * num_strips

        if remaining_exp > 0:
            reduced_B_factorization.append([int(prime), int(remaining_exp)])
            reduced_B *= prime ** remaining_exp

    return int(A_value), reduced_B_factorization, int(reduced_B)

# =============================================================================
# Selmer matrix construction
# =============================================================================

def build_selmer_matrix(check_primes, basis_prime_powers, num_special_rows,
                        B_value, cubic_residue_symbol):
    """
    Construct the Selmer matrix for the 3-isogeny descent.

    Rows correspond to "check primes" where cubic residue symbols are evaluated.
    Columns correspond to prime powers dividing B (ordered so primes dividing A
    with p ≡ 1 (mod 3) appear first).

    The first `num_special_rows` rows correspond to primes p ≡ 1 (mod 3) dividing
    both A and B, and use the modified diagonal rule appropriate for the dual
    isogeny contribution.

    For tractability, very large matrices are discarded (returning None).

    Parameters
    ----------
    check_primes : list[int]
        Primes indexing rows.
    basis_prime_powers : list[tuple[int,int]]
        Column descriptors (p, exponent) describing the prime power p^exponent.
    num_special_rows : int
        Number of initial rows with special handling.
    B_value : int
        The coefficient B.
    cubic_residue_symbol : callable
        Function (a, p) -> {0,1,2} giving the cubic residue symbol for integer a mod p.

    Returns
    -------
    num_rows : int
    num_cols_after_trim : int
    matrix : np.ndarray or None
        Matrix with entries in {0,1,2}, with final torsion column removed, or None.
    """
    num_rows = len(check_primes)
    num_cols = len(basis_prime_powers)

    # There are too many possible matrices of size num_rows x num_cols to see
    # meaningful distributions in a brute-force experiment; skip large instances.
    if num_rows * num_cols > 12:
        return num_rows, num_cols, None

    matrix = np.empty((num_rows, num_cols), dtype=np.int8)

    for row_index, check_prime in enumerate(check_primes):
        if row_index < num_special_rows:
            # Special rows: primes p ≡ 1 (mod 3) dividing both A and B.
            for col_index, (prime, exponent) in enumerate(basis_prime_powers):
                if row_index == col_index:
                    prime_power = prime ** exponent
                    residue = cubic_residue_symbol(B_value // prime_power, check_prime)
                    # Convention swap on the diagonal for the special rows.
                    matrix[row_index, col_index] = {0: 0, 1: 2, 2: 1}[residue]
                else:
                    matrix[row_index, col_index] = cubic_residue_symbol(prime ** exponent, check_prime)
        else:
            # Regular rows: primes from the discriminant.
            # Only test the underlying prime, not its exponent.
            for col_index, (prime, _exponent) in enumerate(basis_prime_powers):
                matrix[row_index, col_index] = cubic_residue_symbol(prime, check_prime)

    # Remove the final column: it corresponds to the rational 3-torsion point.
    # The nullspace dimension over F_3 gives the Selmer rank contribution.
    return num_rows, num_cols - 1, matrix[:, :-1]

def compute_selmer_matrix(B_value, B_factorization, A_value, cubic_residue_symbol, string_matrix=True):
    """
    Compute the Selmer matrix for y² + Axy + By = x³.

    The check primes consist of:
      (i) primes p ≡ 1 (mod 3) dividing both A and B,
     (ii) primes p ≡ 1 (mod 3) dividing the discriminant Δ = 27B - A^3
          that are not already divisors of B.

    The basis columns consist of all prime powers dividing B, ordered so that
    primes dividing A with p ≡ 1 (mod 3) appear first.

    Parameters
    ----------
    B_value : int
    B_factorization : list[list[int,int]]
    A_value : int
    cubic_residue_symbol : callable

    Returns
    -------
    (num_rows, num_cols, matrix_string) : tuple[int,int,str]
        If the matrix is discarded due to size, matrix_string is ''.
    """
    discriminant = 27 * B_value - A_value ** 3

    check_primes = []
    basis_prime_powers = []

    # (i) p ≡ 1 (mod 3) dividing both A and B
    for prime, exponent in B_factorization:
        if (A_value % prime == 0) and (prime % 3 == 1):
            check_primes.append(prime)
            basis_prime_powers.append((prime, exponent))

    num_special_rows = len(check_primes)

    # Remaining primes dividing B
    for prime, exponent in B_factorization:
        if not ((A_value % prime == 0) and (prime % 3 == 1)):
            basis_prime_powers.append((prime, exponent))

    # (ii) p ≡ 1 (mod 3) dividing discriminant but not dividing B
    for prime, _ in discriminant.factor():
        if (prime % 3 == 1) and (B_value % prime != 0):
            check_primes.append(prime)

    num_rows, num_cols, matrix = build_selmer_matrix(
        check_primes=check_primes,
        basis_prime_powers=basis_prime_powers,
        num_special_rows=num_special_rows,
        B_value=B_value,
        cubic_residue_symbol=cubic_residue_symbol,
    )
    if string_matrix:
    
        if matrix is None:
            return int(num_rows), int(num_cols), ""

        matrix_string = "".join(str(int(x)) for x in matrix.ravel())
        return int(num_rows), int(num_cols), matrix_string
    
    else: 
        return int(num_rows), int(num_cols), matrix

# =============================================================================
# Cache sizing
# =============================================================================

def compute_cache_limits(memory_fraction=0.5):
    """
    Choose per-process LRU cache sizes from available RAM.

    The goal is to avoid memory pressure under multiprocessing: each worker
    has its own caches, so cache sizes must be conservative.

    Parameters
    ----------
    memory_fraction : float
        Fraction of available RAM to allocate per process to caching.

    Returns
    -------
    cubic_residue_cache_size : int
        Maximum number of cached (a,p) values.
    cube_root_cache_size : int
        Maximum number of cached cube-root auxiliary data values.
    """
    available_bytes = psutil.virtual_memory().available * float(memory_fraction)

    # Coarse per-entry estimates (Python object overhead dominates).
    bytes_per_cube_root_entry = 200
    bytes_per_cubic_residue_entry = 300

    cube_root_cache_size = int(available_bytes * 0.25 / bytes_per_cube_root_entry)
    cubic_residue_cache_size = int(available_bytes * 0.75 / bytes_per_cubic_residue_entry)
    return cubic_residue_cache_size, cube_root_cache_size


# =============================================================================
# Per-process cached cubic residue symbol
# =============================================================================

# These globals are initialized once per worker process and then reused across tasks.
_CUBE_ROOT_DATA = None
_CUBIC_RESIDUE_CACHED = None
_CUBIC_RESIDUE_SYMBOL = None


def _init_worker_caches(cache_fraction):
    """
    Initialize per-process cached arithmetic for cubic residue evaluation.

    Under the "spawn" start method, any caches defined inside the worker entry
    function are re-created for each task and are therefore ineffective. Defining
    caches at module scope and initializing them once per worker process ensures
    that a worker can reuse its warmed cache across many batches.
    """
    global _CUBE_ROOT_DATA, _CUBIC_RESIDUE_CACHED, _CUBIC_RESIDUE_SYMBOL

    if _CUBIC_RESIDUE_SYMBOL is not None:
        return

    cubic_residue_cache_size, cube_root_cache_size = compute_cache_limits(cache_fraction)

    @lru_cache(maxsize=cube_root_cache_size)
    def cube_root_data(p):
        """
        Return the nontrivial cube roots of unity modulo p.

        For p ≡ 1 (mod 3), the multiplicative group (Z/pZ)^× contains a unique
        subgroup of order 3. We store (z, z^2) where z has exact order 3 mod p.
        """
        generator = primitive_root(p)
        z = pow(int(generator), (p - 1) // 3, p)
        z2 = (z * z) % p
        return int(z), int(z2)

    @lru_cache(maxsize=cubic_residue_cache_size)
    def cubic_residue_cached(a, p):
        """
        Compute the cubic residue symbol (a/p)_3 for integer a and prime p ≡ 1 (mod 3).

        Returns
        -------
        0 if a is a cubic residue modulo p,
        1 or 2 for the two nontrivial cubic characters.

        Notes
        -----
        This function is performance-critical. It avoids raising exceptions in the
        hot path; if a ≡ 0 (mod p) occurs, the return value is set to 0 to keep the
        computation total and inexpensive.
        """
        a %= p
        if a == 0:
            return 0

        value = pow(int(a), (p - 1) // 3, p)
        if value == 1:
            return 0

        z, z2 = cube_root_data(p)
        return 1 if value == z else 2

    def cubic_residue_symbol(a, p):
        """
        Compute (a/p)_3 for any prime p.

        For p ≡ 2 (mod 3), every nonzero class is a cube and the symbol is trivial.
        For p ≡ 1 (mod 3), the value is computed by exponentiation and identified
        with the cube roots of unity modulo p.
        """
        if p % 3 == 2:
            return 0
        return cubic_residue_cached(a, p)

    _CUBE_ROOT_DATA = cube_root_data
    _CUBIC_RESIDUE_CACHED = cubic_residue_cached
    _CUBIC_RESIDUE_SYMBOL = cubic_residue_symbol


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
        (B_values, min_height, max_height, forbidden_prime_product, cache_fraction)

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
    ) = task_args

    _init_worker_caches(cache_fraction)
    cubic_residue_symbol = _CUBIC_RESIDUE_SYMBOL

    counts = defaultdict(int)

    for B_value in B_values:
        # B_values were prefiltered so gcd(B_value, forbidden_prime_product)=1 holds.
        B_factorization = list(factor(B_value))

        # Heuristic lower bound for |A| based on cube root of B (as in the original code).
        cube_root_B = int(B_value ** (1.0 / 3.0) + 1.0)
        A_lower = 1 if cube_root_B > min_height else min_height

        for A_abs, A_abs_cubed in _iterate_A_abs_with_cubes(A_lower, max_height):
            # Check we haven't already seen this curve
            reduced_A, _reduced_B_fac, reduced_B = strip_common_cube_factors(A_abs, B_factorization)
            if not (reduced_B == B_value and reduced_A == A_abs):
                continue

            # Ensure elliptic curve has good reduction at forbidden primes
            for sign in (1, -1):
                if not _passes_bad_prime_sieve_for_sign(
                    A_abs=A_abs,
                    A_abs_cubed=A_abs_cubed,
                    B_value=B_value,
                    forbidden_prime_product=forbidden_prime_product,
                    sign=sign,
                ):
                    continue

                A_value = sign * A_abs
                key = compute_selmer_matrix(B_value, B_factorization, A_value, cubic_residue_symbol)
                counts[key] += 1

    return [
        {"pair": [int(r), int(c)], "matrix": m, "count": int(k)}
        for (r, c, m), k in counts.items()
    ]

    
# =============================================================================
# Fast discriminant screening (avoid bad primes dividing B*(A^3 - 27B))
# =============================================================================

def _passes_bad_prime_sieve_for_sign(A_abs, A_abs_cubed, B_value, forbidden_prime_product, sign):
    """
    Return True if the discriminant of the model with A = sign*A_abs is coprime
    to the forbidden prime product, assuming gcd(B_value, forbidden_prime_product)=1.

    For models y^2 + Axy + By = x^3, the relevant factor is B^3*(A^3 - 27B).
    Since batches already enforce gcd(B, forbidden)=1, it suffices to test
        gcd(A^3 - 27B, forbidden) = 1
    with A = sign*A_abs.

    Notes
    -----
    For sign = +1, we test A_abs^3 - 27B.
    For sign = -1, we test (-A_abs)^3 - 27B = -(A_abs^3 + 27B),
    so we test A_abs^3 + 27B instead.
    """
    if sign == 1:
        disc_factor = A_abs_cubed - 27 * B_value
    else:
        disc_factor = A_abs_cubed + 27 * B_value  # sign just introduces a minus overall

    return gcd(disc_factor, forbidden_prime_product) == 1


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
# Aggregation utilities
# =============================================================================

def aggregate_temp_files(temp_files, output_file):
    """
    Merge a list of temporary batch files into the main output file.

    Each input file is expected to contain one JSON object per line with keys
    "pair", "matrix", and "count". Counts are summed over identical keys.

    The main output file is updated atomically (write to a temporary file and rename),
    and processed temporary files are deleted.
    """
    aggregated = defaultdict(int)

    for temp_path in temp_files:
        with open(temp_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                aggregated[key] += rec["count"]
        os.remove(temp_path)

    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                aggregated[key] += rec["count"]

    tmp_path = str(output_path) + ".tmp"
    with open(tmp_path, "w") as f:
        for (pair, mat), c in aggregated.items():
            f.write(json.dumps({"pair": list(pair), "matrix": mat, "count": c}) + "\n")

    os.replace(tmp_path, output_path)


def aggregate_main_file(output_file):
    """
    Deduplicate and sum counts within the main output file in place.
    """
    aggregated = defaultdict(int)
    output_path = Path(output_file)

    if not output_path.exists():
        return

    with open(output_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (tuple(rec["pair"]), rec["matrix"])
            aggregated[key] += rec["count"]

    with open(output_path, "w") as f:
        for (pair, mat), c in aggregated.items():
            f.write(json.dumps({"pair": list(pair), "matrix": mat, "count": c}) + "\n")


# =============================================================================
# Parallel driver
# =============================================================================

def run_parallel(min_height, max_height, forbidden_prime_product, output_file,
                 batch_size=10, aggregate_every=10, cache_fraction=0.5, num_processes=None):
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
    if num_processes is None:
        num_processes = mp.cpu_count()

    if num_processes == 1:
        return run_non_parallel(
            min_height=min_height,
            max_height=max_height,
            forbidden_prime_product=forbidden_prime_product,
            output_file=output_file,
            cache_fraction=cache_fraction,
        )

    B_max = max_height ** 3

    # Construct tasks: each task is one batch of admissible B values.
    tasks = []
    for start in range(2, B_max + 1, batch_size):
        end = min(start + batch_size, B_max + 1)
        B_values = [
            B for B in range(start, end)
            if gcd(B, forbidden_prime_product) == 1
        ]
        if B_values:
            tasks.append((B_values, min_height, max_height, forbidden_prime_product, cache_fraction))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "tmp"
    temp_dir.mkdir(exist_ok=True)

    temp_files = []

    with mp.Pool(processes=num_processes) as pool:
        for task_id, result in enumerate(pool.imap_unordered(worker_process, tasks, chunksize=1)):
            temp_path = temp_dir / f"{output_path.stem}_task{task_id}.tmp"
            temp_files.append(temp_path)

            with open(temp_path, "w") as f:
                for rec in result:
                    f.write(json.dumps(rec) + "\n")

            if (task_id + 1) % aggregate_every == 0:
                aggregate_temp_files(temp_files, output_path)
                temp_files = []

    if temp_files:
        aggregate_temp_files(temp_files, output_path)

    print("Finished all batches.")

# =============================================================================
# Serial path
# =============================================================================

def run_non_parallel(min_height, max_height, forbidden_prime_product, output_file,
                     cache_fraction=0.5):
    """
    Serial implementation used for debugging and profiling.

    This shares the same per-process caching initialization and the same
    mathematical filters as the parallel path.
    """
    _init_worker_caches(cache_fraction)
    cubic_residue_symbol = _CUBIC_RESIDUE_SYMBOL

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    B_max = max_height ** 3
    block_size = 1000

    for start in range(2, B_max + 1, block_size):
        end = min(start + block_size, B_max + 1)
        B_values = [B for B in range(start, end) if gcd(B, forbidden_prime_product) == 1]
        if not B_values:
            continue

        for B_value in B_values:
            B_factorization = list(factor(B_value))

            cube_root_B = int(B_value ** (1.0 / 3.0) + 1.0)
            A_lower = 1 if cube_root_B > min_height else min_height

            for A_abs, A_abs_cubed in _iterate_A_abs_with_cubes(A_lower, max_height):
                reduced_A, _reduced_B_fac, reduced_B = strip_common_cube_factors(A_abs, B_factorization)
                if not (reduced_B == B_value and reduced_A == A_abs):
                    continue

                for sign in (1, -1):
                    if not _passes_bad_prime_sieve_for_sign(
                        A_abs=A_abs,
                        A_abs_cubed=A_abs_cubed,
                        B_value=B_value,
                        forbidden_prime_product=forbidden_prime_product,
                        sign=sign,
                    ):
                        continue

                    A_value = sign * A_abs
                    key = compute_selmer_matrix(B_value, B_factorization, A_value, cubic_residue_symbol)
                    counts[key] += 1

        # Write an intermediate snapshot.
        snapshot = [
            {"pair": [int(r), int(c)], "matrix": m, "count": int(k)}
            for (r, c, m), k in counts.items()
        ]
        with open(output_path, "w") as f:
            for rec in snapshot:
                f.write(json.dumps(rec) + "\n")

    aggregate_main_file(output_path)
    print("Finished all computations.")



# =============================================================================
# CLI entry point
# =============================================================================
def forbidden_prime_product_from_cutoff(min_prime):
    """
    Return the product of all primes < min_prime, together with 3.

    The prime 3 is always excluded explicitly, even if min_prime <= 3.
    """
    bad_primes = [p for p in prime_range(min_prime) if p != 3]
    bad_primes.append(3)
    return prod(bad_primes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_height", type=int, default=1)
    parser.add_argument("--max_height", type=int, required=True)
    parser.add_argument("--nprocesses", type=int, default=None)
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

    output_file = f"data/output_{args.min_height}_output_{args.max_height}.jsonl"

    if args.debug:
        import cProfile
        import pstats

        with cProfile.Profile() as profiler:
            run_parallel(
                min_height=args.min_height,
                max_height=args.max_height,
                forbidden_prime_product=forbidden_prime_product,
                output_file=output_file,
                batch_size=10,
                aggregate_every=10,
                cache_fraction=0.5,
                num_processes=args.nprocesses,
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
            batch_size=10,
            aggregate_every=10,
            cache_fraction=0.5,
            num_processes=args.nprocesses,
        )
    

