"""Shared utilities for 3-isogeny Selmer matrix experiments.

This file is intended to be `load()`-ed from multiple Sage scripts.
It contains the performance-critical core:
- cube root data + cubic residue symbol with per-process LRU caches
- Selmer matrix construction
- small helper utilities (cube-factor stripping, cache sizing)

Nothing here should parse command-line args or run experiments directly.
"""

import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from math import gcd as _gcd
from pathlib import Path

import psutil

# =============================================================================
# General use functions
# =============================================================================

def selmer_ranks(A, B):
    """
    Convenience wrapper: build the Selmer matrix for (A,B) and return its nullities.

    Returns (left_nullity, right_nullity). This is intended for single-instance use
    (debugging / inspection), not high-throughput enumeration.
    """
    if B == 0:
        raise ValueError("B must be nonzero")

    # The normal form used throughout has B > 0; replacing (A,B) by
    # (-A,-B) preserves the isomorphism class.
    if B < 0:
        A, B = -A, -B

    A, B_fac, B = strip_common_cube_factors(A, list(factor(B)))

    if any([A % 3 == 0, B % 3 == 0]):
        raise NotImplementedError("Curve must have good reduction at 3")

    discriminant_factor = A**3 - 27*B
    if discriminant_factor == 0:
        raise ValueError("The Weierstrass equation is singular")
        
    # Reuse an existing cache when this convenience function is called more
    # than once in the same process.  Reinitializing here would silently throw
    # away all cached cubic-residue computations on every call.
    if _CUBIC_RESIDUE_SYMBOL is None:
        _init_worker_caches(0.5)

    l, m, mat = compute_selmer_matrix(
        B, B_fac, A, discriminant_factor,
        _CUBIC_RESIDUE_SYMBOL, False, False
    )
    M = Matrix(GF(3), l, m, mat)
    return M.left_nullity(), M.right_nullity()


def selmer_sizes(A, B):
    """
    Convenience wrapper: return Selmer group sizes from ranks (powers of 3).
    """
    r, rd = selmer_ranks(A, B)
    return 3 ** r, 3 ** rd
    
    
# =============================================================================
# Aggregation utilities
# =============================================================================

def write_jsonl_atomic(records, output_file):
    """Write JSONL records atomically to ``output_file``.

    Serial runs periodically replace their complete snapshot.  Writing via a
    sibling temporary file prevents an interruption during ``write`` from
    leaving a truncated file that looks like valid completed output.
    """
    output_path = Path(output_file)
    tmp_path = Path(str(output_path) + ".tmp")

    with open(tmp_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    os.replace(tmp_path, output_path)

def aggregate_temp_files(temp_files, output_file):
    """
    Merge a list of temporary batch files into the main output file.

    Each input file is expected to contain one JSON object per line with keys
    "pair", "matrix", and "count". Counts are summed over identical keys.

    The main output file is updated atomically (write to a temporary file and rename),
    and processed temporary files are deleted.
    """
    aggregated = defaultdict(int)

    processed_paths = []
    for temp_path in temp_files:
        with open(temp_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                aggregated[key] += rec["count"]
        processed_paths.append(temp_path)

    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                aggregated[key] += rec["count"]

    records = (
        {"pair": list(pair), "matrix": mat, "count": c}
        for (pair, mat), c in aggregated.items()
    )
    write_jsonl_atomic(records, output_path)

    # Delete a batch only after its counts have been committed atomically.
    for temp_path in processed_paths:
        os.remove(temp_path)


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

    records = (
        {"pair": list(pair), "matrix": mat, "count": c}
        for (pair, mat), c in aggregated.items()
    )
    write_jsonl_atomic(records, output_path)


def prepare_output_file(output_file, overwrite=False):
    """Create the output directory and protect existing data from mixing.

    Independent runs must not silently aggregate into the same file.  Pass
    ``overwrite=True`` explicitly to replace an existing output file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Choose a new --output path or pass --overwrite."
            )
        output_path.unlink()

    return output_path



# =============================================================================
# Cache sizing / initialization (per worker)
# =============================================================================

# These globals are *set* by _init_worker_caches in each worker process.
_CUBE_ROOT_DATA = None
_CUBIC_RESIDUE_CACHED = None
_CUBIC_RESIDUE_SYMBOL = None

def _compute_cache_limits(memory_fraction=0.5, num_processes=1):
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
    memory_fraction = float(memory_fraction)
    num_processes = int(num_processes)
    if not 0 < memory_fraction <= 1:
        raise ValueError("cache_fraction must lie in (0, 1]")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")

    available_bytes_total = psutil.virtual_memory().available
    # budget per worker
    per_worker_bytes = (available_bytes_total * memory_fraction) / num_processes

    bytes_per_cube_root_entry = 200
    bytes_per_cubic_residue_entry = 300

    cube_root_cache_size = max(1, int(per_worker_bytes * 0.25 / bytes_per_cube_root_entry))
    cubic_residue_cache_size = max(1, int(per_worker_bytes * 0.75 / bytes_per_cubic_residue_entry))
    return cubic_residue_cache_size, cube_root_cache_size

def _init_worker_caches(cache_fraction=0.5, num_processes=1, precomputed_root_data=None):
    """Initialize per-process caches and bind fast globals.

    Call once at the beginning of each worker process.
    """
    global cube_root_data, cubic_residue_cached, cubic_residue_symbol
    global _CUBE_ROOT_DATA, _CUBIC_RESIDUE_CACHED, _CUBIC_RESIDUE_SYMBOL

    cubic_residue_cache_size, cube_root_cache_size = _compute_cache_limits(cache_fraction, num_processes)
    
    @lru_cache(maxsize=cube_root_cache_size)
    def cube_root_data_cached(p):
        """Return the nontrivial cube roots of unity modulo p (p ≡ 1 mod 3).

        Returns (z, z^2) where z has exact order 3 mod p.
        """
        # Find a non-cube a mod p; then z = a^((p-1)/3) is a nontrivial cube root of unity.
        p = int(p)
        if p % 3 != 1:
            raise ValueError("cube_root_data requires a prime p congruent to 1 mod 3")

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

    if precomputed_root_data:
        def cube_root_data(p):
            val = precomputed_root_data.get(p)
            return val if val is not None else cube_root_data_cached(p)
        
    else:
        cube_root_data = cube_root_data_cached
            
    @lru_cache(maxsize=cubic_residue_cache_size)
    def cubic_residue_cached(a, p):
        """Compute the cubic residue symbol (a/p)_3 as an element of F_3.

        Output convention:
          0  -> 1 (trivial cube root)
          1  -> z
          2  -> z^2
        where z is a fixed nontrivial cube root of unity mod p.
        For p ≡ 2 (mod 3), the symbol is always trivial (return 0).
        """
        p = int(p)
        a = int(a) % p
        if a == 0:
            raise ValueError("The cubic residue symbol is undefined when p divides a")
        if p % 3 == 2:
            return 0
        if p % 3 != 1:
            raise ValueError("The restricted cubic residue symbol requires p != 3")

        value = pow(a, (p - 1) // 3, p)
        if value == 1:
            return 0

        z, z2 = cube_root_data(p)
        if value == z:
            return 1
        if value == z2:
            return 2
        raise ArithmeticError("Computed value is not a cube root of unity modulo p")

    def cubic_residue_symbol(a, p):
        """Compute (a/p)_3 for any prime p (fast wrapper)."""
        p = int(p)
        if p % 3 == 2:
            return 0
        if p % 3 != 1:
            raise ValueError("The restricted cubic residue symbol requires p != 3")
        return cubic_residue_cached(a, p)

    _CUBE_ROOT_DATA = cube_root_data
    _CUBIC_RESIDUE_CACHED = cubic_residue_cached
    _CUBIC_RESIDUE_SYMBOL = cubic_residue_symbol

    return cube_root_data, cubic_residue_cached, cubic_residue_symbol

# =============================================================================
# Small arithmetic helpers
# =============================================================================

def is_perfect_cube(n) -> bool:
    """
    Exact perfect-cube test for Python ints / Sage Integers.
    Works for negative n as well.
    """
    if n == 0:
        return True
    if n < 0:
        n = -n
        
    _, exact = Integer(n).nth_root(3, truncate_mode=True)
    return bool(exact)
        

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
# Selmer matrix construction (core)
# =============================================================================

def build_selmer_matrix(row_primes, column_primes, num_special_rows,
                        B_value, cubic_residue_symbol,
                        strip_col_index=None):
    """
    Construct the Selmer matrix for the 3-isogeny descent.

    Rows correspond to "check primes" where cubic residue symbols are evaluated.
    Columns correspond to prime powers dividing B (ordered so primes dividing A
    with p ≡ 1 (mod 3) appear first).

    The first `num_special_rows` rows correspond to primes p ≡ 1 (mod 3) dividing
    both A and B, and use the modified diagonal rule appropriate for the dual
    isogeny contribution.

    Parameters
    ----------
    row_primes : list[int]
        Primes indexing rows.
    column_primes : list[int]
        Primes indexing columns.
    num_special_rows : int
        Number of initial rows with special handling.
    B_value : int
        The coefficient B.
    cubic_residue_symbol : callable
        Function (a, p) -> {0,1,2} giving the cubic residue symbol for integer a mod p.
    strip_col_index : int or None
        If supplied, delete this column after constructing the full matrix.  It
        must index a nonzero coordinate of the distinguished kernel vector.

    Returns
    -------
    matrix : list[list[int]]
        Matrix with entries in {0,1,2}, with the specified torsion column
        removed when ``strip_col_index`` is supplied.
    """
    matrix = [[0]*len(column_primes) for _ in range(len(row_primes))]

    for row_index, (check_prime, exponent) in enumerate(row_primes):
        if row_index < num_special_rows:
            # Special rows: primes p ≡ 1 (mod 3) dividing both A and B.
            for col_index, prime in enumerate(column_primes):
                if row_index == col_index:
                    prime_power = prime ** exponent
                    residue = cubic_residue_symbol(B_value // prime_power, check_prime)
                    # Convention swap on the diagonal for the special rows.
                    matrix[row_index][col_index] = {0: 0, 1: 2, 2: 1}[residue]
                else:
                    matrix[row_index][col_index] = cubic_residue_symbol(prime ** exponent, check_prime)
        else:
            # Regular rows: primes from the discriminant.
            # Only test the underlying prime, not its exponent.
            for col_index, prime in enumerate(column_primes):
                matrix[row_index][col_index] = cubic_residue_symbol(prime, check_prime)
    
    if strip_col_index is None:
        return matrix

    if not 0 <= strip_col_index < len(column_primes):
        raise IndexError("strip_col_index is outside the column range")

    return [
        row[:strip_col_index] + row[strip_col_index + 1:]
        for row in matrix
    ]
        
def compute_selmer_matrix(B_value, B_factorization, A_value, discriminant,
                          cubic_residue_symbol, string_matrix=True,
                          strip_col=True, cutoff=None):

    """
    Compute the Selmer matrix for y² + Axy + By = x³.

    The check primes consist of:
      (i) primes p ≡ 1 (mod 3) dividing both A and B,
     (ii) primes p ≡ 1 (mod 3) dividing A^3 - 27B
          that are not already divisors of B.

    The basis columns consist of all prime powers dividing B, ordered so that
    primes dividing A with p ≡ 1 (mod 3) appear first.

    Parameters
    ----------
    B_value : int
    B_factorization : list[list[int,int]]
    A_value : int
    discriminant : int
        The nonzero factor A^3 - 27B (its sign does not affect the matrix).
    cubic_residue_symbol : callable

    Returns
    -------
    (num_rows, num_cols, matrix_string) : tuple[int,int,str]
        If the matrix is discarded due to size, matrix_string is None.
    """
    B_value = int(B_value)
    A_value = int(A_value)
    discriminant = int(discriminant)

    if B_value <= 0:
        raise ValueError("The normalized model must have B > 0")
    if discriminant == 0:
        raise ValueError("The Weierstrass equation is singular")
    if A_value % 3 == 0 or B_value % 3 == 0:
        raise ValueError(
            "The implemented Selmer matrix requires good reduction at 3 "
            "(3 must divide neither A nor B)"
        )

    # Canonicalize the prime basis.  The special columns are placed first
    # below; within each block the primes are in increasing order.
    B_factorization = sorted(
        [(int(prime), int(exponent)) for prime, exponent in B_factorization],
        key=lambda item: item[0],
    )
    if any(prime <= 1 or exponent <= 0 for prime, exponent in B_factorization):
        raise ValueError("B_factorization must contain positive prime powers")
    if len({prime for prime, _ in B_factorization}) != len(B_factorization):
        raise ValueError("B_factorization contains a repeated prime")

    reconstructed_B = 1
    for prime, exponent in B_factorization:
        reconstructed_B *= prime ** exponent
    if reconstructed_B != B_value:
        raise ValueError("B_factorization does not multiply to B_value")

    row_primes = []
    special_columns = []
    remaining_columns = []

    # (i) p ≡ 1 (mod 3) dividing both A and B
    for prime, exponent in B_factorization:
        if (A_value % prime == 0) and (prime % 3 == 1):
            row_primes.append((prime,exponent))
            special_columns.append((prime, exponent))

    num_special_rows = len(row_primes)

    # Remaining primes dividing B
    for prime, exponent in B_factorization:
        if not ((A_value % prime == 0) and (prime % 3 == 1)):
            remaining_columns.append((prime, exponent))

    column_factors = special_columns + remaining_columns
    column_primes = [prime for prime, _ in column_factors]

    # (ii) p ≡ 1 (mod 3) dividing discriminant but not dividing B
    for prime_gen in pari(abs(discriminant)).factor()[0]:
        prime = int(prime_gen)
        if (prime % 3 == 1) and (B_value % prime != 0):
            row_primes.append((prime,None))

    nrows = len(row_primes)
    ncols = len(column_primes)
    
    strip_col_index = None
    if strip_col:
        nonzero_torsion_coordinates = [
            index for index, (_, exponent) in enumerate(column_factors)
            if exponent % 3 != 0
        ]
        if not nonzero_torsion_coordinates:
            raise ValueError(
                "Cannot form the reduced matrix because B is a perfect cube"
            )
        # This is the definition used in the manuscript: remove the last
        # nonzero coordinate of the distinguished vector (v_p(B))_p.
        strip_col_index = nonzero_torsion_coordinates[-1]
        ncols -= 1
    
    if cutoff is not None and nrows * ncols > cutoff:
        return nrows, ncols, None
    
    matrix = build_selmer_matrix(
        row_primes=row_primes,
        column_primes=column_primes,
        num_special_rows=num_special_rows,
        B_value=B_value,
        cubic_residue_symbol=cubic_residue_symbol,
        strip_col_index=strip_col_index,
    )

    if string_matrix:
        matrix_string = "".join(str(int(x)) for row in matrix for x in row)
        return nrows, ncols, matrix_string

    return nrows, ncols, matrix
