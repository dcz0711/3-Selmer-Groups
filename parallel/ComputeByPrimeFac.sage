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


# ============================================================
# Eisenstein integer arithmetic using tuples (a, b) = a + bω
# Converts to python int arithmetic for speed
# ============================================================

def multiply_eisenstein(x, y):
    """
    Multiply two Eisenstein integers x = a + bω and y = c + dω.
    
    Uses the relation ω² = -1 - ω to reduce products.
    
    Args:
        x: tuple (a, b) representing a + bω
        y: tuple (c, d) representing c + dω
        
    Returns:
        tuple: Product as (real, omega_coeff)
    """
    a, b = int(x[0]), int(x[1])
    c, d = int(y[0]), int(y[1])
    bd = b * d
    return (int(a*c - bd), int(a*d + b*c - bd))
    
def eisenstein_norm(x):
    """
    Compute the norm N(a+bω) = a²-ab+b².
    
    Args:
        x: (a, b) representing a + bω
    
    Returns:
        Non-negative integer norm
    """
    a, b = x
    return int(a*a - a*b + b*b)


def divide_and_round(z, pi):
    """
    Find the nearest Eisenstein integer to z/π.
    
    This implements exact division with rounding in Z[ω].
    We compute z/π by multiplying by the conjugate of π,
    then rounding each component to the nearest integer.
    
    The conjugate of c+dω is (c-d)-dω, with norm c²-cd+d².
    
    Args:
        z: (a, b) dividend
        pi: (c, d) divisor
    
    Returns:
        (q0, q1) the quotient rounded to nearest
    """
    a, b = int(z[0]), int(z[1])
    c, d = int(pi[0]), int(pi[1])
    norm_pi = eisenstein_norm(pi)
    
    # Multiply z by conjugate of π: (c-d)-dω
    # Real part: a(c-d) + b·d
    # ω part: bc - ad
    num0 = a * (c - d) + b * d
    num1 = b * c - a * d
    
    # Round to nearest integer (add half before integer division)
    norm_half = norm_pi >> 1  # Bit shift is faster than // 2
    q0 = int((num0 + norm_half) // norm_pi)
    q1 = int((num1 + norm_half) // norm_pi)
    
    return (q0, q1)

def mod_pi(z, pi):
    """
    Reduce z modulo pi in Z[ω].
    
    Computes the unique representative r with z ≡ r (mod pi)
    where r is "small" (nearest to zero).
    
    Algorithm: z mod pi = z - q·pi where q = round(z/pi)
    
    Args:
        z: (a, b) value to reduce
        pi: (c, d) modulus
    
    Returns:
        z reduced modulo pi
    """
    q0, q1 = divide_and_round(z, pi)
    c, d = int(pi[0]), int(pi[1])
    c_minus_d = c - d 
    
    return (
        int(z[0] - q0*c + q1*d),
        int(z[1] - q0*d - q1*(c-d))
    )
    
def power_mod_pi(a, exp, pi):
    """
    Compute a^exp mod π using binary exponentiation.
    
    Args:
        a: int base
        exp: Integer exponent (handles both int and Sage Integer)
        pi: (c, d) modulus in Z[ω]
    
    Returns:
        a^exp mod π as Eisenstein integer (tuple)
    """
    exp = int(exp)
    a = int(a)
    pi = (int(pi[0]), int(pi[1]))
    
    # Handle base cases for efficiency
    if exp == 0:
        return (int(1), int(0))  # a^0 = 1
    if exp == 1:
        return mod_pi((a, 0), pi)  # a^1 = a mod π
    
    result = (1, 0)  # Start with 1
    base = mod_pi((a, 0), pi)  # Reduce a mod π once
    
    # Binary exponentiation loop
    while exp:
        if exp & 1:  # If lowest bit is 1
            result = mod_pi(multiply_eisenstein(result, base), pi)
        exp >>= 1  # Shift right (divide by 2)
        if exp:  # Skip final squaring
            base = mod_pi(multiply_eisenstein(base, base), pi)
    
    return result

# ============================================================
# CUBIC RESIDUE SYMBOL
# Determines whether an integer a is a cubic residue modulo p
# ============================================================

def prime_above_uncached(p):
    """
    Compute a generator for a prime ideal above p in Z[ω].
    
    For p ≡ 2 (mod 3): p is inert (remains prime), use (p, 0)
    For p ≡ 1 (mod 3): p splits into two conjugate primes,
                       we pick one by factoring in Sage
    
    Args:
        p: Prime number
    
    Returns:
        (a, b) representing a+bω, a generator of prime above p
    """
    # Inert case: p stays prime in Z[ω]
    if p % 3 == 2:
        return (int(p), int(0))
    
    # Split case: factor p in Z[ω]
    K.<ω> = CyclotomicField(3)
    O = K.ring_of_integers()
    
    # p factors as π·π̄ where π and π̄ are conjugate
    # We take the first factor
    factor = O(p).factor()[0][0]
    return (int(factor[0]), int(factor[1]))

def cubic_residue_uncached(a, p, pi=None):
    """
    Compute the cubic residue symbol (a/p)₃ in Z[ω] where a is an integer.
    
    The cubic residue symbol tells us which cube root of unity
    a^((p-1)/3) equals modulo π:
        0 → a^((p-1)/3) ≡ 1   (a is a cubic residue)
        1 → a^((p-1)/3) ≡ ω   (a is a cubic non-residue)
        2 → a^((p-1)/3) ≡ ω²  (a is a cubic non-residue)
    
    For p ≡ 2 (mod 3): All non-zero integers a are cubic residues
    For p ≡ 1 (mod 3): Exactly 1/3 of non-zero elements are residues
    
    Args:
        a: Integer to test (must not be divisible by p)
        p: Prime modulus
        pi: Generator of prime above p (required for p ≡ 1 mod 3)
    
    Returns:
        0, 1, or 2 representing the cubic character
    
    Raises:
        ValueError: If a divisible by p or π missing when needed
        RuntimeError: If computation gives unexpected result
    """
    if a % p == 0:
        raise ValueError("a divisible by p")

    # Inert prime case: p ≡ 2 (mod 3)
    # Since p is inert, a mod p is an element of F_p, and all elements
    # of F_p are cubes.
    if p % 3 == 2:
        return 0

    # Split prime case: p ≡ 1 (mod 3)
    if pi is None:
        raise ValueError("π must be supplied for p ≡ 1 mod 3")

    # Ensure π is a valid prime above p
    if eisenstein_norm(pi) != p:
        raise ValueError("π does not have norm p")

    # Compute a^((p-1)/3) mod π
    exponent = (p - 1) // 3
    residue = power_mod_pi(a, exponent, pi)

    # Map the result to 0, 1, or 2
    if residue == (1, 0):      # Result is 1
        return 0
    elif residue == (0, 1):    # Result is ω
        return 1
    elif residue == (-1, -1):  # Result is ω² = -1-ω
        return 2
    else:
        raise RuntimeError(f"Invalid residue value {residue} for a={a}, p={p}, π={pi}")


# ============================================================
# RANDOM SAMPLING
# Generate random (A,B) pairs with controlled prime structure
# ============================================================

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


def generate_random_a(b_value, delta=0.1, method="height"):
    """
    Generate random A near the "height" H = B^(1/3), coprime to 3.
    
    Three sampling methods:
    - "height": Sample |A| ∈ [(1-δ)H, (1+δ)H] with random sign
    - "box": Sample A ∈ [-(1+δ)H, (1+δ)H]
    - "ignore height": Sample A ∈ [-B, B]
    
    Args:
        b_value: Value of B (determines height)
        delta: Relative width of sampling interval
        method: Sampling method
    
    Returns:
        Random integer A with gcd(A, 3) = 1
    """
    height = int(b_value ** (1 / 3))
        
    if method == "box":
        # Symmetric interval around 0
        low = int(-(1 + delta) * height)
        high = int((1 + delta) * height)
        sign = 1
    elif method == "ignore height":
        # Full range up to B
        low = int(-b_value)
        high = int(b_value)
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
    if result % 3 != 0:
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

def strip_cube_factors(a, b_factorization):
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
    
    Args:
        a: Integer A
        b_factorization: List of [prime, exponent] for B
    
    Returns:
        (reduced_a, reduced_b_factorization, reduced_b)
    """
    reduced_b_factorization = []
    reduced_b = 1
    
    for p, exp in b_factorization:
        # Can't apply transformation if exp < 3 or p doesn't divide A
        if exp < 3 or a % p != 0:
            reduced_b_factorization.append([p, exp])
            reduced_b *= p^exp
            continue
        
        # How many times does p divide A?
        p_power_in_a = valuation(a, p)
        
        # Apply transformation as many times as possible
        # Limited by both p^k || A and p^(3k) || B
        num_reductions = min(p_power_in_a, exp // 3)
        
        # Perform the reductions
        a //= p^num_reductions
        remaining_exp = exp - 3 * num_reductions
        
        # Store what's left of this prime in B
        if remaining_exp > 0:
            reduced_b_factorization.append([p, remaining_exp])
            reduced_b *= p^remaining_exp
    
    return a, reduced_b_factorization, reduced_b



def generate_random_a_b_pair(primes, num_factors, method="height"):
    """
    Generate a random (A,B) pair already in reduced form.
    
    Strategy:
    1. Build B from random primes with geometric exponents
    2. Sample A near height B^(1/3)
    3. Check if strip_cube_factors would reduce it
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
    
    # Rejection sampling: keep trying until we get reduced pair
    while True:
        a = generate_random_a(b_value, method=method)
        
        # Test if this (a,b) is already in reduced form
        reduced_a, reduced_b_fac, reduced_b = strip_cube_factors(a, b_factorization)
        
        # Accept only if B was reduced (already in canonical form). 
        # This ensures B has exactly num_factors prime factors
        if reduced_b == b_value:
            return reduced_a, reduced_b_fac, reduced_b
        # Otherwise, loop and try new A


# ============================================================
# SELMER MATRIX CONSTRUCTION
# Build matrix whose nullspace gives the Selmer group
# ============================================================

def build_selmer_matrix(check_primes, basis_primes, num_dividing_a, 
                        b_value, cubic_residue_func):
    """
    Build the matrix M where ker(M) ⊗ Z/3Z ≅ Sel_φ(E).
    
    Matrix structure:
    - Rows: check primes (where we evaluate cubic residues)
    - Columns: basis primes (the elements being tested)
    - Entry M[i,j]: cubic residue symbol of basis[j] at check[i]
    
    Special structure:
    - First num_dividing_a rows are primes dividing both A and B
    - These use modified entries for the dual isogeny
    - Remaining rows are primes from the discriminant
    
    Args:
        check_primes: Primes for rows (evaluation points)
        basis_primes: Prime powers for columns (basis elements)
        num_dividing_a: Number of primes ≡ 1 (mod 3) dividing A
        b_value: The value B
        cubic_residue_func: Function to compute (a/p)₃
    
    Returns:
        (num_rows, num_cols_after_trim, trimmed_matrix)
    """
    num_rows = len(check_primes)
    num_cols = len(basis_primes)
    
    # There are too many n x m matrices to see any meaningful distribution
    if num_rows * num_cols > 12:
        return num_rows, num_cols, None
    
    matrix = np.empty((num_rows, num_cols), dtype=int)
    
    for i in range(num_rows):
        if i < num_dividing_a:
            # Special rows: primes ≡ 1 (mod 3) dividing A
            for j in range(num_cols):
                p, exp = basis_primes[j]
                
                if i == j:
                    q = p ** exp
                    residue = cubic_residue_func(b_value / q, check_primes[i])

                    matrix[i][j] = {0: 0, 1: 2, 2: 1}[residue]
                else:
                    matrix[i][j] = cubic_residue_func(p ** exp, check_primes[i])
        else:
            # Regular rows: primes from discriminant
            # Only test the prime itself, not the power
            for j in range(num_cols):
                p, _ = basis_primes[j]
                matrix[i][j] = cubic_residue_func(p, check_primes[i])
    
    # Remove last column: its contribution to the nullspace corresponds
    # to the torsion point.
    # The nullspace dimension over Z/3Z gives the Selmer rank
    return num_rows, num_cols - 1, np.delete(matrix, -1, axis=1)


def compute_selmer_matrix(b_value, b_factorization, a_value, cubic_residue_func):
    """
    Construct the complete Selmer matrix for y² + Axy + By = x³.
    
    The Selmer group Sel_φ(E) is the kernel of a map from a certain
    group to a direct sum of F_3's (one for each check prime).
    This matrix represents that map.
    
    Construction:
    1. Identify check primes (rows):
       - Primes ≡ 1 (mod 3) dividing both A and B
       - Primes ≡ 1 (mod 3) dividing discriminant Δ = 27B - A³
    
    2. Identify basis primes (columns):
       - All primes dividing B, with their exponents
       - Ordered so primes dividing A come first
    
    3. Fill matrix with cubic residue symbols
    
    Args:
        b_value: Integer B
        b_factorization: List of [prime, exponent] for B
        a_value: Integer A
        cubic_residue_func: Function to compute (a/p)₃
    
    Returns:
        (num_rows, num_cols, matrix_as_string)
    """
    # Compute discriminant for additional check primes
    discriminant = 27 * b_value - a_value ** 3
    
    check_primes = []
    basis_primes = []
    
    # FIRST: Primes ≡ 1 (mod 3) dividing both A and B
    for p, exp in b_factorization:
        if a_value % p == 0 and p % 3 == 1:
            check_primes.append(p)
            basis_primes.append([p, exp])
    
    num_dividing_a = len(check_primes)
    
    # SECOND: All other primes from B
    for p, exp in b_factorization:
        if not (a_value % p == 0 and p % 3 == 1):
            basis_primes.append([p, exp])
    
    # THIRD: Primes ≡ 1 (mod 3) from discriminant (not already in B)
    for p, _ in discriminant.factor():
        if p % 3 == 1 and b_value % p != 0:
            check_primes.append(p)
    
    # Build the matrix
    num_rows, num_cols, matrix = build_selmer_matrix(
        check_primes, basis_primes, num_dividing_a, 
        b_value, cubic_residue_func
    )
    
    # Flatten to string for compact storage
    # Each entry is a single digit (0, 1, or 2)
    matrix_string = ''.join(str(element) for row in matrix for element in row)
    
    return int(num_rows), int(num_cols), matrix_string

def compute_random_instance(primes, num_factors, cubic_residue_func, method="height"):
    """
    Generate one random curve instance and compute its Selmer matrix.
    
    This is the main computational unit: given a pool of primes,
    we randomly sample a curve and compute its invariants.
    
    Args:
        primes: Pool of available primes
        num_factors: Number of prime factors for B
        cubic_residue_func: Function to compute cubic residues
        method: Sampling method for A
    
    Returns:
        (num_rows, num_cols, matrix_string) characterizing the Selmer group
    """
    matrix_str = None
    
    while matrix_str == None:
        # Generate random curve parameters
        a, b_factorization, b = generate_random_a_b_pair(primes, num_factors, method)
        
        num_rows, num_cols, matrix_str = compute_selmer_matrix(b, b_factorization, a, cubic_residue_func)
    
    return num_rows, num_cols, matrix_str

# ============================================================
# MEMORY MANAGEMENT
# Estimate cache sizes based on available RAM
# ============================================================

def compute_cache_limits(memory_fraction=0.5):
    """
    Estimate appropriate LRU cache sizes based on available RAM.
    
    Conservative estimates to avoid memory pressure in multiprocessing:
    - Use only a fraction of available memory
    - Split between prime_above cache (25%) and cubic_residue cache (75%)
    - Rough per-entry estimates: 200 bytes for primes, 300 for residues
    
    Args:
        memory_fraction: Fraction of available RAM to use (default 0.5)
    
    Returns:
        (cubic_residue_cache_size, prime_above_cache_size)
    """
    # Get available memory in bytes
    available_bytes = psutil.virtual_memory().available * memory_fraction

    # Rough estimates of memory per cache entry
    bytes_per_prime_entry = 200
    bytes_per_cubic_residue_entry = 300

    # Allocate 25% to prime_above cache, 75% to cubic_residue cache
    # (cubic residue is called more frequently)
    prime_cache_size = int(available_bytes * 0.25 / bytes_per_prime_entry)
    cubic_residue_cache_size = int(available_bytes * 0.75 / bytes_per_cubic_residue_entry)

    return cubic_residue_cache_size, prime_cache_size

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
    ) = args

    # Initialize Sage objects (must be done in each worker)
    K.<ω> = CyclotomicField(3)
    O = K.ring_of_integers()
    
    # Compute cache sizes based on available memory
    cubic_residue_cache_size, prime_cache_size = compute_cache_limits(cache_fraction)
    
    # ===== CACHE LAYER 1: Prime ideals above p =====
    @lru_cache(maxsize=prime_cache_size)
    def _prime_above_cached(p):
        """Compute prime above p (for values not in precomputed dict)."""
        # This is only called for p ≡ 1 (mod 3)
        factor = O(p).factor()[0][0]
        return (int(factor[0]), int(factor[1]))
    
    def prime_above(p):
        """
        Get prime above p, using precomputed dict when possible.
        
        Fast path for p ≡ 2 (mod 3): return (p,0) immediately.
        For p ≡ 1 (mod 3): check precomputed dict, then cache.
        """
        if p % 3 == 2:
            return (int(p), int(0))
        return prime_above_precomputed.get(p) or _prime_above_cached(p)
    
    # ===== CACHE LAYER 2: Cubic residue symbols =====
    @lru_cache(maxsize=cubic_residue_cache_size)
    def cubic_residue_cached(a, p):
        """Compute cubic residue symbol with LRU caching."""
        pi = prime_above(p)
        return cubic_residue_uncached(a, p, pi)
    
    def cubic_residue(a, p):
        """
        Compute cubic residue symbol with fast path for p ≡ 2 (mod 3).
        
        For p ≡ 2 (mod 3): Always return 0 (all integers are residues).
        For p ≡ 1 (mod 3): Use cached computation.
        """
        if p % 3 == 2:
            return 0
        return cubic_residue_cached(a, p)
    
    # ===== MAIN COMPUTATION LOOP =====
    counts = defaultdict(int)

    for _ in range(num_iterations):
        # Compute Selmer matrix for one random instance
        key = compute_random_instance(primes, num_factors, cubic_residue, method)
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
# Aggregation
# ============================================================

def aggregate_temp_files(temp_files, output_file):
    """
    Aggregate temporary files into the main output file.
    
    Process:
    1. Read all temp files and accumulate counts
    2. Merge with existing output file if present
    3. Write aggregated results atomically (using temp file + rename)
    4. Delete processed temp files
    
    This is called periodically during computation to avoid
    accumulating too many temp files.
    
    Args:
        temp_files: List of temporary file paths
        output_file: Path to main output file
    """
    aggregated = defaultdict(int)

    for tf in temp_files:
        with open(tf) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                aggregated[key] += rec["count"]
        os.remove(tf)

    if Path(output_file).exists():
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                key = (tuple(rec["pair"]), rec["matrix"])
                aggregated[key] += rec["count"]

    tmp = str(output_file) + ".tmp"
    with open(tmp, "w") as f:
        for (pair, mat), c in aggregated.items():
            f.write(json.dumps({
                "pair": list(pair),
                "matrix": mat,
                "count": c
            }) + "\n")

    os.replace(tmp, output_file)

def aggregate_main_file(output_file):
    aggregated = defaultdict(int)

    with open(output_file) as f:
        for line in f:
            rec = json.loads(line)
            key = (tuple(rec["pair"]), rec["matrix"])
            aggregated[key] += rec["count"]

    with open(output_file, "w") as f:
        for (pair, mat), c in aggregated.items():
            f.write(json.dumps({
                "pair": list(pair),
                "matrix": mat,
                "count": c
            }) + "\n")

# ============================================================
# Parallel driver
# ============================================================

def compute_prime_above_batch(primes_batch):
    """Worker function to compute prime_above for a batch of primes."""
    return {p: prime_above_uncached(p) for p in primes_batch}
    
def compute_single_prime(p):
    """Compute for single prime - minimal data transfer."""
    return (p, prime_above_uncached(p))



def run_parallel(primes, num_trials, num_factors, output_file, method="height", 
                 batch_size=2000, aggregate_every=10, cache_fraction=0.5, num_processes=None):
    """Parallel computation across workers."""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    if num_processes == 1:
        return run_non_parallel(primes, num_trials, num_factors, output_file, 
                                method, cache_fraction)
    
    # ============================================================
    # Parallelize prime_above_cache precomputation
    # ============================================================
    primes_to_cache = [p for p in primes if p % 3 == 1]
    
    if primes_to_cache:
        print(f"Precomputing prime_above_cache for {len(primes_to_cache)} primes...")
        
        
        # Use imap for lazy iteration
        prime_above_cache = {}
        with mp.Pool(processes=num_processes) as pool:
            for p, result in pool.imap(compute_single_prime, primes_to_cache, chunksize=100):
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
         prime_above_cache, method, cache_fraction)
        for i in range(0, num_trials, batch_size)
    ]
    
    # Process batches in parallel
    temp_files = []
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
    
    # Final aggregation of remaining files
    if temp_files:
        aggregate_temp_files(temp_files, output_path)
    
    print("Finished all batches.")


def run_non_parallel(primes, num_trials, num_factors, output_file,
                     method="height", cache_fraction=0.5):
    """Single-process computation."""
     
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)     
    
    K.<ω> = CyclotomicField(3)
    O = K.ring_of_integers()  
    
    CR_MAX, PRIME_MAX = compute_cache_limits(cache_fraction)
    
    # ---------------- prime_above ----------------
    @lru_cache(maxsize=PRIME_MAX)
    def _prime_above_cached(p):
        # only called for values NOT in the precomputed table
        # prime must be 1 mod 3

        factor = O(p).factor()[0][0]
        return (int(factor[0]), int(factor[1]))

    def prime_above(p):
        '''Return the precomputed value or compute it'''
        if p % 3 == 2:
            return (int(p), int(0))
        
        return _prime_above_cached(p)

    @lru_cache(maxsize=CR_MAX)
    def cubic_residue_cached(a, p):
        pi = prime_above(p)
        
        return cubic_residue_uncached(a, p, pi)
    
    def cubic_residue(a, p):
        if p & 3 == 2:
            return 0
            
        return cubic_residue_cached(a, p)

    counts = defaultdict(int)
    log_every = 2000
    aggregate_every = 20000
    
    for i in range(0, num_trials, log_every):
        for _ in range(min(log_every, num_trials-i)):
            key = compute_random_instance(
                primes, num_factors, cubic_residue, method
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

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, required=True)
    parser.add_argument("--primes", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--method", type=str, default="height")
    parser.add_argument("--nprocesses", type=int, default=None)
    args = parser.parse_args()
    
    primes = primes_first_n(args.primes + 2)[2:]  # Skip 2,3


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
    

