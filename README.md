# 3-isogeny Selmer group experiments

SageMath code and matrix-frequency data accompanying **Experiments on 3-isogeny
Selmer groups of elliptic curves with a 3-torsion point**, by Ariel Weiss and
Dongchen Zou.

The code studies the curves

$$
E_{A,B}: y^2 + Axy + By = x^3
$$

and constructs the reduced cubic-residue matrix $M'_{A,B}$ over $\mathbf F_3$
defined in Section 3 of the paper. Section 5 describes two experiments:

- **Sifted height:** enumerate normalized curves in a height interval, excluding
  small primes from the discriminant.
- **Prime factors:** sample curves for which $B$ has a fixed number of distinct
  prime factors drawn from a specified prime pool.

Both experiments save counts of individual matrices in JSONL format.

## Repository contents

| Path | Contents |
| --- | --- |
| `Code/3-selmer-height.sage` | Sifted height experiment (Section 5.1.1) |
| `Code/3-selmer-factor.sage` | Prime factor experiment (Section 5.1.2) |
| `Code/selmer_shared.sage` | Cubic characters, matrix construction, caches, and output helpers |
| `Data/Height Experiment/` | Archived height runs and parameter descriptions |
| `Data/Factor experiment/` | Archived prime factor runs |

## Requirements

Use **SageMath**, not plain Python. The scripts use Sage integer/rational
arithmetic and PARI factorization. The scripts also import `numpy` and `psutil`.

For example, with Conda installed:

```sh
conda create -n selmer -c conda-forge --strict-channel-priority sage numpy psutil
conda activate selmer
```

Other installation options are in the official
[SageMath installation guide](https://doc.sagemath.org/html/en/installation/).
The original software versions were not recorded with the archived data.

Run the commands below **from the `Code` directory**, because both drivers use
`load("selmer_shared.sage")` relative to the current working directory. From the
repository root, first run:

```sh
cd Code
```

The drivers create output directories as needed. Explicit output paths under
`../runs/` keep new computations in the repository's `runs` directory.

## Sifted height experiment

For example, to generate H-1000-100 from Table 1:

```sh
sage 3-selmer-height.sage \
  --min_height 1000 --max_height 1010 --min_prime 100 \
  --nprocesses 4 --output ../runs/H-1000-100.jsonl
```

The paper defines $H(E_{A,B})=\max(|A|^3,B)$. The CLI takes **cube-root height
bounds**: `--min_height h0 --max_height h1` enumerates

$$
h_0^3 \le H(E_{A,B}) \le h_1^3.
$$

Bounds are inclusive. The code uses $B>0$, both signs of nonzero $A$, and rejects
pairs with $p\mid A$ and $p^3\mid B$. It excludes cubes $B$ and $A^3-27B$.
`--min_prime K` requires that no prime strictly below $K$ divide
$B(A^3-27B)$; the prime 3 is always excluded as well.

These are all seven height runs in Table 1:

| Paper ID | `--min_height` | `--max_height` | `--min_prime` | Archived curves |
| --- | ---: | ---: | ---: | ---: |
| H-1000-100 | 1000 | 1010 | 100 | 1,244,605,158 |
| H-1000-300 | 1000 | 1010 | 300 | 816,536,769 |
| H-1000-500 | 1000 | 1010 | 500 | 690,451,279 |
| H-1000-800 | 1000 | 1010 | 800 | 603,045,814 |
| H-1200 | 1200 | 1300 | 500 | 12,638,762,546 |
| H-2250 | 2250 | 2260 | 2000 | 5,267,460,247 |
| H-3250 | 3250 | 3260 | 1000 | 19,442,965,506 |

To generate the other runs, substitute the row's three parameters into the
command above and choose a distinct output filename. These are large
computations: even a narrow height interval scans $B$ up to `max_height**3`.
The archived data can be used directly for analysis without rerunning this step.

## Prime factor experiment

For F1k-2* from Table 5:

```sh
sage 3-selmer-factor.sage \
  --primes 1000 --factors 2 --trials 100000000 --method box \
  --nprocesses 4 --output ../runs/F1k-2.jsonl
```

For F10k-7:

```sh
sage 3-selmer-factor.sage \
  --primes 10000 --factors 7 --trials 124800000 --method height \
  --nprocesses 4 --output ../runs/F10k-7.jsonl
```

- `--primes N` selects the **first N primes greater than 3**, not primes below N.
- `--factors n` selects n distinct primes uniformly without replacement. Their
  positive exponents have distribution $P(e=k)=p^{-(k-1)}-p^{-k}$.
- **Use `--method box` for n = 2**, as specified in Remark 5.1 and denoted by
  asterisks in the paper. Use **`--method height` for n >= 3**.
- `--trials` is the number of **accepted** matrices. Samples are rejected if B or
  $A^3-27B$ is a cube, or if the reduced matrix has more than 12 entries. A is
  resampled until the pair is normalized and 3 does not divide A.

The `height` sampler uses a random sign and a narrow interval near the cube root
of B; `box` uses an interval around zero. The implementation sets
`h = floor(cube_root(B))`: `height` samples magnitudes in
`[int(0.9*h), int(1.1*h))`, and `box` samples A in
`[int(-1.1*h), int(1.1*h))`, rejecting multiples of 3. These bounds use exact
rationals before integer truncation. The default method is `height`, so the
`box` flag must be supplied explicitly for the two-prime runs.

The full parameter list from Table 5 is:

| Paper ID(s) | N | n | Method | Accepted trials per experiment |
| --- | --- | --- | --- | ---: |
| F1k-2*, F10k-2*, F100k-2* | 1000, 10000, 100000 | 2 | `box` | 100,000,000 |
| F1k-3, F10k-3, F100k-3 | 1000, 10000, 100000 | 3 | `height` | 100,000,000 |
| F1k-4, F100k-4 | 1000, 100000 | 4 | `height` | 100,000,000 |
| F10k-4 | 10000 | 4 | `height` | **110,000,000** |
| F1k-5, F100k-5 | 1000, 100000 | 5 | `height` | 100,000,000 |
| F10k-5 | 10000 | 5 | `height` | **100,780,000** |
| F1k-6, F10k-6, F100k-6 | 1000, 10000, 100000 | 6 | `height` | 100,000,000 |
| F10k-7 | 10000 | 7 | `height` | **124,800,000** |
| F1k-8 through F1k-12 | 1000 | 8, 9, 10, 11, 12 | `height` | 100,000,000 |

For another row, substitute N, n, the method, and the accepted-trial count into
the factor command. These are randomized experiments: new runs produce new
matrix counts, not the exact historical realization.

## Command-line options and output handling

Both drivers accept:

| Option | Meaning |
| --- | --- |
| `--nprocesses k` | Worker count; 1 uses the serial path; omission uses the CPU count |
| `--output path` | JSONL destination, relative to the current working directory |
| `--log-every k` | Print progress every k completed batches or serial blocks |
| `--overwrite` | Replace an existing output file; otherwise the script refuses it |
| `--debug` | Enable profiling; unnecessary for the paper runs |

Use separate output paths for independent runs. If `--output` is omitted, files
are named under `data/` in the current working directory. There is no automatic
resume operation: intermediate snapshots contain only the counts completed so
far. Parallel output line order may vary.

The factor driver also provides an `ignore_height` sampling method, which is
not used in the paper. There is no production `--seed`
option; `--debug` seeds the parent process only and does not fix random streams
in spawned workers.

## JSONL format and matrix conventions

Each line represents a distinct reduced matrix and its multiplicity:

```json
{"pair": [2, 3], "matrix": "012120", "count": 57}
```

This example is a 2-by-3 matrix with rows `[0, 1, 2]` and `[1, 2, 0]`, observed
57 times. `matrix` is flattened row by row; entries belong to GF(3). Empty strings
are valid when one dimension is zero. The paper calls those matrices *trivial*;
an all-zero matrix with positive dimensions is not trivial in this terminology.

Columns follow the prime order in Section 3.3. The character convention assigns
value 1 to the smaller nontrivial cube root of unity modulo each row prime.
The reduced matrix removes the last column whose exponent in B is nonzero
modulo 3, as in Definition 3.8.

For a stored a-by-b matrix of rank r over GF(3), Theorem 3.9 and Corollary 3.11 give

$$
\dim\operatorname{Sel}^{\hat\phi}(E'_{A,B})=b-r+1,
\qquad
\dim\operatorname{Sel}^{\phi}(E_{A,B})=a-r.
$$

The optional `selmer_ranks(A, B)` helper computes the full matrix and returns
these two dimensions in the order `(phi_dimension, dual_phi_dimension)`, under
the paper's hypotheses. Unlike the experiment drivers, that helper does not
exclude the exceptional cube families.

## Archived data and the paper's tables

The repository contains all seven height datasets. The four H-1000 files encode
the cutoff in their filenames. The other height parameters are recorded in each
folder's `explanation.txt` and in the height table above.

Factor files normally use `factor_N{N}_n{n}.jsonl`. Two historical filenames are
`factorN_1000_n5.jsonl` and `factorN_1000_n6.jsonl`.
For F10k-7, merge `factor_N10000_n7.jsonl` and `factor_N10000_n7-2.jsonl` by adding
counts for identical `(pair, matrix)` keys. They contain 100,000,000 and
24,800,000 accepted samples, respectively.

**The checked-in factor archive is incomplete relative to the supplied paper:**

| Experiment | Paper total | Checked-in samples | Missing samples |
| --- | ---: | ---: | ---: |
| F10k-4 | 110,000,000 | 100,000,000 | 10,000,000 |
| F10k-5 | 100,780,000 | 100,000,000 | 780,000 |
| F1k-8 through F1k-12 | 100,000,000 each | No files | 100,000,000 each |

The other factor groups have the paper's total sample counts. Exact reproduction
of every factor table requires the missing historical data. The files also do
not record random seeds or the source/software versions that generated them.

| Paper table | Input / operation |
| --- | --- |
| 1 | Totals, trivial counts, and nontrivial counts for all height runs |
| 2 | Nontrivial matrix blocks from H-3250 |
| 3 | Nontrivial matrix blocks from the merged F10k-7 files |
| 4 | Differences between nested H-1000 histograms, as described below |
| 5 | Totals, trivial counts, and nontrivial counts for all factor runs |
| 6 | Selected factor blocks, comparing N = 1000, 10000, 100000 for fixed n |
| 7 | All height block statistics |
| 8 | All factor block statistics |

For Table 4, subtract **counts for matching matrices** before computing any
statistics: cutoff 100 minus 300, 300 minus 500, and 500 minus 800 give the first
three strata. The cutoff-800 histogram gives the final stratum. Subtracting
standard deviations or other summary statistics would not give these results.

### Statistics used in Section 5.2

The experiment scripts generate raw histograms; this repository does not include
the original table-generation script. To turn a histogram into a table, group
records by dimensions a-by-b, use `count` as the weight, and let
$K=3^{ab}$, $T=\sum_M C_M$, and $\mu=T/K$. Include all K possible matrices in the
formulas, assigning count zero to unobserved matrices.

$$
\mathrm{SD}_{\mathrm{obs}}=
\sqrt{\frac{1}{K}\sum_M(C_M-\mu)^2},
\qquad
\mathrm{SD}_{\mathrm{unif}}=
\sqrt{\frac{T}{K}\left(1-\frac{1}{K}\right)},
\qquad
\mathrm{MSE}=\frac{\mathrm{SD}_{\mathrm{obs}}^2}{T^2}.
$$

Report the SD ratio, minimum and maximum counts, and percentage deviations
`100 * (count / mu - 1)`. An unobserved matrix makes the minimum count zero.
The last two columns are the maximum absolute deviations of entry probabilities
from 1/3 and of the rank distribution from that of a uniform random matrix over
GF(3). These are probability differences, not percentages. The paper does not
report goodness-of-fit p-values.

## License

See [LICENSE](LICENSE).
