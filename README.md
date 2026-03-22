# Fast GREML

Fast multi-component GREML for partitioned heritability estimation from pre-computed GRMs.

## Overview

`run_greml.py` provides a wrapper to use GCTA-like flags for REML estimation (with `--method` specifying either exact or stochastic).

`greml.py` provides two AI-REML implementations that accept pre-computed dense GRM matrices (as produced by GCTA `--make-grm`) and return per-component heritability estimates with standard errors.

| Method | Trace computation | Precision | Speed vs GCTA | Use when |
|---|---|---|---|---|
| `greml_stochastic` | Hutchinson (s=50 probes) | float32 Chol | ~20× | n > 3k or many runs |
| `greml_exact` | Explicit V⁻¹ / eigensystem for K=1 | float64 | ~2× | n ≤ 6k, exact reference |

Both methods use an HE-regression warm start that reduces AI-REML iterations from ~11 (GCTA's EM initialisation) to ~4, and compute the AI matrix exactly (pure quadratic forms) so SE estimates are identical to GCTA's Cramér–Rao bound regardless of which method is used. Intercept-only REML is handled with a specialized exact projection path. In the Python API, `greml_exact` uses a one-time eigendecomposition automatically when `K=1`, reusing that eigensystem on repeated fits of the same GRM object; the CLI keeps the dense exact path by default and enables the eigen-based path only with `--eigendecomp`. `greml_stochastic` can optionally use a small compiled kernel for the GRM matvec/trace loop on macOS.

Empirical speedups at n=5,000, K=3 components, single thread:

```
GCTA --reml-no-lrt      25s    1×
greml_exact             13s    2×
greml_stochastic        1.4s   18×
```

## Requirements

- Python ≥ 3.9
- NumPy
- SciPy

```bash
pip install numpy scipy
```

No other dependencies. The code calls LAPACK (via SciPy) directly; on Apple Silicon it uses the Accelerate framework automatically through NumPy/SciPy.

### Optional compiled stochastic kernel

The stochastic path will use `_greml_accel` automatically if it is available. Build it in-place with:

```bash
python3 setup.py build_ext --inplace
```

This extension uses NumPy's C API plus Apple's Accelerate BLAS and falls back to the pure-Python/NumPy implementation if it is not built.

For exact one-component runs, the CLI uses the dense exact path by default. Pass `--eigendecomp` to enable the eigen-based exact path; when enabled, the CLI reports when it computes or reuses the eigendecomposition. In the Python API, `greml_exact([K], y)` uses the eigen-based path by default; pass `use_eigendecomp=False` to force the dense exact path instead.

## Usage

### Reading GRMs

```python
from greml import read_grm

# Read GCTA binary GRM files ({prefix}.grm.bin + {prefix}.grm.id)
K1, ids = read_grm("path/to/grm_bucket1")
K2, _   = read_grm("path/to/grm_bucket2")
K3, _   = read_grm("path/to/grm_bucket3")
```

`read_grm` returns a dense symmetric `(n, n)` matrix. Pass `dtype=np.float32` to halve memory (GRMs are stored as float32 on disk anyway).

### Running GREML

```python
import numpy as np
from greml import greml_stochastic, greml_exact

# Phenotype: shape (n,), should be standardised
y = (raw_pheno - raw_pheno.mean()) / raw_pheno.std()

K_list = [K1, K2, K3]

# Fast stochastic (recommended for n > 3k)
result = greml_stochastic(K_list, y)

# Exact reference (recommended for n ≤ 6k)
result = greml_exact(K_list, y)

# Single-component exact GREML uses the eigen-based path by default
result = greml_exact([K1], y)

# Force the dense exact path for single-component GREML
result = greml_exact([K1], y, use_eigendecomp=False)

print(result['h2'])    # array([h2_1, h2_2, h2_3])
print(result['se'])    # array([se_1, se_2, se_3])
print(result['n_iter'])
```

### Command-line usage

```bash
# Dense exact path (default)
python3 run_greml.py --grm path/to/grm --pheno pheno.txt --method exact

# Enable the eigen-based single-component exact path
python3 run_greml.py --grm path/to/grm --pheno pheno.txt --method exact --eigendecomp
```

### Return value

Both functions return a dict:

| Key | Type | Description |
|---|---|---|
| `h2` | `ndarray (K,)` | Heritability per component: θ_k / Σθ |
| `se` | `ndarray (K,)` | SE via delta method from AI⁻¹ |
| `se_theta` | `ndarray (K+1,)` | SE of the raw variance components |
| `se_total_h2` | `float` | SE of total genetic variance fraction: Σ_k θ_k / Σθ |
| `theta` | `ndarray (K+1,)` | Raw variance components σ²_1,…,σ²_e |
| `n_iter` | `int` | AI-REML iterations taken |

### Key parameters for `greml_stochastic`

| Parameter | Default | Notes |
|---|---|---|
| `n_probes` | `50` | Hutchinson probe vectors. Antithetic pairs are used when possible; odd counts add one independent probe. 50 → ~1% trace error; 30 is sufficient for standardised phenotypes |
| `dtype` | `np.float32` | `float32` is ~2.85× faster on arm64/Accelerate. Safe because κ(V) ≈ 4–5 for genetics GRMs |
| `mean_correct` | `True` | Apply full REML mean projection. Set `False` for a small speedup when y is standardised (correction is O(1/n)) |
| `seed` | `42` | RNG seed for reproducible probe vectors |

## Algorithm notes

**AI-REML** (Gilmour et al. 1995) iterates:

```
θ ← θ + AI⁻¹ · g
```

where `g_k = ½(yᵀ P K_k P y − tr(P K_k))` is the REML score and `AI[j,k] = ½ yᵀ P K_j P K_k P y` is the average information matrix.

**Stochastic traces** (Hutchinson 1990): `tr(V⁻¹ K_k) ≈ (1/s) Σᵢ zᵢᵀ Kₖ (V⁻¹ zᵢ)` with antithetic Rademacher probes. This replaces the O(n³) V⁻¹ materialisation with O(s·n²) triangular solves, batched into a single BLAS-3 call.

**Exact traces** in `greml_exact`: for the dense multi-component path, traces are computed from the explicit inverse `V⁻¹` formed from the Cholesky factor. For the single-component path, the model is diagonalised once and exact traces are evaluated in the rotated basis.

**The AI matrix is always computed exactly** (as quadratic forms `(K_k Py)ᵀ V⁻¹ (K_j Py)`) in both methods, so SE estimates are at the Cramér–Rao bound regardless of `n_probes`.
