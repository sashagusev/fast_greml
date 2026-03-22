"""
greml.py — fast multi-component GREML from pre-computed GRMs
=============================================================
Two implementations of Average Information REML (AI-REML):

  greml_stochastic  – Hutchinson stochastic trace, float32 Cholesky
                      ~20× faster than GCTA at n=5k, exact SE
  greml_exact       – explicit V⁻¹ traces, float64
                      ~2× faster than GCTA at n=5k, identical results

Both accept a list of pre-computed dense GRM matrices (e.g. from GCTA
--make-grm) and a phenotype vector, and return h² estimates with SE.

Key algorithmic choices
-----------------------
1. HE warm start     — reduces AI-REML iterations from ~11 (GCTA EM) to ~4
2. Stochastic traces — O(s·n²) Hutchinson vs O(n³) exact; s=50 gives ~1%
                       relative error, well within the SE of estimates
3. Antithetic probes — halves Hutchinson variance at no extra cost
4. Batched dpotrs    — all RHS solved in one BLAS-3 call per iteration
5. float32 Cholesky  — 2.85× faster than float64 on arm64/Accelerate;
                       safe because κ(V) ≈ 4–5 for typical genetics GRMs
6. Exact AI matrix   — computed as pure quadratic forms regardless of
                       n_probes, so SE is exact in both implementations

References
----------
Gilmour et al. 1995 — Average Information REML
Hutchinson 1990     — stochastic trace estimator
Yang et al. 2011    — partitioned GREML (GCTA)
Avron & Toledo 2011 — variance analysis of Hutchinson estimator
"""

import numpy as np
import weakref
from scipy import linalg
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import get_lapack_funcs

_TINY = 1e-14
_SINGLE_GRM_EIG_CACHE = {}

try:
    from _greml_accel import stochastic_ops as _stochastic_ops_accel
except ImportError:
    _stochastic_ops_accel = None


# ── GRM I/O ────────────────────────────────────────────────────────────────────

def read_grm(prefix, dtype=np.float64):
    """Read a GCTA binary GRM into a dense symmetric matrix.

    Parameters
    ----------
    prefix : str or Path
        Path prefix shared by ``{prefix}.grm.bin`` and ``{prefix}.grm.id``.
    dtype : numpy dtype, optional
        Output dtype. Default float64; use float32 to halve memory.

    Returns
    -------
    K : ndarray, shape (n, n)
    ids : list of [FID, IID] pairs
    """
    id_file  = str(prefix) + ".grm.id"
    bin_file = str(prefix) + ".grm.bin"
    with open(id_file) as f:
        ids = [ln.strip().split() for ln in f]
    n   = len(ids)
    raw = np.fromfile(bin_file, dtype=np.float32)
    if len(raw) != n * (n + 1) // 2:
        raise ValueError(f"GRM size mismatch: expected {n*(n+1)//2} values, "
                         f"got {len(raw)} in {bin_file}")
    if not np.isfinite(raw).all():
        raise ValueError(f"GRM {bin_file} contains non-finite values.")
    K  = np.zeros((n, n), dtype=dtype)
    lo = 0
    for i in range(n):
        K[i, :i + 1] = raw[lo:lo + i + 1]
        K[:i + 1, i] = K[i, :i + 1]
        lo += i + 1
    return K, ids


# ── Shared internals ───────────────────────────────────────────────────────────

def _he_warmstart(K_list, y):
    """HE-regression point estimate used to initialise AI-REML."""
    K   = len(K_list)
    dK  = [np.diag(Kk) for Kk in K_list]
    Ky  = [Kk @ y for Kk in K_list]
    y2  = y ** 2
    M   = np.empty((K, K))
    b   = np.empty(K)
    for a in range(K):
        b[a] = np.dot(y, Ky[a]) - np.dot(dK[a], y2)
        for bb in range(K):
            M[a, bb] = (np.einsum('ij,ij->', K_list[a], K_list[bb])
                        - np.dot(dK[a], dK[bb]))
    h0        = np.clip(np.linalg.lstsq(M, b, rcond=None)[0], 0.02, 0.90)
    theta     = np.empty(K + 1)
    theta[:K] = h0
    theta[K]  = max(1.0 - h0.sum(), 0.05)
    return theta


def _validate_inputs(K_list, y, X=None, n_probes=None):
    """Basic shape and finite-value checks for public GREML entry points."""
    if not K_list:
        raise ValueError("K_list must contain at least one GRM.")

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D phenotype vector.")
    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values.")

    n = y.shape[0]
    for idx, Kk in enumerate(K_list, start=1):
        Kk = np.asarray(Kk)
        if Kk.shape != (n, n):
            raise ValueError(
                f"GRM {idx} has shape {Kk.shape}; expected ({n}, {n})."
            )
        if not np.isfinite(Kk).all():
            raise ValueError(f"GRM {idx} contains non-finite values.")

    if X is not None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] != n:
            raise ValueError(f"X has shape {X.shape}; expected ({n}, q).")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")

    if n_probes is not None and n_probes < 1:
        raise ValueError("n_probes must be at least 1.")


def _stochastic_component_ops(Kd, Py, VinvZ, Z, Vinv1=None):
    """Compute KPy and stochastic trace terms, optionally with u'K_ku."""
    if _stochastic_ops_accel is not None:
        try:
            return _stochastic_ops_accel(
                Kd,
                np.ascontiguousarray(Py),
                np.ascontiguousarray(VinvZ),
                np.ascontiguousarray(Z),
                None if Vinv1 is None else np.ascontiguousarray(Vinv1),
            )
        except (TypeError, ValueError):
            pass

    K = len(Kd)
    s = Z.shape[1]
    KPy = np.empty((Py.shape[0], K), dtype=Py.dtype)
    traces = np.empty(K, dtype=np.float64)
    uKu = np.empty(K, dtype=np.float64) if Vinv1 is not None else None
    Vinv1_64 = None if Vinv1 is None else Vinv1.astype(np.float64, copy=False)
    for k, Kk in enumerate(Kd):
        KPy[:, k] = Kk @ Py
        traces[k] = float((Z * (Kk @ VinvZ)).sum(dtype=np.float64)) / s
        if Vinv1 is not None:
            Ku = Kk @ Vinv1
            uKu[k] = float(np.dot(Vinv1_64, Ku.astype(np.float64, copy=False)))
    return KPy, traces, uKu


def _get_single_grm_eigendecomp(K):
    """Return cached eigendecomposition for a single-GRM exact fit."""
    key = id(K)
    cached = _SINGLE_GRM_EIG_CACHE.get(key)
    if cached is not None:
        ref, eigvals, U = cached
        if ref() is K:
            return eigvals, U, True
        _SINGLE_GRM_EIG_CACHE.pop(key, None)

    Kd = np.asarray(K, dtype=np.float64)
    eigvals, U = np.linalg.eigh(Kd)
    try:
        _SINGLE_GRM_EIG_CACHE[key] = (
            weakref.ref(K, lambda _ref, key=key: _SINGLE_GRM_EIG_CACHE.pop(key, None)),
            eigvals,
            U,
        )
    except TypeError:
        pass
    return eigvals, U, False


def _newton_step(AI, grad, theta, constrain=True):
    """Regularised Newton step.

    If ``constrain=True`` (default), step-halves to keep all θ > 0 and clips
    the result.  If ``constrain=False``, takes the full Newton step with no
    positivity requirement (GCTA --reml-no-constrain behaviour).
    """
    try:
        delta = np.linalg.solve(AI + 1e-10 * np.eye(len(theta)), grad)
    except np.linalg.LinAlgError:
        return theta, np.zeros_like(theta)
    if constrain:
        alpha = 1.0
        for _ in range(20):
            if np.all(theta + alpha * delta > _TINY):
                break
            alpha *= 0.5
        return np.maximum(theta + alpha * delta, _TINY), delta
    else:
        return theta + delta, delta


def _se_from_ai(AI, theta):
    """SE for θ_k and h²_k = θ_k / Vp from the AI matrix.

    Returns
    -------
    h2          : ndarray (K,)   heritability per component
    se_h2       : ndarray (K,)   SE(h²_k) via delta method
    se_theta    : ndarray (K+1,) SE(θ_k) = sqrt(AI⁻¹[k,k])
    se_total_h2 : float          SE(sum_k h²_k) via delta method
    """
    K  = len(theta) - 1
    Vp = theta.sum()
    h2 = theta[:K] / Vp
    try:
        Cov      = np.linalg.inv(AI + 1e-12 * np.eye(K + 1))
        se_theta = np.sqrt(np.maximum(np.diag(Cov), 0.0))
        se_h2    = np.empty(K)
        for k in range(K):
            g     = -h2[k] * np.ones(K + 1) / Vp
            g[k] += 1.0 / Vp
            se_h2[k] = np.sqrt(max(g @ Cov @ g, 0.0))
        h2_total = h2.sum()
        g_total = np.concatenate([np.ones(K), np.zeros(1)]) / Vp
        g_total -= h2_total * np.ones(K + 1) / Vp
        se_total_h2 = float(np.sqrt(max(g_total @ Cov @ g_total, 0.0)))
    except Exception:
        se_theta = np.full(K + 1, np.nan)
        se_h2    = np.full(K, np.nan)
        se_total_h2 = np.nan
    return h2, se_h2, se_theta, se_total_h2


# ── Projection helper ──────────────────────────────────────────────────────────

def _proj(Lc, X_full, KPy, K_list_or_Kd, dtype):
    """Compute quantities needed for the REML projection P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹.

    Parameters
    ----------
    Lc       : Cholesky factor of V (scipy cho_factor output)
    X_full   : ndarray (n, p), float64 — design matrix including intercept
    KPy      : ndarray (n, K+1), working dtype — [K_1 Py | … | K_K Py | Py]
    K_list_or_Kd : list of (n,n) arrays — GRM matrices in working dtype
    dtype    : working dtype

    Returns
    -------
    VinvX    : ndarray (n, p), float64 — V⁻¹ X
    XVX_inv  : ndarray (p, p), float64 — (X'V⁻¹X)⁻¹
    PKPy64   : ndarray (n, K+1), float64 — P(K_k Py), corrected
    tr_corr  : ndarray (K+1,), float64 — tr(V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹K_k) per component
    """
    VinvX   = cho_solve(Lc, X_full.astype(dtype), check_finite=False
                        ).astype(np.float64)
    XVX_inv = np.linalg.inv(X_full.T @ VinvX)

    # P(KPy) = V⁻¹(KPy) − V⁻¹X (X'V⁻¹X)⁻¹ X'V⁻¹(KPy)
    VinvKPy = cho_solve(Lc, KPy, check_finite=False).astype(np.float64)
    PKPy64  = VinvKPy - VinvX @ (XVX_inv @ (VinvX.T @ KPy.astype(np.float64)))

    # tr correction: tr((X'V⁻¹X)⁻¹ X'V⁻¹ K_k V⁻¹X) for each component
    K = len(K_list_or_Kd) - 1  # last element is identity (error)
    tr_corr = np.empty(K + 1, dtype=np.float64)
    for k in range(K):
        KVX        = K_list_or_Kd[k].astype(np.float64) @ VinvX   # n×p
        tr_corr[k] = np.trace(XVX_inv @ (VinvX.T @ KVX))
    tr_corr[K] = np.trace(XVX_inv @ (VinvX.T @ VinvX))

    return VinvX, XVX_inv, PKPy64, tr_corr


# ── Public API ─────────────────────────────────────────────────────────────────

def greml_stochastic(K_list, y, n_probes=50, max_iter=30, tol=1e-5,
                     dtype=np.float32, seed=42, mean_correct=True, X=None,
                     constrain=True, verbose=False):
    """AI-REML with Hutchinson stochastic trace estimation.

    Parameters
    ----------
    K_list : list of ndarray, each shape (n, n)
        Pre-computed dense GRM matrices (one per variance component).
    y : ndarray, shape (n,)
        Phenotype vector.
    n_probes : int
        Number of Rademacher probe vectors for the Hutchinson trace
        estimator. Probe pairs are antithetic when possible; odd probe
        counts use one additional independent vector. 50 gives ~1%
        relative error; 30 is sufficient for standardised phenotypes.
    max_iter : int
        Maximum AI-REML iterations.
    tol : float
        Convergence threshold on max |Δθ|.
    dtype : np.float32 | np.float64
        Working precision for Cholesky. float32 is ~2.85× faster on
        arm64/Accelerate and safe for typical GRMs (κ(V) ≈ 4–5).
    seed : int
        RNG seed for reproducible probe vectors.
    mean_correct : bool
        Apply the REML mean projection for an intercept-only model. Ignored
        when ``X`` is provided (projection is always applied with covariates).
        For standardised y with no covariates this correction is O(1/n) and
        can be skipped (``mean_correct=False``) for a small speedup.
    X : ndarray (n, q) or None
        Fixed-effect covariate matrix, NOT including the intercept column
        (it is added automatically). When provided, the full REML projection
        P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹ is applied with X = [1 | covariates].

    Returns
    -------
    dict with keys:
        h2     : ndarray (K,)   — heritability estimates per component
        se     : ndarray (K,)   — standard errors (delta method from AI⁻¹)
        se_theta : ndarray (K+1,) — SE of raw variance components
        se_total_h2 : float     — SE of sum_k h²_k
        theta  : ndarray (K+1,) — raw variance components (σ²_1,…,σ²_e)
        n_iter : int            — iterations taken
    """
    _validate_inputs(K_list, y, X=X, n_probes=n_probes)

    n   = len(y)
    K   = len(K_list)
    rng = np.random.default_rng(seed)

    Kd    = [Kk.astype(dtype, copy=False) for Kk in K_list]
    yd    = y.astype(dtype)
    theta = _he_warmstart(Kd, y).astype(np.float64)

    # Build design matrix (intercept always included)
    do_proj = (X is not None) or mean_correct
    intercept_only = do_proj and X is None
    if do_proj:
        ones_col = np.ones((n, 1), dtype=np.float64)
        X_full   = np.hstack([ones_col, X]) if X is not None else ones_col
        ones_work = np.ones(n, dtype=dtype)

    probe_blocks = []
    n_pairs = n_probes // 2
    if n_pairs:
        Zh = (rng.integers(0, 2, size=(n, n_pairs)) * 2 - 1).astype(dtype)
        probe_blocks.extend([Zh, -Zh])
    if n_probes % 2:
        probe_blocks.append(
            (rng.integers(0, 2, size=(n, 1)) * 2 - 1).astype(dtype)
        )
    Z = np.concatenate(probe_blocks, axis=1)
    s = Z.shape[1]

    AI_last = np.eye(K + 1, dtype=np.float64)

    for it in range(max_iter):
        if constrain:
            theta = np.clip(theta, _TINY, None)

        V = dtype(theta[0]) * Kd[0]
        for k in range(1, K):
            V = V + dtype(theta[k]) * Kd[k]
        d = V.diagonal().copy(); d += dtype(theta[K])
        np.fill_diagonal(V, d)

        try:
            Lc = cho_factor(V, lower=True, overwrite_a=True, check_finite=False)
        except linalg.LinAlgError:
            theta *= 1.1
            continue

        # Py = V⁻¹y, then project out fixed effects
        Vinvy = cho_solve(Lc, yd, check_finite=False)
        if intercept_only:
            Vinv1 = cho_solve(Lc, ones_work, check_finite=False)
            denom = float(np.sum(Vinv1, dtype=np.float64))
            beta = float(np.sum(Vinvy, dtype=np.float64)) / denom
            Py = Vinvy - Vinv1 * dtype(beta)
        elif do_proj:
            VinvX   = cho_solve(Lc, X_full.astype(dtype),
                                check_finite=False).astype(np.float64)
            XVX_inv = np.linalg.inv(X_full.T @ VinvX)
            beta    = XVX_inv @ (VinvX.T @ y)
            Py      = Vinvy - (VinvX @ beta).astype(dtype)
            VinvX_work = VinvX.astype(dtype)
        else:
            Py = Vinvy

        VinvZ = cho_solve(Lc, Z, check_finite=False)
        if intercept_only:
            KPy_gen, traces_gen, uKu = _stochastic_component_ops(
                Kd, Py, VinvZ, Z, Vinv1=Vinv1
            )
        else:
            KPy_gen, traces_gen, _ = _stochastic_component_ops(Kd, Py, VinvZ, Z)

        KPy = np.empty((n, K + 1), dtype=dtype)
        KPy[:, :K] = KPy_gen
        KPy[:, K] = Py

        # AI matrix: P(KPy) = V⁻¹(KPy) − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹(KPy)
        if intercept_only:
            VinvKPy = cho_solve(Lc, KPy, check_finite=False).astype(np.float64)
            Vinv1_64 = Vinv1.astype(np.float64, copy=False)
            coeff = (Vinv1_64 @ KPy.astype(np.float64)) / denom
            PKPy64 = VinvKPy - np.outer(Vinv1_64, coeff)
        elif do_proj:
            VinvKPy = cho_solve(Lc, KPy, check_finite=False).astype(np.float64)
            PKPy64  = VinvKPy - VinvX @ (XVX_inv @ (VinvX.T @ KPy.astype(np.float64)))
        else:
            PKPy64 = cho_solve(Lc, KPy, check_finite=False).astype(np.float64)

        KPy64   = KPy.astype(np.float64)
        AI      = 0.5 * (KPy64.T @ PKPy64)
        AI_last = AI
        quad    = np.einsum('n,nk->k', Py.astype(np.float64), KPy64)

        # Stochastic traces: tr(V⁻¹K_k) via Hutchinson
        traces = np.empty(K + 1, dtype=np.float64)
        traces[:K] = traces_gen
        traces[K] = float((Z * VinvZ).sum()) / s

        # Subtract tr(V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹K_k) to get tr(PK_k)
        if intercept_only:
            traces[:K] -= uKu / denom
            traces[K] -= float(np.dot(Vinv1.astype(np.float64, copy=False),
                                      Vinv1.astype(np.float64, copy=False))) / denom
        elif do_proj:
            for k in range(K):
                KVX        = (Kd[k] @ VinvX_work).astype(np.float64)
                traces[k] -= np.trace(XVX_inv @ (VinvX.T @ KVX))
            traces[K] -= np.trace(XVX_inv @ (VinvX.T @ VinvX))

        grad           = 0.5 * (quad - traces)
        theta, delta   = _newton_step(AI_last, grad, theta, constrain=constrain)
        if verbose:
            Vp  = theta.sum()
            h2v = theta[:K] / Vp
            print(f"  iter {it+1:2d}:  "
                  f"h²=[{', '.join(f'{v:.4f}' for v in h2v)}]  "
                  f"max|Δθ|={np.max(np.abs(delta)):.2e}")
        if it >= 3 and np.max(np.abs(delta)) < tol:
            break

    h2, se_h2, se_theta, se_total_h2 = _se_from_ai(AI_last, theta)
    return {'h2': h2, 'se': se_h2, 'se_theta': se_theta,
            'theta': theta, 'n_iter': it + 1, 'se_total_h2': se_total_h2}


def _greml_exact_dense_ai(K_list, y, max_iter=30, tol=1e-6, X=None,
                          constrain=True, verbose=False):
    """Dense exact AI-REML path used for K > 1 and regression testing."""
    _validate_inputs(K_list, y, X=X)

    n  = len(y)
    K  = len(K_list)
    In = np.eye(n, dtype=np.float64)
    potri = get_lapack_funcs("potri", (In,))
    ones = np.ones(n, dtype=np.float64)

    ones_col = np.ones((n, 1), dtype=np.float64)
    X_full   = np.hstack([ones_col, X]) if X is not None else ones_col
    intercept_only = X is None

    theta   = _he_warmstart(K_list, y)
    AI_last = np.eye(K + 1)

    for it in range(max_iter):
        if constrain:
            theta = np.clip(theta, _TINY, None)
        V = theta[K] * In.copy()
        for k in range(K):
            V += theta[k] * K_list[k]

        try:
            Lc = cho_factor(V, lower=True, overwrite_a=True, check_finite=False)
        except linalg.LinAlgError:
            theta *= 1.1
            continue

        if intercept_only:
            Vinvy = cho_solve(Lc, y, check_finite=False)
            Vinv1 = cho_solve(Lc, ones, check_finite=False)
            denom = float(np.sum(Vinv1))
            beta = float(np.sum(Vinvy)) / denom
            Py = Vinvy - Vinv1 * beta
        else:
            VinvX   = cho_solve(Lc, X_full, check_finite=False)
            XVX_inv = np.linalg.inv(X_full.T @ VinvX)
            beta    = XVX_inv @ (VinvX.T @ y)
            Py      = cho_solve(Lc, y, check_finite=False) - VinvX @ beta
        Vinv_tri, info = potri(Lc[0].copy(order="F"), lower=1, overwrite_c=1)
        if info != 0:
            raise linalg.LinAlgError(f"potri failed with info={info}")
        Vinv = np.tril(Vinv_tri)
        Vinv += np.tril(Vinv, k=-1).T

        KPy  = np.column_stack([K_list[k] @ Py for k in range(K)] + [Py])
        # P(KPy) = V⁻¹(KPy) − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹(KPy)
        if intercept_only:
            coeff = (Vinv1 @ KPy) / denom
            PKPy = Vinv @ KPy - np.outer(Vinv1, coeff)
        else:
            PKPy = Vinv @ KPy - VinvX @ (XVX_inv @ (VinvX.T @ KPy))

        AI      = 0.5 * (KPy.T @ PKPy)
        AI_last = AI
        quad    = np.einsum('n,nk->k', Py, KPy)

        # Exact traces: tr(V⁻¹K_k) − tr((X'V⁻¹X)⁻¹X'V⁻¹K_kV⁻¹X)
        traces = np.empty(K + 1)
        for k in range(K):
            traces[k] = (Vinv * K_list[k]).sum()
        traces[K] = np.trace(Vinv)
        if intercept_only:
            for k in range(K):
                traces[k] -= float(Vinv1 @ (K_list[k] @ Vinv1)) / denom
            traces[K] -= float(Vinv1 @ Vinv1) / denom
        else:
            for k in range(K):
                KVX        = K_list[k] @ VinvX
                traces[k] -= np.trace(XVX_inv @ (VinvX.T @ KVX))
            traces[K] -= np.trace(XVX_inv @ (VinvX.T @ VinvX))

        grad           = 0.5 * (quad - traces)
        theta, delta   = _newton_step(AI_last, grad, theta, constrain=constrain)
        if verbose:
            Vp  = theta.sum()
            h2v = theta[:K] / Vp
            print(f"  iter {it+1:2d}:  "
                  f"h²=[{', '.join(f'{v:.4f}' for v in h2v)}]  "
                  f"max|Δθ|={np.max(np.abs(delta)):.2e}")
        if it >= 3 and np.max(np.abs(delta)) < tol:
            break

    h2, se_h2, se_theta, se_total_h2 = _se_from_ai(AI_last, theta)
    return {'h2': h2, 'se': se_h2, 'se_theta': se_theta,
            'theta': theta, 'n_iter': it + 1, 'se_total_h2': se_total_h2}


def greml_exact_single(K, y, max_iter=30, tol=1e-6, X=None, constrain=True,
                       verbose=False):
    """Exact AI-REML for a single GRM via one eigendecomposition.

    After diagonalising K = U diag(λ) U', each AI-REML iteration reduces to
    O(n p + n) operations in the rotated basis instead of repeated O(n^3)
    dense factorizations. The eigendecomposition is cached per GRM object so
    repeated fits on the same matrix amortise the O(n^3) setup cost.
    """
    _validate_inputs([K], y, X=X)

    n = len(y)
    Kd = np.asarray(K, dtype=np.float64)
    eigvals, U, cache_hit = _get_single_grm_eigendecomp(K)
    if verbose:
        status = "reusing cached eigendecomposition" if cache_hit else "computing eigendecomposition"
        print(f"  single-component exact GREML: {status}")
    z = U.T @ y

    ones_col = np.ones((n, 1), dtype=np.float64)
    X_full = np.hstack([ones_col, X]) if X is not None else ones_col
    W = U.T @ X_full

    theta = _he_warmstart([Kd], y)
    AI_last = np.eye(2, dtype=np.float64)

    def apply_p_rot(inv_v, xvx_chol, vec):
        rhs = W.T @ (inv_v * vec)
        coeff = cho_solve(xvx_chol, rhs, check_finite=False)
        return inv_v * vec - inv_v * (W @ coeff)

    for it in range(max_iter):
        if constrain:
            theta = np.clip(theta, _TINY, None)

        v = theta[0] * eigvals + theta[1]
        min_v = float(np.min(v))
        if min_v <= _TINY:
            theta[1] += (_TINY - min_v) + 1e-12
            continue

        inv_v = 1.0 / v
        XVX = W.T @ (inv_v[:, None] * W)
        try:
            xvx_chol = cho_factor(XVX, lower=True, check_finite=False)
        except linalg.LinAlgError:
            theta *= 1.1
            continue

        beta_rhs = W.T @ (inv_v * z)
        beta = cho_solve(xvx_chol, beta_rhs, check_finite=False)
        Py_rot = inv_v * (z - W @ beta)

        KPy_rot = np.column_stack([eigvals * Py_rot, Py_rot])
        PKPy_rot = np.column_stack([
            apply_p_rot(inv_v, xvx_chol, KPy_rot[:, 0]),
            apply_p_rot(inv_v, xvx_chol, KPy_rot[:, 1]),
        ])

        AI = 0.5 * (KPy_rot.T @ PKPy_rot)
        AI_last = AI
        quad = np.array([
            float(Py_rot @ KPy_rot[:, 0]),
            float(Py_rot @ KPy_rot[:, 1]),
        ])

        lambda_weight = (eigvals * inv_v * inv_v)[:, None] * W
        resid_weight = (inv_v * inv_v)[:, None] * W
        trace_g = np.sum(eigvals * inv_v)
        trace_e = np.sum(inv_v)
        trace_g -= np.trace(cho_solve(
            xvx_chol, W.T @ lambda_weight, check_finite=False
        ))
        trace_e -= np.trace(cho_solve(
            xvx_chol, W.T @ resid_weight, check_finite=False
        ))
        grad = 0.5 * (quad - np.array([trace_g, trace_e]))

        theta, delta = _newton_step(AI_last, grad, theta, constrain=constrain)
        if verbose:
            Vp = theta.sum()
            h2v = theta[0] / Vp
            print(f"  iter {it+1:2d}:  h²=[{h2v:.4f}]  max|Δθ|={np.max(np.abs(delta)):.2e}")
        if it >= 3 and np.max(np.abs(delta)) < tol:
            break

    h2, se_h2, se_theta, se_total_h2 = _se_from_ai(AI_last, theta)
    return {'h2': h2, 'se': se_h2, 'se_theta': se_theta,
            'theta': theta, 'n_iter': it + 1, 'se_total_h2': se_total_h2}


def greml_exact(K_list, y, max_iter=30, tol=1e-6, X=None, constrain=True,
                verbose=False, use_eigendecomp=True):
    """AI-REML with exact traces.

    Uses a dedicated eigen-based solver for the single-component case and the
    dense AI-REML path for multi-component models.
    """
    if len(K_list) == 1 and use_eigendecomp:
        return greml_exact_single(
            K_list[0], y, max_iter=max_iter, tol=tol, X=X,
            constrain=constrain, verbose=verbose
        )
    return _greml_exact_dense_ai(
        K_list, y, max_iter=max_iter, tol=tol, X=X,
        constrain=constrain, verbose=verbose
    )
