#!/usr/bin/env python3
"""
run_greml.py — command-line wrapper for fast multi-component GREML
===================================================================
Drop-in replacement for GCTA's partitioned GREML.  Reads GRMs and
phenotypes in GCTA format, writes a .hsq output file in GCTA format.

Basic usage (mirrors GCTA):
    python run_greml.py --mgrm mgrm.txt --pheno pheno.txt --out results

Method selection:
    --method stochastic   Hutchinson stochastic traces, float32 (default)
    --method exact        Explicit V⁻¹ traces, float64 (~2× slower)

GCTA-compatible flags accepted:
    --grm <prefix>        Single GRM (alternative to --mgrm)
    --mgrm <file>         Text file listing GRM prefixes, one per line
    --pheno <file>        Phenotype file: FID IID PHENO (whitespace-delimited)
    --out <prefix>        Output prefix  [default: greml_out]
    --reml-no-lrt         Accepted for compatibility; LRT is never run
    --thread-num <n>      Set BLAS/OpenMP thread count

Stochastic-specific flags:
    --n-probes <int>      Hutchinson probe vectors  [default: 50]
    --no-mean-correct     Skip REML mean projection (safe for standardised y)

Output:
    <prefix>.hsq     GCTA-format heritability results
    stdout           Summary table
"""

import argparse
import os
import sys


# ── Argument parsing ───────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="run_greml.py",
        description="Fast multi-component GREML (drop-in for GCTA --reml)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Basic usage")[0].strip(),
    )

    grm_grp = p.add_mutually_exclusive_group(required=True)
    grm_grp.add_argument("--grm",  metavar="PREFIX",
                         help="Single GRM prefix ({prefix}.grm.bin/.grm.id)")
    grm_grp.add_argument("--mgrm", metavar="FILE",
                         help="File listing GRM prefixes, one per line")

    p.add_argument("--pheno",  metavar="FILE", required=True,
                   help="Phenotype file: FID IID PHENO")
    p.add_argument("--out",    metavar="PREFIX", default="greml_out",
                   help="Output prefix (default: greml_out)")
    p.add_argument("--method", choices=["stochastic", "exact"],
                   default="stochastic",
                   help="GREML algorithm (default: stochastic)")

    # Stochastic options
    p.add_argument("--n-probes", type=int, default=50, metavar="INT",
                   help="Hutchinson probe vectors for stochastic method (default: 50)")
    p.add_argument("--no-mean-correct", action="store_true",
                   help="Skip REML mean projection (safe when phenotype is standardised)")

    # GCTA-compatible flags
    p.add_argument("--covar", metavar="FILE",
                   help="Covariate file: FID IID COV1 [COV2 …] (quantitative, "
                        "whitespace-delimited). Intercept is added automatically.")
    p.add_argument("--reml-no-constrain", action="store_true",
                   help="Allow variance components to be negative (unconstrained optimisation)")
    p.add_argument("--reml-no-lrt", action="store_true",
                   help="Accepted for GCTA compatibility; LRT is never computed here")
    p.add_argument("--thread-num", type=int, default=None, metavar="N",
                   help="BLAS/OpenMP thread count")
    p.add_argument("--eigendecomp", action="store_true",
                   help="Enable the single-component exact eigendecomposition path")

    return p


# ── I/O helpers ────────────────────────────────────────────────────────────────

def read_grm_prefixes(args):
    if args.mgrm:
        with open(args.mgrm) as fh:
            prefixes = [ln.strip() for ln in fh if ln.strip()]
    else:
        prefixes = [args.grm]
    return prefixes


def read_pheno(path):
    """Read FID IID PHENO file; return (ordered IID list, pheno dict).
    Missing phenotypes (-9 or NA) are excluded."""
    import math

    iids   = []
    phenos = {}
    with open(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.split()
            if len(parts) < 3:
                continue
            fid, iid, raw = parts[0], parts[1], parts[2]
            if raw in ("-9", "NA", "na", "NaN", "."):
                continue
            key = (fid, iid)
            if key in phenos:
                raise ValueError(
                    f"Duplicate phenotype entry for {fid}/{iid} on line {line_no}."
                )
            iids.append(key)
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(
                    f"Phenotype for {fid}/{iid} on line {line_no} is not finite."
                )
            phenos[key] = value
    return iids, phenos


def read_covar(path):
    """Read FID IID COV1 [COV2 …] covariate file.
    Returns dict mapping (FID, IID) -> covariate row (1-D float array).
    Rows with any missing value (-9 or NA) are excluded."""
    import numpy as np

    covars = {}
    width = None
    with open(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.split()
            if len(parts) < 3:
                continue
            fid, iid = parts[0], parts[1]
            vals = parts[2:]
            if any(v in ("-9", "NA", "na", "NaN", ".") for v in vals):
                continue
            key = (fid, iid)
            if key in covars:
                raise ValueError(
                    f"Duplicate covariate entry for {fid}/{iid} on line {line_no}."
                )
            row = np.array([float(v) for v in vals], dtype=np.float64)
            if not np.isfinite(row).all():
                raise ValueError(
                    f"Covariates for {fid}/{iid} on line {line_no} are not finite."
                )
            if width is None:
                width = row.shape[0]
            elif row.shape[0] != width:
                raise ValueError(
                    f"Inconsistent covariate width on line {line_no}: "
                    f"expected {width}, got {row.shape[0]}."
                )
            covars[key] = row
    return covars


def align_covar_to_grm(ids_grm, covars):
    """Return covariate matrix (n, q) aligned to GRM row order.
    Exits if any GRM individual is missing from the covariate file."""
    import numpy as np

    rows = []
    for fid, iid in ids_grm:
        key = (fid, iid)
        if key not in covars:
            sys.exit(f"Error: individual {fid}/{iid} in GRM has no covariate entry.")
        rows.append(covars[key])
    X = np.array(rows, dtype=np.float64)
    # Sanity check: warn if any covariate has zero variance (likely a mistake)
    for j in range(X.shape[1]):
        if X[:, j].std() < 1e-10:
            print(f"Warning: covariate column {j+1} has near-zero variance.",
                  file=sys.stderr)
    return X


def align_pheno_to_grm(ids_grm, pheno_iids, phenos):
    """Return phenotype vector aligned to GRM row order.

    This implementation requires all GRM individuals to have a phenotype
    (matching GCTA's default behaviour).
    """
    import numpy as np

    pheno_set = set(pheno_iids)
    y    = []
    kept = []
    for fid, iid in ids_grm:
        key = (fid, iid)
        if key not in pheno_set:
            sys.exit(f"Error: individual {fid}/{iid} in GRM has no phenotype. "
                     f"Ensure all GRM individuals appear in the phenotype file.")
        y.append(phenos[key])
        kept.append(key)
    return np.array(y, dtype=np.float64), kept


# ── .hsq writer ────────────────────────────────────────────────────────────────

def write_hsq(path, K_list, result, n):
    """Write GCTA-format .hsq file."""
    h2       = result["h2"]
    se_h2    = result["se"]
    se_theta = result["se_theta"]
    se_total_h2 = result["se_total_h2"]
    theta    = result["theta"]
    K        = len(h2)
    Vp       = theta.sum()

    lines = ["Source\tVariance\tSE"]
    for k in range(K):
        lines.append(f"V(G{k+1})\t{theta[k]:.6f}\t{se_theta[k]:.6f}")
    lines.append(f"V(e)\t{theta[K]:.6f}\t{se_theta[K]:.6f}")
    lines.append(f"Vp\t{Vp:.6f}\tNA")
    for k in range(K):
        lines.append(f"V(G{k+1})/Vp\t{h2[k]:.6f}\t{se_h2[k]:.6f}")
    lines.append(f"Sum of V(G)/Vp\t{h2.sum():.6f}\t{se_total_h2:.6f}")
    lines += ["logL\tNA", "logL0\tNA", "LRT\tNA", "df\tNA", "Pval\tNA"]
    lines.append(f"n\t{n}")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Optional: set BLAS thread count before importing numpy/scipy.
    if args.thread_num is not None:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = str(args.thread_num)

    import numpy as np

    # Allow running from the fast_greml directory or its parent.
    sys.path.insert(0, os.path.dirname(__file__))
    from greml import read_grm, greml_stochastic, greml_exact

    # ── Load GRMs ──────────────────────────────────────────────────────────────
    prefixes = read_grm_prefixes(args)
    K        = len(prefixes)
    print(f"Loading {K} GRM(s)...", end="", flush=True)

    dtype   = np.float32 if args.method == "stochastic" else np.float64
    K_list  = []
    ids_grm = None
    try:
        for pfx in prefixes:
            Km, ids = read_grm(pfx, dtype=dtype)
            K_list.append(Km)
            if ids_grm is None:
                ids_grm = ids
            elif ids != ids_grm:
                sys.exit(f"Error: GRM {pfx} has different individuals from the first GRM.")
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    n = K_list[0].shape[0]
    print(f" done  (n={n}, K={K})")

    # ── Load and align phenotype ───────────────────────────────────────────────
    try:
        pheno_iids, phenos = read_pheno(args.pheno)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    y, _ = align_pheno_to_grm(ids_grm, pheno_iids, phenos)
    print(f"Phenotype: {len(y)} individuals, "
          f"mean={y.mean():.4f}, sd={y.std():.4f}")

    X_cov = None
    if args.covar:
        try:
            covars = read_covar(args.covar)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        X_cov  = align_covar_to_grm(ids_grm, covars)
        print(f"Covariates: {X_cov.shape[1]} column(s) loaded")

    # ── Run GREML ──────────────────────────────────────────────────────────────
    label = f"GREML-{'Stochastic' if args.method == 'stochastic' else 'Exact'} (K={K})"
    print(f"Running {label}...")

    import time
    t0 = time.perf_counter()

    constrain = not args.reml_no_constrain
    if args.method == "stochastic":
        result = greml_stochastic(
            K_list, y,
            n_probes=args.n_probes,
            mean_correct=not args.no_mean_correct,
            X=X_cov,
            constrain=constrain,
            verbose=True,
        )
    else:
        result = greml_exact(
            K_list, y,
            X=X_cov,
            constrain=constrain,
            verbose=True,
            use_eigendecomp=args.eigendecomp,
        )

    elapsed = time.perf_counter() - t0
    print(f"Converged in {result['n_iter']} iterations ({elapsed:.2f}s)")

    # ── Write output ───────────────────────────────────────────────────────────
    hsq_path = args.out + ".hsq"
    write_hsq(hsq_path, K_list, result, n)
    print(f"Results written to {hsq_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    h2       = result["h2"]
    se_h2    = result["se"]
    se_theta = result["se_theta"]
    theta    = result["theta"]
    print()
    print(f"{'Component':<14s}  {'Estimate':>10s}  {'SE':>10s}")
    print("-" * 40)
    for k in range(K):
        print(f"  V(G{k+1})      {theta[k]:>10.4f}  {se_theta[k]:>10.4f}")
    print(f"  V(e)         {theta[K]:>10.4f}  {se_theta[K]:>10.4f}")
    print("-" * 40)
    for k in range(K):
        print(f"  V(G{k+1})/Vp   {h2[k]:>10.4f}  {se_h2[k]:>10.4f}")
    print("-" * 40)
    print(f"  {'Sum':<11s}  {h2.sum():>10.4f}  {result['se_total_h2']:>10.4f}")


if __name__ == "__main__":
    main()
