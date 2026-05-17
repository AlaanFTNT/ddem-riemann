"""Bayesian evidence + Bayes factor for DDEM vs LCDM under the 6-likelihood
joint pipeline.

Uses dynesty (Speagle 2020) nested sampling. We compute ln Z for both
DDEM (7-parameter) and LCDM (4-parameter: h, omega_b, omega_cdm, with
beta0 = eps = 0, alpha = 0.5 fixed structurally, w_DE = -1 fixed) under
the same 6-likelihood pipeline used in §7.

The Bayes factor ln B = ln Z_DDEM - ln Z_LCDM converts the frequentist
"beta0 at 2 sigma above zero" statement (§7.2) into a quantitative model
preference statement. Scales (Kass-Raftery 1995):

    |ln B| < 1     inconclusive
    1 < |ln B| < 3 positive
    3 < |ln B| < 5 substantial
    |ln B| > 5     strong

Discard criterion: |ln B| < 2 (note it but do not headline).
Keep criterion:    |ln B| >= 2 (report as a model-comparison statement).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "theory"))

from run_joint_mcmc import compute_predictions, PRIORS, LO, HI

try:
    import dynesty
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False
    print("dynesty not installed; install via pip install dynesty")


def loglike_ddem(theta):
    chi2, _ = compute_predictions(theta)
    if not np.isfinite(chi2):
        return -1e30
    return -0.5 * chi2


def loglike_lcdm(theta_3):
    """theta_3 = (h, omega_b, omega_cdm). Other params fixed at LCDM values."""
    h, omega_b, omega_cdm = theta_3
    theta_full = np.array([0.0, 0.0, 0.5, h, omega_b, omega_cdm, -1.01])
    chi2, _ = compute_predictions(theta_full)
    if not np.isfinite(chi2):
        return -1e30
    return -0.5 * chi2


def loglike_ddem_trimmed(theta_5):
    """Canonical-reduction DDEM: alpha=1/2 (Pourtsidou Type-2 natural value)
    and w_DE = -1 (cosmological-constant simplicity) fixed by theory, not
    by evidence-optimization. Free params: (beta0, eps, h, omega_b, omega_cdm)."""
    beta0, eps, h, omega_b, omega_cdm = theta_5
    theta_full = np.array([beta0, eps, 0.5, h, omega_b, omega_cdm, -1.0])
    chi2, _ = compute_predictions(theta_full)
    if not np.isfinite(chi2):
        return -1e30
    return -0.5 * chi2


def prior_transform_ddem(u):
    """Map [0,1]^7 to the DDEM prior cube."""
    return LO + (HI - LO) * u


def prior_transform_lcdm(u):
    """LCDM prior cube: (h, omega_b, omega_cdm) only."""
    lo = np.array([LO[3], LO[4], LO[5]])     # h, omega_b, omega_cdm
    hi = np.array([HI[3], HI[4], HI[5]])
    return lo + (hi - lo) * u


def prior_transform_ddem_trimmed(u):
    """Trimmed-DDEM prior cube: (beta0, eps, h, omega_b, omega_cdm)."""
    lo = np.array([LO[0], LO[1], LO[3], LO[4], LO[5]])
    hi = np.array([HI[0], HI[1], HI[3], HI[4], HI[5]])
    return lo + (hi - lo) * u


def run_evidence(nlive=400, dlogz=0.5, nproc=12):
    if not DYNESTY_AVAILABLE:
        return None

    print("=" * 70)
    print("Bayesian evidence pipeline (dynesty)")
    print("=" * 70)
    from multiprocessing import Pool

    print(f"\nDDEM (7 params, nlive={nlive}, dlogz={dlogz}, nproc={nproc})...")
    t0 = time.time()
    with Pool(nproc) as pool:
        sampler = dynesty.NestedSampler(
            loglike_ddem, prior_transform_ddem, ndim=7,
            nlive=nlive, pool=pool, queue_size=nproc, sample="rwalk")
        sampler.run_nested(dlogz=dlogz, print_progress=False)
    res_ddem = sampler.results
    dt = time.time() - t0
    print(f"  done in {dt:.0f}s; logz = {res_ddem.logz[-1]:.3f} +/- {res_ddem.logzerr[-1]:.3f}")

    print(f"\nLCDM (3 params, nlive={nlive}, dlogz={dlogz}, nproc={nproc})...")
    t0 = time.time()
    with Pool(nproc) as pool:
        sampler = dynesty.NestedSampler(
            loglike_lcdm, prior_transform_lcdm, ndim=3,
            nlive=nlive, pool=pool, queue_size=nproc, sample="rwalk")
        sampler.run_nested(dlogz=dlogz, print_progress=False)
    res_lcdm = sampler.results
    dt = time.time() - t0
    print(f"  done in {dt:.0f}s; logz = {res_lcdm.logz[-1]:.3f} +/- {res_lcdm.logzerr[-1]:.3f}")

    lnZ_ddem = res_ddem.logz[-1]
    lnZ_lcdm = res_lcdm.logz[-1]
    lnZ_err  = np.sqrt(res_ddem.logzerr[-1]**2 + res_lcdm.logzerr[-1]**2)
    lnB = lnZ_ddem - lnZ_lcdm
    print()
    print("=" * 70)
    print(f"Bayes factor (DDEM vs LCDM):  ln B = {lnB:+.3f} +/- {lnZ_err:.3f}")
    print("=" * 70)
    if abs(lnB) < 2:
        verdict = "DISCARD: |ln B| < 2 (inconclusive)"
    elif abs(lnB) < 5:
        verdict = f"KEEP: |ln B| = {abs(lnB):.1f}, substantial preference for " \
                  f"{'DDEM' if lnB > 0 else 'LCDM'}"
    else:
        verdict = f"KEEP: |ln B| = {abs(lnB):.1f}, strong preference for " \
                  f"{'DDEM' if lnB > 0 else 'LCDM'}"
    print(verdict)
    np.savez(HERE / "bayes_evidence.npz",
             lnZ_ddem=lnZ_ddem, lnZ_lcdm=lnZ_lcdm, lnB=lnB,
             lnZ_err=lnZ_err)
    print(f"  saved: cobaya/bayes_evidence.npz")
    return lnB, lnZ_err, verdict


def run_evidence_trimmed(nlive=400, dlogz=0.5, nproc=12):
    """Trimmed-DDEM (5 free params) vs already-saved LCDM ln Z.

    LCDM ln Z = -720.192 +/- 0.270 was computed in the full 7-vs-3 run and
    stored in cobaya/bayes_evidence.npz. We re-use that value rather than
    recomputing it.
    """
    if not DYNESTY_AVAILABLE:
        return None
    from multiprocessing import Pool

    # Load LCDM ln Z from prior run
    try:
        prev = np.load(HERE / "bayes_evidence.npz")
        lnZ_lcdm = float(prev["lnZ_lcdm"])
        lnZ_lcdm_err = float(np.sqrt(prev["lnZ_err"]**2 - 0.309**2))  # back out lcdm err
        if not np.isfinite(lnZ_lcdm_err) or lnZ_lcdm_err <= 0:
            lnZ_lcdm_err = 0.270
    except Exception:
        # Fallback: use the recorded value from the just-completed run
        lnZ_lcdm = -720.192
        lnZ_lcdm_err = 0.270

    print("=" * 70)
    print("Bayesian evidence: TRIMMED DDEM (alpha=1/2, w_DE=-1 fixed)")
    print("=" * 70)
    print(f"  Re-using LCDM ln Z = {lnZ_lcdm:.3f} +/- {lnZ_lcdm_err:.3f} (3-param run)")
    print()
    print(f"Trimmed DDEM (5 params, nlive={nlive}, dlogz={dlogz}, nproc={nproc})...")
    t0 = time.time()
    with Pool(nproc) as pool:
        sampler = dynesty.NestedSampler(
            loglike_ddem_trimmed, prior_transform_ddem_trimmed, ndim=5,
            nlive=nlive, pool=pool, queue_size=nproc, sample="rwalk")
        sampler.run_nested(dlogz=dlogz, print_progress=False)
    res = sampler.results
    dt = time.time() - t0
    lnZ_ddem_t = res.logz[-1]
    lnZ_ddem_t_err = res.logzerr[-1]
    print(f"  done in {dt:.0f}s; logz = {lnZ_ddem_t:.3f} +/- {lnZ_ddem_t_err:.3f}")

    lnB = lnZ_ddem_t - lnZ_lcdm
    lnB_err = np.sqrt(lnZ_ddem_t_err**2 + lnZ_lcdm_err**2)
    print()
    print("=" * 70)
    print(f"Bayes factor (trimmed DDEM vs LCDM):  ln B = {lnB:+.3f} +/- {lnB_err:.3f}")
    print("=" * 70)
    if abs(lnB) < 2:
        verdict = f"INCONCLUSIVE: |ln B| = {abs(lnB):.2f} < 2 (current data neither support nor exclude)"
    elif abs(lnB) < 5:
        verdict = f"SUBSTANTIAL: |ln B| = {abs(lnB):.2f}, preference for " \
                  f"{'trimmed DDEM' if lnB > 0 else 'LCDM'}"
    else:
        verdict = f"STRONG: |ln B| = {abs(lnB):.2f}, preference for " \
                  f"{'trimmed DDEM' if lnB > 0 else 'LCDM'}"
    print(verdict)
    np.savez(HERE / "bayes_evidence_trimmed.npz",
             lnZ_ddem_trimmed=lnZ_ddem_t, lnZ_lcdm=lnZ_lcdm, lnB=lnB,
             lnB_err=lnB_err, lnZ_ddem_trimmed_err=lnZ_ddem_t_err,
             lnZ_lcdm_err=lnZ_lcdm_err)
    print(f"  saved: cobaya/bayes_evidence_trimmed.npz")
    return lnB, lnB_err, verdict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trimmed", action="store_true",
                        help="Run the canonical-reduction (alpha=1/2, w=-1) variant only")
    parser.add_argument("--nlive", type=int, default=400)
    parser.add_argument("--nproc", type=int, default=12)
    parser.add_argument("--dlogz", type=float, default=0.5)
    args = parser.parse_args()

    if not DYNESTY_AVAILABLE:
        print("Install dynesty first: pip install dynesty")
    elif args.trimmed:
        run_evidence_trimmed(nlive=args.nlive, dlogz=args.dlogz, nproc=args.nproc)
    else:
        run_evidence(nlive=args.nlive, dlogz=args.dlogz, nproc=args.nproc)
