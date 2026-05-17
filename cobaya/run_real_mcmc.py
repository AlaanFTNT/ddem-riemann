"""Task #6 — Real-data MCMC against DESI DR2 BAO using the modified DDEM-CLASS.

Samples (beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld) under flat priors
per Refined_Hypothesis.md §8. Saves chain to NPY for offline analysis.

Run from WSL Ubuntu via /opt/ddem-venv/bin/python.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import numpy as np
import emcee
from classy import Class

OUT_DIR = Path(__file__).parent
CHAIN_PATH = OUT_DIR / "chain_desi_bao.npy"
LNP_PATH   = OUT_DIR / "chain_lnp.npy"

sys.path.insert(0, str(OUT_DIR))
from desi_dr2_bao_data import chi2_desi_bao

# ------------------ Theory wrapper ------------------
def compute_predictions(theta):
    """Return (DM_over_rd_fn, DH_over_rd_fn, DV_over_rd_fn) for the input parameters."""
    beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld = theta
    params = {
        "output": "",  # background only — fast
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "n_s": 0.9667,
        "tau_reio": 0.066,
        "ln10^{10}A_s": 3.064,
        "Omega_Lambda": 0.0,
        "fluid_equation_of_state": "CLP",
        "w0_fld": w0_fld,
        "wa_fld": 0.0,
        "cs2_fld": 1.0,
        "use_ppf": "yes",
        "ddem_enabled": "yes",
        "ddem_beta0": float(beta0),
        "ddem_eps": float(eps),
        "ddem_alpha": float(alpha),
        "ddem_N": 100,
    }
    c = Class()
    try:
        c.set(params)
        c.compute()
        rd = c.rs_drag()  # Mpc
        # closures
        def DM_over_rd(z):
            return (1.0 + z) * c.angular_distance(z) / rd
        def DH_over_rd(z):
            return 1.0 / (c.Hubble(z) * rd)  # Hubble(z) returned in 1/Mpc by classy
        def DV_over_rd(z):
            DM = (1.0 + z) * c.angular_distance(z)
            DH = 1.0 / c.Hubble(z)
            return (z * DM**2 * DH)**(1.0/3.0) / rd
        # Eager-evaluate before cleanup
        results = {}
        for z in [0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330]:
            results[z] = {
                "DM": DM_over_rd(z),
                "DH": DH_over_rd(z),
                "DV": DV_over_rd(z),
            }
        return results, rd, c
    finally:
        try:
            c.struct_cleanup()
        except Exception:
            pass


def chi2_from_cache(cache):
    return chi2_desi_bao(
        lambda z: cache[z]["DM"],
        lambda z: cache[z]["DH"],
        lambda z: cache[z]["DV"],
    )


# ------------------ Priors ------------------
# (beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld)
PRIORS = {
    "beta0":     (0.0, 0.02),
    "eps":       (0.0, 2.0),
    "alpha":     (-1.0, 2.0),
    "h":         (0.55, 0.85),
    "omega_b":   (0.018, 0.026),
    "omega_cdm": (0.08, 0.16),
    "w0_fld":    (-1.5, -1.01),
}
LO = np.array([v[0] for v in PRIORS.values()])
HI = np.array([v[1] for v in PRIORS.values()])

def log_prior(theta):
    if np.any(theta < LO) or np.any(theta > HI):
        return -np.inf
    return 0.0

def log_post(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        cache, rd, _ = compute_predictions(theta)
        chi2 = chi2_from_cache(cache)
    except Exception:
        return -np.inf
    return lp - 0.5 * chi2


def main(nwalkers=24, nsteps=400, burn=100):
    # Initialize walkers in a tight ball near Planck-like cosmology + small DDEM coupling
    ndim = 7
    init_center = np.array([0.005, 0.5, 0.5, 0.6774, 0.0223, 0.1188, -1.05])
    init_scale  = np.array([0.003, 0.2, 0.2, 0.005, 0.0005, 0.002, 0.02])
    p0 = init_center + init_scale * np.random.randn(nwalkers, ndim)
    # clip into prior
    p0 = np.clip(p0, LO + 1e-6, HI - 1e-6)

    print(f"Starting MCMC: {nwalkers} walkers x {nsteps} steps (ndim={ndim})")
    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)
    sampler.run_mcmc(p0, nsteps, progress=False)
    print(f"MCMC done in {time.time()-t0:.1f}s  acceptance={np.mean(sampler.acceptance_fraction):.3f}")

    chain = sampler.get_chain(discard=burn, flat=True)
    lnp = sampler.get_log_prob(discard=burn, flat=True)
    np.save(CHAIN_PATH, chain)
    np.save(LNP_PATH, lnp)
    print(f"Chain saved: {CHAIN_PATH}")
    print(f"Chain shape: {chain.shape}")

    labels = list(PRIORS.keys())
    print()
    print("Posterior quantiles (16/50/84):")
    for i, lab in enumerate(labels):
        q16, q50, q84 = np.percentile(chain[:, i], [16, 50, 84])
        print(f"  {lab:10s}  {q50:+.5f}  [{q16:+.5f}, {q84:+.5f}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwalkers", type=int, default=24)
    ap.add_argument("--nsteps",   type=int, default=400)
    ap.add_argument("--burn",     type=int, default=100)
    a = ap.parse_args()
    main(nwalkers=a.nwalkers, nsteps=a.nsteps, burn=a.burn)
