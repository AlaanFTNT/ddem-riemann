"""
Task #5 — Synthetic injection-recovery test for DDEM (eps, alpha).

Pre-registered per Refined_Hypothesis.md §8.

Two experiments:
    A. Inject alpha = 0.5 (Riemann critical-line value) -> require 68% CI of alpha
       to enclose 0.5 and exclude both 0 and 1.
    B. Inject alpha = 0.3 (non-Riemann control) -> require 95% CI of alpha to
       EXCLUDE 0.5 (no spurious recovery of the critical line).

Fast model: P_DDEM(k)/P_LCDM(k) = 1 + 2*eps * sum_n (a_eff^alpha / |rho_n|) *
exp(-((k - k_n)/sigma_n)^2/2), with k_n = gamma_n / (eta0 * h) fixed at the
LCDM value (eta0 ~ 14136 Mpc). beta0 enters only through the background growth
modification (treated as a fixed offset for this proof-of-concept; full
joint inference requires the modified-CLASS pipeline).

Likelihood: chi^2 on 80 k-bins from 0.005 to 0.3 h/Mpc with sigma_P/P = 1.5%.
Sampler: emcee, 48 walkers, 2000 steps, 500 burn.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import time
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from ddem_background import riemann_zeros
from predict_signature import Eisenstein_Hu_pk

OUT_DIR = Path(__file__).parent
H_PLANCK = 0.6774
ETA0_MPC = 14136.17  # from ddem_perturbation.conformal_time_today() for baseline LCDM
N_ZEROS = 200
A_EFF = 0.5
SIGMA_FACTOR = 0.02   # Gaussian width = SIGMA_FACTOR * k_n (heuristic)

# Pre-compute zeros and k_n
GAMMA_N = riemann_zeros(N_ZEROS)
K_N_H_MPC = GAMMA_N / ETA0_MPC / H_PLANCK
RHO_MOD = lambda alpha: np.sqrt(alpha**2 + GAMMA_N**2)

# Synthetic data k-grid
np.random.seed(20260514)
K_GRID = np.logspace(np.log10(0.005), np.log10(0.3), 80)
SIGMA_FRAC = 0.015


def fast_ratio(k_h, eps, alpha):
    """Vectorized comb residual. k_h shape (nk,)."""
    rho_mod = np.sqrt(alpha**2 + GAMMA_N**2)
    amp = A_EFF**alpha / rho_mod   # shape (N,)
    sig = SIGMA_FACTOR * K_N_H_MPC  # shape (N,)
    # broadcast (nk, N)
    gauss = np.exp(-0.5 * ((k_h[:, None] - K_N_H_MPC[None, :]) / sig[None, :])**2)
    contribution = np.sum(amp[None, :] * gauss, axis=1)
    return 1.0 + 2.0 * eps * contribution


def make_synthetic_data(eps_true, alpha_true):
    ratio_true = fast_ratio(K_GRID, eps_true, alpha_true)
    Pk_template = Eisenstein_Hu_pk(K_GRID)
    Pk_true = Pk_template * ratio_true
    sigma_Pk = SIGMA_FRAC * Pk_true
    noise = np.random.normal(0, sigma_Pk)
    Pk_data = Pk_true + noise
    return Pk_data, sigma_Pk, Pk_template


def log_prior(theta):
    eps, alpha = theta
    if not (0.0 <= eps <= 2.0):
        return -np.inf
    if not (-1.0 <= alpha <= 2.0):
        return -np.inf
    return 0.0


def log_likelihood(theta, Pk_data, sigma_Pk, Pk_template):
    eps, alpha = theta
    ratio = fast_ratio(K_GRID, eps, alpha)
    Pk_model = Pk_template * ratio
    chi2 = np.sum(((Pk_data - Pk_model) / sigma_Pk)**2)
    return -0.5 * chi2


def log_posterior(theta, Pk_data, sigma_Pk, Pk_template):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, Pk_data, sigma_Pk, Pk_template)


def run_experiment(name, eps_inj, alpha_inj, nwalkers=48, nsteps=2000, burn=500):
    print(f"\n==== Experiment {name}: inject (eps={eps_inj}, alpha={alpha_inj}) ====")
    Pk_data, sigma_Pk, Pk_template = make_synthetic_data(eps_inj, alpha_inj)
    ndim = 2
    p0 = np.column_stack([
        np.random.uniform(0.1, 1.9, nwalkers),
        np.random.uniform(-0.8, 1.8, nwalkers),  # flat on [-1, 2] (§8)
    ])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                    args=(Pk_data, sigma_Pk, Pk_template))
    t0 = time.time()
    sampler.run_mcmc(p0, nsteps, progress=False)
    print(f"  MCMC wall time: {time.time()-t0:.1f}s, acceptance = {np.mean(sampler.acceptance_fraction):.3f}")

    samples = sampler.get_chain(discard=burn, flat=True)
    quantiles = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=0)
    labels = ["eps", "alpha"]
    summary = {}
    for i, lab in enumerate(labels):
        q025, q16, q50, q84, q975 = quantiles[:, i]
        summary[lab] = (q50, q16, q84, q025, q975)
        inj = [eps_inj, alpha_inj][i]
        print(f"  {lab}: median={q50:+.4f}  68%CI=[{q16:+.4f}, {q84:+.4f}]  "
              f"95%CI=[{q025:+.4f}, {q975:+.4f}]  injected={inj:+.4f}")

    fig = corner.corner(samples, labels=labels, truths=[eps_inj, alpha_inj],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f")
    fig_path = OUT_DIR / f"inject_recover_{name}.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"  Corner saved: {fig_path}")
    return summary, samples


if __name__ == "__main__":
    sumA, _ = run_experiment("A_alpha05", eps_inj=0.5, alpha_inj=0.5)
    sumB, _ = run_experiment("B_alpha03", eps_inj=0.5, alpha_inj=0.3)

    pass_A = (sumA["alpha"][1] <= 0.5 <= sumA["alpha"][2]) and \
             (sumA["alpha"][1] > 0.0) and (sumA["alpha"][2] < 1.0)
    # 95% CI excludes 0.5
    pass_B = (sumB["alpha"][3] > 0.5) or (sumB["alpha"][4] < 0.5)

    lines = [
        "# Synthetic Injection-Recovery Results (Task #5)",
        "",
        "Pre-registered per Refined_Hypothesis.md §8.",
        f"Data: 80 k-bins log-spaced from 0.005 to 0.3 h/Mpc, sigma_P/P = {SIGMA_FRAC*100:.1f}% per bin.",
        f"Priors: eps in [0, 2], alpha in [-1, 2].",
        f"MCMC: emcee, 48 walkers, 2000 steps, 500 burn.",
        "",
        "## Experiment A: inject alpha = 0.5 (Riemann critical-line)",
        ""
    ]
    for lab in ["eps", "alpha"]:
        q50, q16, q84, q025, q975 = sumA[lab]
        lines.append(f"- **{lab}**: median = {q50:+.4f}; 68% CI = [{q16:+.4f}, {q84:+.4f}]; "
                     f"95% CI = [{q025:+.4f}, {q975:+.4f}]")
    lines += [
        "",
        "**Pass criterion (§8):** 68% CI of alpha encloses 0.5 AND excludes 0 and 1.",
        f"68% CI of alpha: [{sumA['alpha'][1]:+.4f}, {sumA['alpha'][2]:+.4f}]  ->  "
        f"**{'PASS' if pass_A else 'FAIL'}**",
        "",
        "## Experiment B: inject alpha = 0.3 (non-Riemann control)",
        ""
    ]
    for lab in ["eps", "alpha"]:
        q50, q16, q84, q025, q975 = sumB[lab]
        lines.append(f"- **{lab}**: median = {q50:+.4f}; 68% CI = [{q16:+.4f}, {q84:+.4f}]; "
                     f"95% CI = [{q025:+.4f}, {q975:+.4f}]")
    lines += [
        "",
        "**Pass criterion (§8):** 95% CI of alpha EXCLUDES 0.5 (no spurious recovery).",
        f"95% CI of alpha: [{sumB['alpha'][3]:+.4f}, {sumB['alpha'][4]:+.4f}]  ->  "
        f"**{'PASS' if pass_B else 'FAIL'}**",
        "",
        f"## Overall: {'BOTH PASS' if (pass_A and pass_B) else 'AT LEAST ONE FAILED'}",
    ]
    (OUT_DIR / "inject_recover_report.md").write_text("\n".join(lines), encoding="utf-8")
    print()
    for ln in lines[-6:]:
        print(ln)
