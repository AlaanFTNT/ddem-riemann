"""
Task #5 final — Injection-recovery with TWO redshifts (z=0 and z=1).

The single-redshift fit (v2) showed a strong (eps, alpha) degeneracy because
the predicted comb amplitude scales as ~ eps * a_eff^alpha. At fixed a_eff,
eps and a^alpha are degenerate.

Two-redshift data breaks the degeneracy: signal ratio at z=0 to z=1 is
~ (a=1)^alpha / (a=0.5)^alpha = 2^alpha, which depends only on alpha.

This is exactly what DESI DR2 provides — tracer samples at multiple z. The
multi-z fit mirrors the realistic analysis the paper will run.

We inject at z=0 (a=1) AND z=1 (a=0.5) and fit jointly.
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
ETA0_MPC = 14136.17
N_ZEROS = 200
SIGMA_FACTOR = 0.02

GAMMA_N = riemann_zeros(N_ZEROS)
K_N_H_MPC = GAMMA_N / ETA0_MPC / H_PLANCK


def fast_ratio(k_h, eps, alpha, a_eff):
    rho_mod = np.sqrt(alpha**2 + GAMMA_N**2)
    amp = a_eff**alpha / rho_mod
    sig = SIGMA_FACTOR * K_N_H_MPC
    gauss = np.exp(-0.5 * ((k_h[:, None] - K_N_H_MPC[None, :]) / sig[None, :])**2)
    return 1.0 + 2.0 * eps * np.sum(amp[None, :] * gauss, axis=1)


def run(name, eps_inj, alpha_inj, sigma_frac, n_kbins, seed):
    np.random.seed(seed)
    k_grid = np.logspace(np.log10(0.005), np.log10(0.3), n_kbins)
    Pk_template_z0 = Eisenstein_Hu_pk(k_grid)
    # at z=1 the template scales by D(z=1)/D(z=0) for LCDM ~ 0.6 (matter-dominated growth)
    # for the synthetic test we use a simple LCDM growth factor approximation
    D_lcdm_z1_over_z0 = 0.61
    Pk_template_z1 = Pk_template_z0 * D_lcdm_z1_over_z0**2

    # inject: a_eff for z=0 is 1.0, for z=1 is 0.5
    ratio_z0_true = fast_ratio(k_grid, eps_inj, alpha_inj, a_eff=1.0)
    ratio_z1_true = fast_ratio(k_grid, eps_inj, alpha_inj, a_eff=0.5)
    Pk_z0_true = Pk_template_z0 * ratio_z0_true
    Pk_z1_true = Pk_template_z1 * ratio_z1_true
    sigma_z0 = sigma_frac * Pk_z0_true
    sigma_z1 = sigma_frac * Pk_z1_true
    Pk_z0 = Pk_z0_true + np.random.normal(0, sigma_z0)
    Pk_z1 = Pk_z1_true + np.random.normal(0, sigma_z1)

    def log_prior(theta):
        eps, alpha = theta
        if not (0.0 <= eps <= 3.0): return -np.inf
        if not (-1.0 <= alpha <= 2.0): return -np.inf
        return 0.0

    def log_like(theta):
        eps, alpha = theta
        r0 = fast_ratio(k_grid, eps, alpha, a_eff=1.0)
        r1 = fast_ratio(k_grid, eps, alpha, a_eff=0.5)
        chi2 = (np.sum(((Pk_z0 - Pk_template_z0 * r0) / sigma_z0)**2) +
                np.sum(((Pk_z1 - Pk_template_z1 * r1) / sigma_z1)**2))
        return -0.5 * chi2

    def log_post(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + log_like(theta)

    nwalkers, nsteps, burn = 64, 3000, 800
    p0 = np.column_stack([
        np.random.uniform(0.1, 2.5, nwalkers),
        np.random.uniform(-0.8, 1.8, nwalkers),
    ])
    sampler = emcee.EnsembleSampler(nwalkers, 2, log_post)
    t0 = time.time()
    sampler.run_mcmc(p0, nsteps, progress=False)
    samples = sampler.get_chain(discard=burn, flat=True)
    q = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=0)
    print(f"\n[{name}]  inject (eps={eps_inj}, alpha={alpha_inj}, sig={sigma_frac*100:.2f}%)  "
          f"wall={time.time()-t0:.1f}s  accept={np.mean(sampler.acceptance_fraction):.3f}")
    out = {}
    for i, lab in enumerate(["eps", "alpha"]):
        q025, q16, q50, q84, q975 = q[:, i]
        out[lab] = (q50, q16, q84, q025, q975)
        inj = [eps_inj, alpha_inj][i]
        print(f"  {lab}: med={q50:+.4f}  68%=[{q16:+.4f},{q84:+.4f}]  95%=[{q025:+.4f},{q975:+.4f}]  inj={inj:+.4f}")

    # Persist raw chain for downstream KDE / posterior plotting
    np.save(OUT_DIR / f"chain_v3_{name}.npy", samples)

    fig = corner.corner(samples, labels=["eps", "alpha"],
                        truths=[eps_inj, alpha_inj], quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_fmt=".3f")
    fig.savefig(OUT_DIR / f"inject_recover_v3_{name}.png", dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    R1 = run("multiz_dr2like_a05",  eps_inj=0.5, alpha_inj=0.5, sigma_frac=0.015, n_kbins=80,  seed=20260514)
    R1b = run("multiz_dr2like_a03", eps_inj=0.5, alpha_inj=0.3, sigma_frac=0.015, n_kbins=80,  seed=20260514)
    R2 = run("multiz_dr3like_a05",  eps_inj=0.5, alpha_inj=0.5, sigma_frac=0.005, n_kbins=200, seed=20260514)
    R3 = run("multiz_boosted_a05",  eps_inj=1.5, alpha_inj=0.5, sigma_frac=0.005, n_kbins=200, seed=20260514)
    R3b = run("multiz_boosted_a03", eps_inj=1.5, alpha_inj=0.3, sigma_frac=0.005, n_kbins=200, seed=20260514)

    def passes_A(s):
        return s["alpha"][1] <= 0.5 <= s["alpha"][2] and s["alpha"][1] > 0 and s["alpha"][2] < 1
    def passes_B(s):
        return s["alpha"][3] > 0.5 or s["alpha"][4] < 0.5

    rows = [
        ("multiz DR2-like alpha=0.5", 0.5, 0.5, 1.5, 80,  R1,  passes_A(R1),  "A"),
        ("multiz DR2-like alpha=0.3", 0.3, 0.5, 1.5, 80,  R1b, passes_B(R1b), "B"),
        ("multiz DR3-like alpha=0.5", 0.5, 0.5, 0.5, 200, R2,  passes_A(R2),  "A"),
        ("multiz boost alpha=0.5",    0.5, 1.5, 0.5, 200, R3,  passes_A(R3),  "A"),
        ("multiz boost alpha=0.3",    0.3, 1.5, 0.5, 200, R3b, passes_B(R3b), "B"),
    ]
    lines = [
        "# Injection-Recovery — Multi-Redshift (Task #5 final)",
        "",
        "Pre-registered per Refined_Hypothesis.md §8. Data at z=0 AND z=1.",
        "",
        "| Regime | alpha_inj | eps_inj | sigma | n_k | Recovered alpha (68% CI) | Test | Pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, a_i, e_i, sig, nk, res, pf, t in rows:
        q50, q16, q84, _, _ = res["alpha"]
        lines.append(f"| {name} | {a_i} | {e_i} | {sig}% | {nk} | "
                     f"{q50:+.3f} [{q16:+.3f}, {q84:+.3f}] | {t} | "
                     f"{'PASS' if pf else 'FAIL'} |")
    lines += [
        "",
        "Multi-redshift data breaks the (eps, alpha) degeneracy because the signal",
        "amplitude scales as a_eff^alpha. At a=1.0 vs a=0.5, the ratio of comb",
        "amplitudes is 2^alpha, which depends only on alpha.",
    ]
    (OUT_DIR / "inject_recover_v3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print()
    for r in rows:
        print(f"  {r[0]:32s}  test {r[7]}  -> {'PASS' if r[6] else 'FAIL'}")
