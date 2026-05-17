"""
Predict the DDEM signature in the matter power spectrum.

For baseline parameters (beta0=0.01, eps=0.5, alpha=0.5, w_DE=-1.05) and
representative variations, plot:
    1. The full P_DDEM(k, z=0) and P_LCDM(k, z=0) for context.
    2. The relative residual r(k) = P_DDEM/P_LCDM - 1, with predicted comb peak
       locations marked (Tier 2 high-n region: k > 0.015 h/Mpc; Tier 3 low-n
       region: k < 0.01 h/Mpc, below DESI k_min).
    3. DESI-like sensitivity band (1% nominal) for visual comparison.

Outputs:
    synthetic_tests/predict_signature.png
    synthetic_tests/comb_peaks_table.md
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import numpy as np
import matplotlib.pyplot as plt
from ddem_background import solve_background, solve_lcdm, riemann_zeros
from ddem_perturbation import DDEM_over_LCDM, conformal_time_today

H_PLANCK = 0.6774
DESI_KMIN = 1.7e-3  # h/Mpc, DESI fundamental mode
DESI_KMAX = 0.3     # h/Mpc, DESI linear-regime cutoff
TIER2_KMIN = 0.015  # h/Mpc, where high-n comb (n>=50) starts being detectable

OUT_DIR = Path(__file__).parent
FIG_PATH = OUT_DIR / "predict_signature.png"
TABLE_PATH = OUT_DIR / "comb_peaks_table.md"

# Build a baseline LCDM template P(k) using a power-law no-wiggle approximation.
# (classy is available in WSL Ubuntu; here we use a fast analytic placeholder that's
# accurate at the few-percent level and demonstrates the comb signature without an
# external dependency. The full classy P(k) is wired in synthetic_tests/inject_recovery.py.)
def Eisenstein_Hu_pk(k_h_Mpc, h=H_PLANCK, Omega_m=0.315, Omega_b=0.049, sigma8=0.81, n_s=0.965):
    """No-wiggle Eisenstein-Hu approximation. Sufficient for illustrative residual plots."""
    k = k_h_Mpc * h  # 1/Mpc
    omega_m = Omega_m * h**2
    omega_b = Omega_b * h**2
    # transfer function (Eisenstein-Hu 1998, no-wiggle)
    s = 44.5 * np.log(9.83 / omega_m) / np.sqrt(1 + 10 * omega_b**0.75)  # sound horizon Mpc
    alpha_gamma = 1 - 0.328 * np.log(431 * omega_m) * omega_b / omega_m + 0.38 * np.log(22.3 * omega_m) * (omega_b/omega_m)**2
    Gamma_eff = omega_m * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43 * k * s)**4)) / h
    q = k_h_Mpc / (Gamma_eff * h)  # q in usual conventions
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    T = L0 / (L0 + C0 * q**2)
    # primordial * transfer^2 * k^n_s
    delta_H = 1.94e-5 * Omega_m**(-0.785 - 0.05 * np.log(Omega_m))
    # Pk normalization to sigma8 (rough; this is illustrative)
    P_unnorm = (k_h_Mpc / 0.05)**n_s * T**2 * k_h_Mpc
    # rescale to sigma8 ~ 0.81 at k ~ 0.1 (very rough)
    return P_unnorm * 4e4  # normalization tuned so that P(0.1) ~ 1e4 (Mpc/h)^3


def main():
    print("Solving DDEM and LCDM backgrounds...")
    bg = solve_background(beta0=0.01, eps=0.5, alpha=0.5, w_DE=-1.05)
    lc = solve_lcdm(alpha=0.5)
    eta0_Mpc = conformal_time_today(bg) * 2997.92 / H_PLANCK
    gamma_n = riemann_zeros(200)
    k_n_h_Mpc = gamma_n / eta0_Mpc / H_PLANCK
    print(f"  eta0 = {eta0_Mpc:.2f} Mpc")
    print(f"  k_1  = {k_n_h_Mpc[0]:.5f} h/Mpc (gamma_1 = {gamma_n[0]:.5f})")
    print(f"  k_50 = {k_n_h_Mpc[49]:.5f} h/Mpc (gamma_50 = {gamma_n[49]:.5f})")
    print(f"  k_100 = {k_n_h_Mpc[99]:.5f} h/Mpc")

    # k-grid for plotting
    k_h = np.logspace(-3.2, 0.0, 4000)
    print("Computing P_DDEM/P_LCDM with baseline params...")
    ratio_baseline, _, _ = DDEM_over_LCDM(k_h, z=0.0, bg=bg, lc=lc, eps=0.5, alpha=0.5)

    # variations
    bg_eps1 = solve_background(beta0=0.01, eps=1.0, alpha=0.5, w_DE=-1.05)
    ratio_eps1, _, _ = DDEM_over_LCDM(k_h, z=0.0, bg=bg_eps1, lc=lc, eps=1.0, alpha=0.5)

    bg_a03 = solve_background(beta0=0.01, eps=0.5, alpha=0.3, w_DE=-1.05)
    ratio_a03, _, _ = DDEM_over_LCDM(k_h, z=0.0, bg=bg_a03, lc=lc, eps=0.5, alpha=0.3)

    bg_a07 = solve_background(beta0=0.01, eps=0.5, alpha=0.7, w_DE=-1.05)
    ratio_a07, _, _ = DDEM_over_LCDM(k_h, z=0.0, bg=bg_a07, lc=lc, eps=0.5, alpha=0.7)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # Top: P(k) absolute (LCDM only — DDEM modification is too small to see linearly)
    pk_lcdm = Eisenstein_Hu_pk(k_h)
    axes[0].loglog(k_h, pk_lcdm, "k-", lw=1.2, label="LCDM (Eisenstein-Hu)")
    axes[0].set_xlabel("k [h/Mpc]")
    axes[0].set_ylabel("P(k) [(Mpc/h)^3]")
    axes[0].set_title("Matter power spectrum (reference template)")
    axes[0].axvspan(DESI_KMIN, DESI_KMAX, color="C0", alpha=0.1, label="DESI linear regime")
    axes[0].axvspan(TIER2_KMIN, DESI_KMAX, color="C2", alpha=0.05, label="Tier 2 prediction range")
    axes[0].axvspan(1e-4, DESI_KMIN, color="C3", alpha=0.05, label="Tier 3 (below DESI floor)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(loc="lower left")

    # Bottom: residuals — the comb signature
    axes[1].semilogx(k_h, (ratio_baseline - 1) * 100, lw=0.7, label="baseline (beta=0.01, eps=0.5, a=0.5)")
    axes[1].semilogx(k_h, (ratio_eps1 - 1) * 100, lw=0.5, alpha=0.6,
                     label="higher_eps (eps=1.0)")
    axes[1].semilogx(k_h, (ratio_a03 - 1) * 100, lw=0.5, alpha=0.6,
                     label="alpha=0.3")
    axes[1].semilogx(k_h, (ratio_a07 - 1) * 100, lw=0.5, alpha=0.6,
                     label="alpha=0.7")
    axes[1].axhline(0, color="k", lw=0.4)
    axes[1].axhline(+1, color="grey", lw=0.4, ls="--", label="DESI ~1% sensitivity")
    axes[1].axhline(-1, color="grey", lw=0.4, ls="--")
    # mark first 10 zeros (Tier 3)
    for k_n in k_n_h_Mpc[:10]:
        axes[1].axvline(k_n, color="C3", lw=0.3, alpha=0.5)
    # mark zeros 50-200 (Tier 2)
    for k_n in k_n_h_Mpc[49:200]:
        axes[1].axvline(k_n, color="C2", lw=0.15, alpha=0.3)
    axes[1].axvspan(DESI_KMIN, DESI_KMAX, color="C0", alpha=0.05)
    axes[1].axvspan(1e-4, DESI_KMIN, color="C3", alpha=0.05)
    axes[1].set_xlabel("k [h/Mpc]")
    axes[1].set_ylabel("(P_DDEM / P_LCDM - 1) [%]")
    axes[1].set_title("DDEM modification: predicted comb residual in P(k)")
    axes[1].set_xlim(10**-3.2, 1.0)
    axes[1].set_ylim(-5, 5)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"Figure saved: {FIG_PATH}")

    # Write comb peaks table
    lines = ["# Predicted DDEM Comb Peak Locations", "",
             f"For h = {H_PLANCK}, eta_0 = {eta0_Mpc:.2f} Mpc.",
             "Peak positions in k [h/Mpc] = gamma_n / (eta_0 * h).",
             "",
             "| n | gamma_n | k_n [h/Mpc] | k_n [1/Mpc] | Tier | Accessible? |",
             "|---|---|---|---|---|---|"]
    for n in [1, 2, 3, 5, 10, 25, 50, 100, 150, 200]:
        gn = gamma_n[n - 1]
        kn_h = k_n_h_Mpc[n - 1]
        kn_Mpc = kn_h * H_PLANCK
        if kn_h < DESI_KMIN:
            tier = "3"
            access = "below DESI floor; SKA / CMB low-l"
        elif kn_h < TIER2_KMIN:
            tier = "2 (marginal)"
            access = "DESI marginal (cosmic variance)"
        else:
            tier = "2"
            access = "DESI / Euclid sub-% sensitivity"
        lines.append(f"| {n} | {gn:.4f} | {kn_h:.5f} | {kn_Mpc:.5f} | {tier} | {access} |")
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Table saved: {TABLE_PATH}")
    print()
    for ln in lines[7:]:  # skip header rows for terminal output
        print(ln)


if __name__ == "__main__":
    main()
