"""
Task #8 — Differentiation of DDEM from competing dark-sector models.

Produces:
    1. A side-by-side P(k) residual comparison: DDEM vs EDE vs S-IDE vs w0waCDM
       vs Frieman-Hill-Stebbins-Waga oscillating quintessence.
    2. A growth-rate signature plot: f sigma_8(z) for each model.
    3. A Markdown table summarizing the qualitative differences.

The competing-model templates are illustrative (analytic approximations of the
characteristic signature, not full Boltzmann output). The DDEM signature uses
the same comb code as predict_signature.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import numpy as np
import matplotlib.pyplot as plt
from ddem_background import riemann_zeros

OUT_DIR = Path(__file__).parent
H_PLANCK = 0.6774
ETA0_MPC = 14136.17
GAMMA_N = riemann_zeros(200)
K_N_H_MPC = GAMMA_N / ETA0_MPC / H_PLANCK


def ddem_residual(k_h, eps=0.5, alpha=0.5):
    rho_mod = np.sqrt(alpha**2 + GAMMA_N**2)
    amp = (0.5**alpha) / rho_mod
    sig = 0.02 * K_N_H_MPC
    gauss = np.exp(-0.5 * ((k_h[:, None] - K_N_H_MPC[None, :]) / sig[None, :])**2)
    return 2.0 * eps * np.sum(amp[None, :] * gauss, axis=1)


def ede_residual(k_h, f_ede=0.08):
    """Early Dark Energy: localized CMB phase shift, low-k suppression in P(k).
    Approximate as a single broad bump near k ~ 0.02 h/Mpc (matter-radiation eq scale)."""
    k_center = 0.02
    sigma = 0.5  # broad
    return -f_ede * np.exp(-0.5 * (np.log(k_h / k_center) / sigma)**2)


def sIDE_residual(k_h, xi=-0.1):
    """S-IDE (Silva et al.): sign-change in coupling -> broadband tilt in P(k)."""
    # one transition; signature is a smooth tilt with sign change near transition scale
    return xi * (np.log(k_h / 0.05))


def w0wa_residual(k_h, w0=-0.85, wa=-0.4):
    """w0waCDM: smooth growth-rate change, broadband BAO amplitude shift."""
    # very smooth modification, no oscillation
    return 0.02 * (w0 + 1) + 0.005 * wa * np.log(k_h / 0.05)


def fhsw_residual(k_h, f_amp=0.02, f_freq=8.0):
    """Frieman-Hill-Stebbins-Waga oscillating quintessence:
    single-frequency oscillation in P(k)."""
    return f_amp * np.cos(f_freq * np.log(k_h))


def make_plot():
    k_h = np.logspace(-2.0, np.log10(0.3), 2000)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(k_h, ddem_residual(k_h) * 100, "C0",  lw=0.7, label="DDEM (this work): comb")
    ax.plot(k_h, ede_residual(k_h)  * 100, "C1--", lw=1.3, label="EDE (single bump)")
    ax.plot(k_h, sIDE_residual(k_h) * 100, "C2:",  lw=1.3, label="S-IDE (one tilt)")
    ax.plot(k_h, w0wa_residual(k_h) * 100, "C3-.", lw=1.3, label="w0waCDM (smooth)")
    ax.plot(k_h, fhsw_residual(k_h) * 100, "C4--", lw=1.3, label="FHSW osc. quint. (single freq)")
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(+1, color="grey", lw=0.4, ls="--")
    ax.axhline(-1, color="grey", lw=0.4, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("(P / P_LCDM - 1) [%]")
    ax.set_title("Differentiation: DDEM comb vs competing dark-sector models")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(-5, 5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "differentiation_residuals.png", dpi=140)
    plt.close(fig)


def write_table():
    rows = [
        ("DDEM (this work)",
         "multi-frequency comb at k_n = γ_n/η_0",
         "ε · a^α / γ_n per peak",
         "multi-mode periodic in ln(a)",
         "Lagrangian: Type-2 conformal (phenomenological γ_n)",
         "perturbation-level oscillatory δQ; growth-rate ripples"),
        ("Early Dark Energy",
         "single transient burst near z_c ≈ 3500",
         "f_EDE up to ~0.09",
         "single CMB phase shift",
         "Lagrangian: scalar with V ∝ ϕ^(2n)",
         "no late-time comb"),
        ("S-IDE (Silva 2025)",
         "one sign-flip in coupling history",
         "ξ ~ ±0.1 (DR2-detected ~2σ)",
         "single transition",
         "Lagrangian: sign-changeable phenomenological Q",
         "one growth-rate kink"),
        ("w0waCDM",
         "smooth CPL EoS",
         "w0 + 1, wa free",
         "no oscillation",
         "phenomenological background parameterization",
         "smooth growth modification"),
        ("FHSW osc. quintessence",
         "single-frequency oscillation",
         "few-% amplitude",
         "monochromatic in ln(a) or φ/f",
         "Lagrangian: ultra-light PNGB cos(φ/f)",
         "single periodic feature"),
        ("Aref'eva-Volovich 2007",
         "zeta-zero spectrum in p-adic strings",
         "not constrained",
         "n/a",
         "Lagrangian: p-adic string",
         "no observational pipeline"),
    ]
    lines = ["# Differentiation Table: DDEM vs Competing Models",
             "",
             "| Model | P(k) signature | Amplitude | Time/scale structure | Lagrangian status | Perturbation observable |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines += [
        "",
        "## Key differentiating observable",
        "",
        "The signature that distinguishes DDEM from all competitors is the **multi-frequency**",
        "P(k) residual comb at k_n = γ_n/η_0, with the spacing density set by the Riemann-zero",
        "counting function (γ_n ≈ 2πn/ln(n) asymptotically). EDE has at most one bump.",
        "S-IDE has one kink. w0waCDM is smooth. FHSW oscillates at one frequency.",
        "",
        "**Where the difference is largest:** the linear regime 0.015 ≲ k ≲ 0.1 h/Mpc, where",
        "DDEM has predicted peaks at k_n for n ∈ [50, 200] while all competitors are smooth.",
        "This is the Tier 2 detection window from Refined_Hypothesis.md §7.",
    ]
    (OUT_DIR / "differentiation_table.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    make_plot()
    write_table()
    print("Differentiation plot:", OUT_DIR / "differentiation_residuals.png")
    print("Differentiation table:", OUT_DIR / "differentiation_table.md")
