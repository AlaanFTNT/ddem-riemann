"""Build all eight Paper 1 figures with a consistent visual style.

Outputs to this directory:
    figure1_framework.png        — schematic of paper's logical structure
    figure2_doom_factor.png      — d(a) stability proof
    figure3_comb_prediction.png  — predicted P(k) comb residual
    figure4_pipeline.png         — pipeline architecture
    figure5_alpha_posteriors.png — multi-z injection-recovery 1D marginals
    figure6_sensitivity.png      — sigma(alpha) vs noise level
    figure7_dr2_corner.png       — DESI DR2 BAO MCMC posterior corner plot
    figure8_differentiation.png  — DDEM vs EDE / S-IDE / w0waCDM / FHSW

Style is set by paper_style.py (palette, font, grid, no top/right spines).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # ddem-riemann
sys.path.insert(0, str(ROOT / "theory"))
sys.path.insert(0, str(ROOT / "synthetic_tests"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.stats import norm
from paper_style import apply_style, COLORS, SIZE_SINGLE_COL, SIZE_DOUBLE_COL, SIZE_SQUARE, SIZE_TALL, label_panel

apply_style()

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Figure 1 — Framework schematic
# ---------------------------------------------------------------------------
def figure1_framework():
    """Clean three-tier framework diagram.

    Colour semantics across the schematic:
        input  (purple)  — mathematical motivation, raw premises
        process (teal)   — derivation steps, Lagrangian, kernel
        output (orange)  — testable artifacts: stability, comb, fits, forecasts
    """
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11)
    ax.set_axis_off()
    ax.set_facecolor("white"); fig.set_facecolor("white")

    C_IN  = COLORS["accent"]     # purple
    C_PR  = COLORS["tertiary"]   # teal
    C_OUT = COLORS["secondary"]  # orange

    def box(x, y, w, h, text, color, fc=None, fontsize=10, weight="normal"):
        if fc is None:
            fc = color + "1A"
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.4, edgecolor=color, facecolor=fc)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color="#1a1a1a", weight=weight)

    def arrow(x1, y1, x2, y2, color="#555555", lw=1.1):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", mutation_scale=14,
                            linewidth=lw, color=color, shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    ax.text(6.0, 10.5, "Framework structure", ha="center", va="bottom",
            fontsize=13, weight="bold", color="#1a1a1a")

    # Row 1: inputs (purple)
    box(3.0, 9.4, 4.6, 1.0, "GUE universality\n(Montgomery–Odlyzko 1973, 1987)", C_IN, fontsize=10)
    box(9.0, 9.4, 4.6, 1.0, "Riemann zero spectrum {γₙ}\ncanonical GUE-class realization", C_IN, fontsize=10)

    # Row 2: derivation (teal)
    box(3.0, 7.5, 4.6, 1.0, "Pourtsidou Type-2 conformal\ncoupled quintessence  (§2.1)", C_PR, fontsize=10)
    box(9.0, 7.5, 4.6, 1.0, "von Mangoldt explicit formula\n(number theory)  (§2.2)", C_PR, fontsize=10)
    arrow(3.0, 8.9, 3.0, 8.0)
    arrow(9.0, 8.9, 9.0, 8.0)

    # Row 3: derived Q kernel (mathtext, simple brackets — \Bigl not in matplotlib)
    eq_text = (r"$Q(a) = \beta_0\,H_0\,\rho_{\rm DE}\,[\,1 + 2\varepsilon "
               r"\sum_n a^{\alpha}\,\cos(\gamma_n \ln a)\,/\,|\rho_n|\,]$"
               "\n"
               r"(Eq. 7)")
    box(6.0, 5.6, 8.6, 1.1, eq_text, C_PR, fc="#ffffff", fontsize=11, weight="bold")
    arrow(3.6, 7.0, 5.0, 6.2)
    arrow(8.4, 7.0, 7.0, 6.2)

    # Row 4: three derived analyses (orange)
    box(2.0, 3.5, 3.2, 1.0, "Stability proof\nVMM doom factor (§3)", C_OUT, fontsize=10)
    box(6.0, 3.5, 3.2, 1.0, "Comb signature\n$k_n = \\gamma_n / \\eta_0$  (§4)", C_OUT, fontsize=10)
    box(10.0, 3.5, 3.2, 1.0, "CLASS modification\nbackground patch (§5)", C_OUT, fontsize=10)
    arrow(5.0, 5.05, 2.5, 4.05)
    arrow(6.0, 5.05, 6.0, 4.05)
    arrow(7.0, 5.05, 9.5, 4.05)

    # Row 5: operational outputs (orange)
    box(3.5, 1.4, 4.4, 1.0, "Forecasts (§6)\nmulti-z injection-recovery", C_OUT, fontsize=10)
    box(8.5, 1.4, 4.4, 1.0, "Preliminary DR2 BAO fit (§7)\nconsistency check", C_OUT, fontsize=10)
    arrow(2.5, 2.95, 3.2, 1.95)
    arrow(6.0, 2.95, 4.2, 1.95)
    arrow(10.0, 2.95, 9.2, 1.95)

    # Legend (inputs / derivation / outputs)
    leg_y = 0.30
    ax.add_patch(FancyBboxPatch((0.4, leg_y - 0.10), 0.4, 0.30,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=C_IN + "1A", edgecolor=C_IN, lw=1.0))
    ax.text(0.95, leg_y + 0.05, "inputs", fontsize=9, va="center")
    ax.add_patch(FancyBboxPatch((2.4, leg_y - 0.10), 0.4, 0.30,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=C_PR + "1A", edgecolor=C_PR, lw=1.0))
    ax.text(2.95, leg_y + 0.05, "derivation", fontsize=9, va="center")
    ax.add_patch(FancyBboxPatch((4.6, leg_y - 0.10), 0.4, 0.30,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=C_OUT + "1A", edgecolor=C_OUT, lw=1.0))
    ax.text(5.15, leg_y + 0.05, "outputs / testables", fontsize=9, va="center")

    fig.savefig(OUT / "figure1_framework.png", facecolor="white")
    plt.close(fig)
    print("  figure1_framework.png")

# ---------------------------------------------------------------------------
# Figure 2 — Doom factor d(a)
# ---------------------------------------------------------------------------
def figure2_doom_factor():
    from ddem_background import riemann_zeros
    GAMMA_N = riemann_zeros(200)

    OMEGA_M, OMEGA_R, OMEGA_L = 0.315, 9e-5, 1.0 - 0.315 - 9e-5
    def hubble(a): return np.sqrt(OMEGA_M*a**-3 + OMEGA_R*a**-4 + OMEGA_L)
    def bracket(a, alpha, eps, N=200):
        rho_mod = np.sqrt(alpha**2 + GAMMA_N[:N]**2)
        a_arr = np.atleast_1d(a)
        phase = np.outer(np.log(a_arr), GAMMA_N[:N])
        return 1.0 + 2.0 * eps * (a_arr**alpha) * np.sum(np.cos(phase) / rho_mod, axis=1)
    def d_of_a(a, beta0, eps, alpha, w_DE):
        return beta0 * bracket(a, alpha, eps) / (3.0 * hubble(np.atleast_1d(a)) * (1.0 + w_DE))

    a = np.logspace(-3, 0, 5000)

    # Sets ordered so canonical (small) renders on top (drawn last).
    fig, ax = plt.subplots(figsize=SIZE_SINGLE_COL)
    sets = [
        ("β₀=0.05, ε=0.5, α=0.5, w=−1.05 (higher β₀, fails at a=1)", 0.05, 0.5, 0.5, -1.05, COLORS["secondary"], 1.2),
        ("β₀=0.01, ε=0.5, α=0.5, w=−1.01 (weakly phantom, fails at a=1)", 0.01, 0.5, 0.5, -1.01, COLORS["accent"], 1.2),
        ("β₀=0.01, ε=1.0, α=0.5, w=−1.05 (higher ε, passes)",  0.01, 1.0, 0.5, -1.05, COLORS["tertiary"], 1.4),
        ("β₀=0.01, ε=0.5, α=0.5, w=−1.05 (canonical, passes)",  0.01, 0.5, 0.5, -1.05, COLORS["primary"],   2.0),
    ]
    for label, b, e, al, w, c, lw in sets:
        d = d_of_a(a, b, e, al, w)
        ax.plot(a, d, color=c, lw=lw, label=label, alpha=0.95)

    threshold = 2 * np.pi / GAMMA_N[0]
    ax.axhline(0.0,         color="#222222", lw=0.6, ls="-")
    ax.axhline(+threshold,  color=COLORS["highlight"], lw=1.0, ls="--",
               label=f"±2π/γ₁ ≈ {threshold:.3f}  (stability threshold)")
    ax.axhline(-threshold,  color=COLORS["highlight"], lw=1.0, ls="--")

    # Shaded "stable" region
    ax.axhspan(-threshold, threshold, color=COLORS["highlight"], alpha=0.06, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("scale factor  a")
    ax.set_ylabel("doom factor  d(a)")
    ax.set_title("Background-level stability of the DDEM oscillatory Q kernel")
    ax.set_ylim(-0.6, 0.6)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=8.0)

    fig.savefig(OUT / "figure2_doom_factor.png")
    plt.close(fig)
    print("  figure2_doom_factor.png")

# ---------------------------------------------------------------------------
# Figure 3 — Comb prediction in P(k)
# ---------------------------------------------------------------------------
def figure3_comb_prediction():
    from ddem_background import riemann_zeros
    GAMMA_N = riemann_zeros(200)
    h = 0.6774
    eta0 = 14136.17
    k_n = GAMMA_N / eta0 / h

    # Simplified comb residual
    def comb_residual(k_h, eps=0.5, alpha=0.5, a_eff=0.5, sigma_factor=0.02):
        rho_mod = np.sqrt(alpha**2 + GAMMA_N**2)
        amp = a_eff**alpha / rho_mod
        sig = sigma_factor * k_n
        gauss = np.exp(-0.5 * ((k_h[:, None] - k_n[None, :]) / sig[None, :])**2)
        return 2.0 * eps * np.sum(amp[None, :] * gauss, axis=1)

    k_h = np.logspace(-3.2, 0.0, 6000)
    r_canon = comb_residual(k_h, eps=0.5, alpha=0.5) * 100
    r_a03   = comb_residual(k_h, eps=0.5, alpha=0.3) * 100
    r_a07   = comb_residual(k_h, eps=0.5, alpha=0.7) * 100

    DESI_KMIN = 1.7e-3
    TIER2_KMIN = 0.015
    DESI_KMAX = 0.3

    fig, ax = plt.subplots(figsize=SIZE_DOUBLE_COL)

    # Tier shading
    ax.axvspan(1e-4, DESI_KMIN, color=COLORS["tier3"], alpha=0.08, label="Tier 3 (k below DESI floor)")
    ax.axvspan(DESI_KMIN, TIER2_KMIN, color=COLORS["tier1"], alpha=0.10, label="Tier 1/2 transition (marginal)")
    ax.axvspan(TIER2_KMIN, DESI_KMAX, color=COLORS["tier2"], alpha=0.10, label="Tier 2 (DESI / Euclid linear regime)")

    # Curves
    ax.plot(k_h, r_canon, color=COLORS["primary"],   lw=1.0, label="α = 0.5  (Riemann critical line)")
    ax.plot(k_h, r_a03,   color=COLORS["secondary"], lw=1.0, ls="--", alpha=0.7, label="α = 0.3  (control)")
    ax.plot(k_h, r_a07,   color=COLORS["tertiary"],  lw=1.0, ls=":",  alpha=0.7, label="α = 0.7  (control)")

    # First 10 zeros as ticks at the bottom
    for kn in k_n[:10]:
        ax.axvline(kn, color=COLORS["tier3"], lw=0.5, alpha=0.5)
    # Zeros 50-200 as fainter ticks
    for kn in k_n[49:200]:
        ax.axvline(kn, color=COLORS["tier2"], lw=0.15, alpha=0.3)

    # 1% sensitivity line
    ax.axhline(+1.0, color=COLORS["control"], lw=0.7, ls="--", alpha=0.7, label="DESI ≈ 1% sensitivity")
    ax.axhline(-1.0, color=COLORS["control"], lw=0.7, ls="--", alpha=0.7)
    ax.axhline(0.0,  color="#222222", lw=0.5)

    ax.set_xscale("log")
    ax.set_xlim(10**-3.2, 1.0)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("wavenumber  k  [h Mpc⁻¹]")
    ax.set_ylabel("P_DDEM / P_ΛCDM − 1   [%]")
    ax.set_title("Predicted comb signature in the matter power spectrum")
    ax.legend(loc="upper left", framealpha=0.95, ncol=2, fontsize=8.5)

    fig.savefig(OUT / "figure3_comb_prediction.png")
    plt.close(fig)
    print("  figure3_comb_prediction.png")

# ---------------------------------------------------------------------------
# Figure 4 — Pipeline architecture
# ---------------------------------------------------------------------------
def figure4_pipeline():
    """Clean two-column pipeline diagram.

    Left column: theory side (model -> CLASS -> predictions).
    Right column: data side (DESI DR2 BAO -> covariance treatment).
    Both converge to likelihood, then emcee, then two outputs.
    Colour semantics: input (purple), process (teal), output (orange).
    """
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 11)
    ax.set_axis_off()
    ax.set_facecolor("white"); fig.set_facecolor("white")

    C_IN  = COLORS["accent"]
    C_PR  = COLORS["tertiary"]
    C_OUT = COLORS["secondary"]
    C_HL  = COLORS["highlight"]

    def box(x, y, w, h, text, color, fc=None, fontsize=10, weight="normal"):
        if fc is None: fc = color + "1A"
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.4, edgecolor=color, facecolor=fc)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color="#1a1a1a", weight=weight)

    def arrow(x1, y1, x2, y2, color="#555555", lw=1.1):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", mutation_scale=14,
                            linewidth=lw, color=color, shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    # Title
    ax.text(6.0, 10.5, "MCMC analysis pipeline", ha="center", va="bottom",
            fontsize=13, weight="bold", color="#1a1a1a")

    # ---- Top row: model parameters (purple) ---------------------------------
    box(6.0, 9.4, 5.4, 1.4,
        r"Model parameters" + "\n" + r"$\theta = \{\beta_0,\varepsilon,\alpha,h,\omega_b,\omega_{cdm},w_{0\rm fld}\}$",
        C_IN, fontsize=10)

    # ---- Theory + emulator (teal) -------------------------------------------
    box(6.0, 7.8, 6.0, 1.0,
        "Python emulator (§5.2)\nbackground ODE + linear growth + comb signature + $f\\sigma_8$ + $d_L(z)$",
        C_PR, fontsize=10)
    arrow(6.0, 8.7, 6.0, 8.3)

    # ---- Six likelihood inputs (purple data) --------------------------------
    # Two-row layout: 3 boxes top (y=6.0), 3 boxes bottom (y=4.7), to keep
    # boxes legible without crowding.
    box(2.20, 6.10, 2.80, 1.0, "DESI DR2 BAO\n7 bins, 12 obs.", C_IN, fontsize=8.5)
    box(6.40, 6.10, 2.80, 1.0, "Planck 2018 marginals\n$(h, \\omega_b, \\omega_{cdm})$", C_IN, fontsize=8.5)
    box(10.60, 6.10, 2.80, 1.0, "Compiled $f\\sigma_8$\n13 z-bins", C_IN, fontsize=8.5)
    box(2.20, 4.75, 2.80, 1.0, "$P(k)$ feature bound\nBeutler+ 2023", C_IN, fontsize=8.5)
    box(6.40, 4.75, 2.80, 1.0, "Qu+ 2025 CMB lensing\n$S_8^{\\rm CMBL}$", C_IN, fontsize=8.5)
    box(10.60, 4.75, 2.80, 1.0, "Pantheon+ SNe\n1580 SNe, M$_B$ marg.", C_IN, fontsize=8.5)

    # ---- Joint likelihood (highlight) ---------------------------------------
    box(6.4, 3.4, 9.2, 0.9,
        r"Joint $\chi^2 = \chi^2_{\rm BAO} + \chi^2_{\rm Planck} + \chi^2_{f\sigma_8} + \chi^2_{\rm feature} + \chi^2_{\rm CMBL} + \chi^2_{\rm SN}$",
        C_HL, fontsize=9.5, weight="bold")
    arrow(2.20, 5.60, 3.5, 3.85)
    arrow(6.40, 5.60, 6.0, 3.85)
    arrow(10.60, 5.60, 8.8, 3.85)
    arrow(2.20, 4.25, 4.0, 3.85)
    arrow(6.40, 4.25, 6.4, 3.85)
    arrow(10.60, 4.25, 8.5, 3.85)

    # ---- emcee + outputs (orange) ------------------------------------------
    box(6.0, 1.9, 4.5, 0.9,
        "emcee Ensemble Sampler\n64 walkers × 3000 steps, 12 cores", C_PR,
        fontsize=9.5, weight="normal")
    arrow(6.4, 2.95, 6.0, 2.35)

    box(2.7, 0.4, 3.8, 0.8, "Joint posterior chain\n(7 parameters)", C_OUT, fontsize=9.5)
    box(9.3, 0.4, 3.8, 0.8, "Constraint on $\\alpha$\n(testing $\\alpha = 1/2$)", C_OUT, fontsize=9.5)
    arrow(5.2, 1.45, 3.3, 0.85)
    arrow(6.8, 1.45, 8.7, 0.85)

    fig.savefig(OUT / "figure4_pipeline.png", facecolor="white")
    plt.close(fig)
    print("  figure4_pipeline.png")

# ---------------------------------------------------------------------------
# Figure 5 — α posteriors across regimes (1D marginals)
# ---------------------------------------------------------------------------
def figure5_alpha_posteriors():
    """1D posterior of alpha for each multi-z injection-recovery regime,
    rendered as Gaussian KDE on the raw chain samples produced by
    synthetic_tests/inject_recover_v3_multiz.py.

    If a chain file is missing, that regime is skipped with a console warning;
    the figure still renders for whatever chains are persisted.
    """
    from scipy.stats import gaussian_kde

    CHAIN_DIR = ROOT / "synthetic_tests"
    regimes = [
        # (chain_name,                       label,                            alpha_inj, color)
        ("multiz_dr2like_a05",  r"DR2-like, $\alpha_\mathrm{inj}=0.5$",       0.5, COLORS["secondary"]),
        ("multiz_dr2like_a03",  r"DR2-like, $\alpha_\mathrm{inj}=0.3$ (control)", 0.3, COLORS["control"]),
        ("multiz_dr3like_a05",  r"DR3-like, $\alpha_\mathrm{inj}=0.5$",       0.5, COLORS["primary"]),
        ("multiz_boosted_a05",  r"Boosted, $\alpha_\mathrm{inj}=0.5$",        0.5, COLORS["tertiary"]),
        ("multiz_boosted_a03",  r"Boosted, $\alpha_\mathrm{inj}=0.3$ (control)", 0.3, COLORS["highlight"]),
    ]

    alpha_axis = np.linspace(-1, 2, 1500)

    fig, ax = plt.subplots(figsize=SIZE_SINGLE_COL)
    rendered = 0
    for chain_name, label, ai, c in regimes:
        chain_path = CHAIN_DIR / f"chain_v3_{chain_name}.npy"
        if not chain_path.exists():
            print(f"  [figure5] missing chain: {chain_path.name} — skipping")
            continue
        chain = np.load(chain_path)             # (nsamples, 2): [eps, alpha]
        alpha_samples = chain[:, 1]
        kde = gaussian_kde(alpha_samples, bw_method=0.20)
        y = kde(alpha_axis)
        ax.plot(alpha_axis, y, color=c, lw=1.6, label=label)
        ax.axvline(ai, color=c, lw=0.6, ls=":", alpha=0.7)
        rendered += 1

    if rendered == 0:
        raise RuntimeError(
            "No chain_v3_*.npy files found. Run "
            "synthetic_tests/inject_recover_v3_multiz.py first."
        )

    ax.axvline(0.5, color="#222222", lw=1.2, ls="--",
               label=r"$\alpha = 1/2$ (Riemann critical line)")
    ax.set_xlim(-1, 2)
    ax.set_xlabel(r"dark-sector spectrum real part  $\alpha$")
    ax.set_ylabel(r"posterior density  $P(\alpha)$")
    ax.set_title(r"$\alpha$ posteriors across forecast regimes (KDE on raw chains)")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)

    fig.savefig(OUT / "figure5_alpha_posteriors.png")
    plt.close(fig)
    print("  figure5_alpha_posteriors.png")

# ---------------------------------------------------------------------------
# Figure 6 — Sensitivity curves σ(α) vs noise level
# ---------------------------------------------------------------------------
def figure6_sensitivity():
    """Empirical σ(α) from injection-recovery, plotted vs σ_P/P per bin.
    Two curves: ε = 0.5 (canonical) and ε = 1.5 (boosted)."""
    # Data points from inject_recover_v3 results (sigma alpha = (q84-q16)/2)
    canonical = {
        "sigma_P_pct": np.array([1.5,  0.5]),
        "sigma_alpha": np.array([0.235, 0.050]),
    }
    boosted = {
        "sigma_P_pct": np.array([0.5]),
        "sigma_alpha": np.array([0.015]),
    }

    # Interpolated curve (rough scaling sigma_alpha ~ sigma_P_pct^1.1 * 0.15 for canonical)
    s = np.logspace(np.log10(0.1), np.log10(3.0), 200)
    canonical_curve = 0.16 * (s / 1.0)**1.0   # rough fit
    boosted_curve   = 0.026 * (s / 0.5)**1.0  # rough fit anchored to (0.5%, 0.015)

    fig, ax = plt.subplots(figsize=SIZE_SINGLE_COL)
    ax.loglog(s, canonical_curve, "-", color=COLORS["primary"], lw=1.6, label="ε = 0.5  (canonical coupling)")
    ax.loglog(s, boosted_curve,   "-", color=COLORS["tertiary"], lw=1.6, label="ε = 1.5  (boosted coupling)")
    ax.plot(canonical["sigma_P_pct"], canonical["sigma_alpha"], "o", color=COLORS["primary"],
            markersize=8, markeredgecolor="white", markeredgewidth=1.2, zorder=3,
            label="canonical: measured σ(α)")
    ax.plot(boosted["sigma_P_pct"], boosted["sigma_alpha"], "s", color=COLORS["tertiary"],
            markersize=8, markeredgecolor="white", markeredgewidth=1.2, zorder=3,
            label="boosted: measured σ(α)")

    # Threshold for "detection" of α = 1/2 — say σ(α) < 0.1 means distinguishable from 0
    ax.axhline(0.1, color=COLORS["highlight"], lw=1.0, ls="--", alpha=0.8,
               label="σ(α) = 0.1 (detection threshold for α = 1/2)")

    # Annotate DR2 and DR3 noise levels
    ax.axvline(1.5, color=COLORS["secondary"], lw=0.8, ls=":", alpha=0.7)
    ax.text(1.55, 0.012, "DESI DR2\n(~1.5%/bin)", color=COLORS["secondary"], fontsize=8)
    ax.axvline(0.5, color=COLORS["tier2"], lw=0.8, ls=":", alpha=0.7)
    ax.text(0.52, 0.012, "DESI DR3 /\nEuclid DR1\n(~0.5%/bin)", color=COLORS["tier2"], fontsize=8)

    ax.set_xlabel("per-bin noise  σ_P / P   [%]")
    ax.set_ylabel("recovered  σ(α)")
    ax.set_title("Forecast α uncertainty vs data sensitivity")
    ax.set_xlim(0.1, 3.0)
    ax.set_ylim(1e-2, 1e0)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    fig.savefig(OUT / "figure6_sensitivity.png")
    plt.close(fig)
    print("  figure6_sensitivity.png")

# ---------------------------------------------------------------------------
# Figure 7 — DR2 BAO posterior corner
# ---------------------------------------------------------------------------
def figure7_dr2_corner():
    """Joint posterior corner plot from the four-likelihood MCMC.

    Source: cobaya/chain_joint.npy if present (preferred); falls back to
    cobaya/chain_desi_bao.npy for the BAO-only chain if the joint chain
    has not been produced yet.
    """
    import corner
    joint_path = ROOT / "cobaya" / "chain_joint.npy"
    bao_path   = ROOT / "cobaya" / "chain_desi_bao.npy"
    if joint_path.exists():
        chain = np.load(joint_path)
        title = (f"Joint posterior (BAO + Planck + $f\\sigma_8$ + Beutler + ACT DR6 lensing),"
                 f"  {chain.shape[0]:,} samples")
    elif bao_path.exists():
        chain = np.load(bao_path)
        title = f"DESI DR2 BAO-only posterior, {chain.shape[0]:,} samples"
    else:
        print(f"  WARNING: no chain found; skipping figure 7")
        return

    labels = [r"$\beta_0$", r"$\varepsilon$", r"$\alpha$", r"$h$",
              r"$\omega_b$", r"$\omega_{cdm}$", r"$w_{0,\,\rm fld}$"]

    fig = corner.corner(
        chain, labels=labels,
        truths=[None, None, 0.5, 0.6774, 0.02236, 0.1200, -1.0],
        truth_color=COLORS["highlight"],
        color=COLORS["primary"],
        plot_density=True, plot_datapoints=False,
        fill_contours=True,
        levels=(0.683, 0.954),
        hist_kwargs={"color": COLORS["primary"], "lw": 1.0},
        contour_kwargs={"colors": [COLORS["primary"]]},
        contourf_kwargs={"colors": ["white", COLORS["primary"] + "33", COLORS["primary"] + "77"]},
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
    )
    fig.set_size_inches(9.5, 9.5)
    fig.suptitle(title, y=1.005, fontsize=12)

    fig.savefig(OUT / "figure7_dr2_corner.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  figure7_dr2_corner.png")

# ---------------------------------------------------------------------------
# Figure 8 — Differentiation: DDEM vs competing models
# ---------------------------------------------------------------------------
def figure8_differentiation():
    """Build Figure 8 from real Boltzmann output (figure8_data.npz).

    Curves: LCDM (classy), w0waCDM (classy), EDE (classy where supported,
    else emulator), S-IDE (Python emulator), FHSW (Python emulator).
    The DDEM comb is overlaid analytically from the theory module since we
    do not have a modified-CLASS run on the same grid.
    """
    from ddem_background import riemann_zeros

    data_path = OUT / "figure8_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run compute_figure8_data.py first."
        )
    d = np.load(data_path)
    k_h        = d["k_h"]
    pk_lcdm    = d["pk_lcdm"]
    pk_w0wa    = d["pk_w0wa"]
    pk_ede     = d["pk_ede"]
    pk_sIDE    = d["pk_sIDE"]
    pk_FHSW    = d["pk_FHSW"]

    res_w0wa = (pk_w0wa / pk_lcdm - 1.0) * 100
    res_ede  = (pk_ede  / pk_lcdm - 1.0) * 100
    res_sIDE = (pk_sIDE / pk_lcdm - 1.0) * 100
    res_FHSW = (pk_FHSW / pk_lcdm - 1.0) * 100

    # DDEM comb (analytic overlay, eps=0.5, alpha=0.5 fiducial)
    GAMMA_N = riemann_zeros(200)
    h_planck = 0.6774
    eta0 = 14136.17
    k_n = GAMMA_N / eta0 / h_planck
    rho_mod = np.sqrt(0.5**2 + GAMMA_N**2)
    amp = (0.5**0.5) / rho_mod
    sig = 0.02 * k_n
    gauss = np.exp(-0.5 * ((k_h[:, None] - k_n[None, :]) / sig[None, :])**2)
    res_ddem = 2 * 0.5 * np.sum(amp[None, :] * gauss, axis=1) * 100

    fig, ax = plt.subplots(figsize=SIZE_DOUBLE_COL)

    ax.plot(k_h, res_ddem, color=COLORS["primary"],  lw=1.0,
            label=r"DDEM (this work) $-$ multi-frequency comb")
    ax.plot(k_h, res_ede,  color=COLORS["secondary"], lw=1.3, ls="--",
            label="EDE (classy or emulator)")
    ax.plot(k_h, res_sIDE, color=COLORS["tertiary"],  lw=1.3, ls=":",
            label="S-IDE (Silva 2025) emulator")
    ax.plot(k_h, res_w0wa, color=COLORS["accent"],    lw=1.3, ls="-.",
            label=r"$w_0 w_a$CDM (classy CPL, DESI DR2 values)")
    ax.plot(k_h, res_FHSW, color=COLORS["highlight"], lw=1.3, ls="--",
            label="FHSW oscillating quintessence emulator")

    ax.axhline(0.0,  color="#222222", lw=0.5)
    ax.axhline(+1.0, color=COLORS["control"], lw=0.6, ls=":", alpha=0.6)
    ax.axhline(-1.0, color=COLORS["control"], lw=0.6, ls=":", alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlim(1e-2, 0.3)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"wavenumber  $k$  [$h$ Mpc$^{-1}$]")
    ax.set_ylabel(r"residual:  $P / P_{\Lambda \mathrm{CDM}} - 1$  [%]")
    ax.set_title("Differentiation: DDEM versus competing dark-sector models")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    fig.savefig(OUT / "figure8_differentiation.png")
    plt.close(fig)
    print("  figure8_differentiation.png")

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building Paper 1 figures...")
    figure1_framework()
    figure2_doom_factor()
    figure3_comb_prediction()
    figure4_pipeline()
    figure5_alpha_posteriors()
    figure6_sensitivity()
    figure7_dr2_corner()
    figure8_differentiation()
    print(f"\nAll figures in: {OUT}")
