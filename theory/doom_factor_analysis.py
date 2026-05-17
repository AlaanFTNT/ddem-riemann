"""
Doom-factor analytical stability proof for the DDEM oscillatory Q kernel.

Hypothesis (Refined_Hypothesis.md §4 and §5):
    Q(a) = beta0 * H0 * rho_Lambda * [1 + 2*eps * sum_n a^alpha * cos(gamma_n * ln a) / |rho_n|]
    rho_n = alpha + i * gamma_n, |rho_n| = sqrt(alpha^2 + gamma_n^2)

VMM 2008 (arXiv:0804.0232) doom factor for IDE in the Q ~ rho_DE class:
    d(a) = Q(a) / [3 H(a) rho_DE(a) (1 + w_DE)]

Stability criteria from Refined_Hypothesis.md §5:
    (i)  Time-averaged d_avg(a) < 0 over the observational a-range.
         Averaging window: one full period of dominant gamma, T_a = 2*pi / gamma_dom(a).
    (ii) Peak |d_peak(a)| < N_osc(a)^-1
         where N_osc(a) = gamma_dom(a) / (2*pi) is oscillations per Hubble time.

This script computes d(a) over a in [1e-3, 1] for representative parameters
that lie inside existing IDE constraints, and reports pass/fail per criterion.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero, mp
from pathlib import Path

mp.dps = 30  # mpmath decimal precision

OUT_DIR = Path(__file__).parent
FIG_PATH = OUT_DIR / "doom_factor.png"
REPORT_PATH = OUT_DIR / "doom_factor_report.md"

# ---- Riemann zero imaginary parts ----
N_ZEROS = 200
print(f"Computing first {N_ZEROS} Riemann zeros via mpmath...")
GAMMA_N = np.array([float(zetazero(n).imag) for n in range(1, N_ZEROS + 1)])
print(f"  gamma_1 = {GAMMA_N[0]:.5f}")
print(f"  gamma_50 = {GAMMA_N[49]:.5f}")
print(f"  gamma_{N_ZEROS} = {GAMMA_N[-1]:.5f}")

# ---- Cosmology (flat LCDM background, for H(a) only) ----
H0 = 1.0  # all rates in units of H0; absolute scale cancels in d/H0 ratio
OMEGA_M = 0.315
OMEGA_R = 9.0e-5
OMEGA_L = 1.0 - OMEGA_M - OMEGA_R

def hubble(a):
    """E(a) = H(a)/H0 in flat LCDM."""
    return np.sqrt(OMEGA_M * a**-3 + OMEGA_R * a**-4 + OMEGA_L)

# ---- DDEM Q kernel ----
def Q_bracket(a, alpha, eps, N=N_ZEROS):
    """Returns [1 + 2*eps * sum_n a^alpha cos(gamma_n ln a)/|rho_n|]."""
    rho_mod = np.sqrt(alpha**2 + GAMMA_N[:N]**2)  # |rho_n|
    # vectorize over a if array
    a_arr = np.atleast_1d(a)
    ln_a = np.log(a_arr)
    # phase matrix: rows = a, cols = n
    phase = np.outer(ln_a, GAMMA_N[:N])
    cos_term = np.cos(phase)
    # amplitude factor a^alpha
    a_pow = a_arr**alpha
    # sum over zeros
    series = a_pow * np.sum(cos_term / rho_mod, axis=1)
    return 1.0 + 2.0 * eps * series

def Q_of_a(a, beta0, eps, alpha, N=N_ZEROS):
    """Q / (H0 * rho_Lambda)."""
    return beta0 * Q_bracket(a, alpha, eps, N=N)

def doom_factor(a, beta0, eps, alpha, w_DE, N=N_ZEROS):
    """
    d(a) per VMM 2008. With rho_DE(a) ~ rho_Lambda treated as approximately constant
    (Pourtsidou Type-2 background near LCDM at zeroth order in beta0):
        d = Q / [3 H rho_DE (1 + w_DE)]
        With rho_DE absorbed and Q in units of H0*rho_Lambda:
        d = beta0 * Bracket / [3 (H/H0) (1 + w_DE)]
    """
    bracket = Q_bracket(a, alpha, eps, N=N)
    E = hubble(np.atleast_1d(a))
    return beta0 * bracket / (3.0 * E * (1.0 + w_DE))

# ---- Averaging prescription (criterion i) ----
def dominant_gamma(a_center, alpha, eps, N=N_ZEROS, n_samples=2001):
    """
    Identify the gamma_n with largest contribution to the bracket at a_center.
    For the doom-factor averaging we need the dominant oscillatory mode.
    """
    rho_mod = np.sqrt(alpha**2 + GAMMA_N[:N]**2)
    a_pow = a_center**alpha
    amplitudes = a_pow / rho_mod  # |contribution| of each mode
    return GAMMA_N[np.argmax(amplitudes)]

def averaged_d(a_center, beta0, eps, alpha, w_DE, N=N_ZEROS):
    """
    Average d over one period in ln a of the dominant gamma.
    Window: ln a in [ln a_c - pi/gamma_dom, ln a_c + pi/gamma_dom].
    """
    gamma_dom = dominant_gamma(a_center, alpha, eps, N=N)
    window = np.pi / gamma_dom  # half-period in ln a
    ln_grid = np.linspace(np.log(a_center) - window, np.log(a_center) + window, 1001)
    a_grid = np.exp(ln_grid)
    d_vals = doom_factor(a_grid, beta0, eps, alpha, w_DE, N=N)
    return np.mean(d_vals), gamma_dom

# ---- Peak amplitude test (criterion ii) ----
def peak_d(a_center, beta0, eps, alpha, w_DE, N=N_ZEROS, n_samples=10001):
    """
    Find |d_peak| within a narrow neighbourhood of a_center.
    Window: half-decade in a around a_center (broad enough to capture extrema).
    """
    ln_grid = np.linspace(np.log(a_center) - 0.25, np.log(a_center) + 0.25, n_samples)
    a_grid = np.exp(ln_grid)
    d_vals = doom_factor(a_grid, beta0, eps, alpha, w_DE, N=N)
    return np.max(np.abs(d_vals))

def N_osc_threshold(a_center, alpha, eps, N=N_ZEROS):
    """
    Threshold |d_peak| < N_osc^-1.
    N_osc = gamma_dom / (2*pi).
    """
    gamma_dom = dominant_gamma(a_center, alpha, eps, N=N)
    N_osc = gamma_dom / (2.0 * np.pi)
    return 1.0 / N_osc, gamma_dom, N_osc

# ---- Parameter scan ----
PARAM_SETS = [
    # (name, beta0, eps, alpha, w_DE)
    ("baseline",          0.01,  0.5,  0.5,  -1.05),
    ("higher_eps",        0.01,  1.0,  0.5,  -1.05),
    ("higher_beta",       0.05,  0.5,  0.5,  -1.05),
    ("alpha_zero",        0.01,  0.5,  0.0,  -1.05),
    ("alpha_one",         0.01,  0.5,  1.0,  -1.05),
    ("weakly_phantom",    0.01,  0.5,  0.5,  -1.01),
    ("strongly_phantom",  0.01,  0.5,  0.5,  -1.15),
    ("nonphantom_FAIL",   0.01,  0.5,  0.5,  -0.95),  # expected fail (d > 0)
]

A_TEST = np.array([1e-3, 1e-2, 1e-1, 0.5, 1.0])

results = []
for name, beta0, eps, alpha, w_DE in PARAM_SETS:
    row = {"name": name, "beta0": beta0, "eps": eps, "alpha": alpha, "w_DE": w_DE,
           "checks": []}
    for a_c in A_TEST:
        d_avg, gamma_dom = averaged_d(a_c, beta0, eps, alpha, w_DE)
        d_pk = peak_d(a_c, beta0, eps, alpha, w_DE)
        thresh, gd2, n_osc = N_osc_threshold(a_c, alpha, eps)
        pass_i = d_avg < 0
        pass_ii = d_pk < thresh
        row["checks"].append({
            "a": a_c, "gamma_dom": gamma_dom, "N_osc": n_osc,
            "d_avg": d_avg, "d_peak": d_pk, "threshold": thresh,
            "pass_i": pass_i, "pass_ii": pass_ii,
        })
    results.append(row)

# ---- Print report ----
def fmt_check(c):
    p1 = "OK" if c["pass_i"]  else "FAIL"
    p2 = "OK" if c["pass_ii"] else "FAIL"
    return (f"  a={c['a']:.0e}  gamma_dom={c['gamma_dom']:7.2f}  N_osc={c['N_osc']:6.2f}  "
            f"d_avg={c['d_avg']:+.3e}  d_peak={c['d_peak']:.3e}  "
            f"thr={c['threshold']:.3e}  [i:{p1}] [ii:{p2}]")

print("\n" + "=" * 100)
print("DOOM-FACTOR STABILITY ANALYSIS")
print("=" * 100)
for r in results:
    print(f"\n[{r['name']}] beta0={r['beta0']} eps={r['eps']} alpha={r['alpha']} w_DE={r['w_DE']}")
    for c in r["checks"]:
        print(fmt_check(c))

# ---- Plot d(a) for the baseline case and one failing case ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
a_plot = np.logspace(-3, 0, 4000)

for ax, (name, beta0, eps, alpha, w_DE) in zip(axes.ravel(), PARAM_SETS[:4]):
    d_vals = doom_factor(a_plot, beta0, eps, alpha, w_DE)
    ax.plot(a_plot, d_vals, lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("scale factor a")
    ax.set_ylabel("doom factor d(a)")
    ax.set_title(f"{name}: beta0={beta0} eps={eps} alpha={alpha} w_DE={w_DE}")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=130)
print(f"\nFigure saved: {FIG_PATH}")

# ---- Write markdown report ----
def md_check(c):
    p1 = "✓" if c["pass_i"]  else "✗"
    p2 = "✓" if c["pass_ii"] else "✗"
    return (f"| {c['a']:.0e} | {c['gamma_dom']:.2f} | {c['N_osc']:.2f} | "
            f"{c['d_avg']:+.3e} | {c['d_peak']:.3e} | {c['threshold']:.3e} | {p1} | {p2} |")

lines = [
    "# Doom-Factor Analytical Stability — Results",
    "",
    "Computed per `theory/doom_factor_analysis.py`. Hypothesis tested:",
    "`Q(a) = beta0 * H0 * rho_Lambda * [1 + 2*eps * sum_n a^alpha cos(gamma_n ln a)/|rho_n|]`",
    "",
    f"First {N_ZEROS} Riemann zeros (mpmath at {mp.dps}-digit precision).",
    "Cosmology: flat LCDM, Omega_m=0.315, Omega_r=9e-5.",
    "",
    "Criteria (Refined_Hypothesis.md §5):",
    "- (i)  d_avg(a) < 0 averaged over one period of dominant gamma.",
    "- (ii) |d_peak(a)| < N_osc(a)^-1, N_osc = gamma_dom / (2*pi).",
    "",
]

for r in results:
    lines.append(f"## {r['name']}  (beta0={r['beta0']}, eps={r['eps']}, alpha={r['alpha']}, w_DE={r['w_DE']})")
    lines.append("")
    lines.append("| a | gamma_dom | N_osc | d_avg | d_peak | threshold | (i) | (ii) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in r["checks"]:
        lines.append(md_check(c))
    n_pass_i  = sum(c["pass_i"]  for c in r["checks"])
    n_pass_ii = sum(c["pass_ii"] for c in r["checks"])
    lines.append("")
    lines.append(f"Summary: (i) pass {n_pass_i}/{len(A_TEST)}, (ii) pass {n_pass_ii}/{len(A_TEST)}.")
    lines.append("")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Markdown report: {REPORT_PATH}")
