"""Validation of the linear-response approximation used for the comb signature.

The §5.2 emulator predicts the comb amplitude at the n-th Riemann mode by
the linear-response formula

    A_n = 2 eps a_eff^alpha / |rho_n| ,    rho_n = alpha + i gamma_n

This is the first-order-in-(beta0 eps) result that follows from inserting
the Q kernel into the modified linear-growth ODE and Fourier-decomposing.
The §8.2 claim "accurate to 10-20% in the observable k window" needs
quantitative justification.

Procedure:
  1. For a set of coupling amplitudes (beta0 eps) spanning the prior region
     down to beta0 eps -> 0, solve the full modified linear-growth ODE
     including the complete Q kernel.
  2. Extract the comb residual A_n(beta0, eps, n) from the modified D(a).
  3. Compare against the linear-response prediction. The ratio should
     approach 1 in the beta0 eps -> 0 limit and deviate at canonical.
  4. Report the residual as a function of beta0 eps and n.

The growth ODE in the IDE regime (per Pourtsidou et al. 2013, Pan et al.
2017 eq. 17 with our Q sign convention):

    delta_m'' + (2 + (H'/H) + a Q / (H rho_m)) delta_m'
              + ((a Q / (H rho_m))' - (3/2) Omega_m(a)) delta_m = 0

primes are derivatives w.r.t. ln a. We solve in N = ln a, ICs deep in
matter era (delta_m = a, delta_m' = a).

The comb residual is extracted by dividing D_DDEM(a)/D_LCDM(a) and looking
at the oscillatory part at frequency gamma_n in ln a.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ddem_background import solve_background, solve_lcdm, riemann_zeros


def solve_growth_full(bg):
    """Solve full modified linear-growth ODE in N = ln a.

    State y = (D, dD/dN). RHS:
        dD/dN  = D'
        d2D/dN2 = -(2 + Hprime/H + a Q / (H rho_m)) D'
                  - ((a Q / (H rho_m))' - 1.5 Omega_m(a)) D
    """
    a_arr  = bg["a"]
    H_arr  = bg["H"]
    rho_m  = bg["rho_m"]
    Q_arr  = bg["Q"]
    lnA    = bg["ln_a"]
    Om_m   = rho_m / H_arr**2

    # Pre-compute friction coefficient gamma_F(a) = a Q / (H rho_m)
    gamma_F = a_arr * Q_arr / (H_arr * rho_m)
    # Logarithmic derivative of H(a): d ln H / d ln a
    dlnH_dlna = np.gradient(np.log(H_arr), lnA)
    # d gamma_F / d ln a
    dgF_dlna = np.gradient(gamma_F, lnA)

    def rhs(N, y):
        D, Dp = y
        gF   = np.interp(N, lnA, gamma_F)
        dlnH = np.interp(N, lnA, dlnH_dlna)
        dgF  = np.interp(N, lnA, dgF_dlna)
        Om   = np.interp(N, lnA, Om_m)
        d2D  = -(2.0 + dlnH + gF) * Dp - (dgF - 1.5 * Om) * D
        return [Dp, d2D]

    N0 = lnA[0]
    Nf = lnA[-1]
    sol = solve_ivp(rhs, [N0, Nf], [np.exp(N0), np.exp(N0)],
                    t_eval=lnA, method="DOP853", rtol=1e-9, atol=1e-12)
    return sol.y[0]  # D(a)


def comb_amplitude_full_at_n(beta0, eps, alpha, w_DE=-1.05, n=1):
    """Extract the actual comb amplitude at the n-th Riemann zero by solving
    the full nonlinear coupled background + growth ODE, and dividing against
    the LCDM (beta0=0, eps=0) baseline."""
    bg_full = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w_DE,
                               a_min=1e-4, a_max=1.0, n_grid=4000, N_zeros=80)
    bg_ref  = solve_background(beta0=0.0,   eps=0.0,   alpha=alpha, w_DE=w_DE,
                               a_min=1e-4, a_max=1.0, n_grid=4000, N_zeros=80)
    D_full = solve_growth_full(bg_full)
    D_ref  = solve_growth_full(bg_ref)
    ratio  = D_full / D_ref
    a_arr  = bg_full["a"]
    lnA    = bg_full["ln_a"]
    # Comb mode: extract Fourier coefficient at frequency gamma_n in ln a.
    # Use simple cosine projection over the observable range a in [1e-2, 1]
    mask = a_arr >= 1e-2
    lnA_obs = lnA[mask]
    rsig    = (ratio[mask] - 1.0)
    gamma_n = float(riemann_zeros(n)[n - 1])
    # Cosine + sine projections
    L = lnA_obs[-1] - lnA_obs[0]
    cos_n = np.cos(gamma_n * lnA_obs)
    sin_n = np.sin(gamma_n * lnA_obs)
    a_cos = (2.0 / L) * np.trapezoid(rsig * cos_n, lnA_obs)
    a_sin = (2.0 / L) * np.trapezoid(rsig * sin_n, lnA_obs)
    A_full = np.sqrt(a_cos**2 + a_sin**2)
    return float(A_full), gamma_n


def linear_response_amplitude(beta0, eps, alpha, n=1, a_eff=0.6):
    """The §5.2 linear-response prediction. We absorb beta0 by recognising
    that the comb feature in the bracket scales as beta0 * eps. The §5.2
    formula reports the *coupling-amplitude* response A_n = 2 eps a_eff^alpha
    / |rho_n|. The full-Q amplitude scales with beta0 * eps in the small-Q
    limit, so the proper comparison fixes the coupling product beta0*eps."""
    gamma_n = float(riemann_zeros(n)[n - 1])
    rho_mod = np.sqrt(alpha**2 + gamma_n**2)
    return 2.0 * (beta0 / 0.01) * eps * a_eff**alpha / rho_mod, gamma_n


def validate_grid():
    """Scan a grid of (beta0, eps) at fixed alpha and n=1, report
    full-ODE vs linear-response amplitude ratio."""
    alpha = 0.5
    n     = 1
    rows  = []
    for beta0 in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2]:
        for eps in [0.01, 0.1, 0.5]:
            A_full, gn = comb_amplitude_full_at_n(beta0, eps, alpha, n=n)
            A_lin, _   = linear_response_amplitude(beta0, eps, alpha, n=n)
            ratio = A_full / A_lin if A_lin > 0 else np.nan
            rows.append((beta0, eps, A_full, A_lin, ratio))
            print(f"  beta0={beta0:.4g}  eps={eps:.3g}:  "
                  f"A_full={A_full:.4e}  A_lin={A_lin:.4e}  ratio={ratio:.3f}")
    return rows


def figure_validation(out_path):
    print("Linear-response approximation validation at gamma_1 = 14.135:")
    print("=" * 70)
    rows = validate_grid()
    print()

    # Plot ratio vs beta0 * eps
    beta0 = np.array([r[0] for r in rows])
    eps   = np.array([r[1] for r in rows])
    ratio = np.array([r[4] for r in rows])
    product = beta0 * eps

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for eps_val, color in zip([0.01, 0.1, 0.5], ["#1f4068", "#3c8dbc", "#e67e22"]):
        msk = np.isclose(eps, eps_val)
        ax.plot(beta0[msk], ratio[msk], marker="o", lw=1.6, color=color,
                label=fr"$\varepsilon = {eps_val}$")
    ax.axhline(1.0, color="#888888", lw=0.7, ls="--", label="linear-response prediction")
    ax.axhspan(0.8, 1.2, color="gold", alpha=0.15, label=r"$\pm 20\%$ window")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel(r"$A_{\rm full} / A_{\rm linear-response}$")
    ax.set_title(r"Linear-response accuracy at $\gamma_1 = 14.135$ ($\alpha = 0.5$)")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    return rows


if __name__ == "__main__":
    OUT = HERE.parent / "figures" / "figure10_linear_response_validation.png"
    rows = figure_validation(OUT)
    # Highlight canonical
    print()
    print("At joint MAP (beta0 ~ 0.008, eps ~ 0.07):")
    A_full, _ = comb_amplitude_full_at_n(beta0=0.008, eps=0.07, alpha=0.93, n=1)
    A_lin, _  = linear_response_amplitude(beta0=0.008, eps=0.07, alpha=0.93, n=1)
    if A_lin > 0:
        print(f"  A_full / A_lin = {A_full/A_lin:.3f}")
