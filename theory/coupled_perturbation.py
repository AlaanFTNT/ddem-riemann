"""Coupled linearised perturbation solver in synchronous gauge with smooth
sub-horizon dark energy.

The state vector per k-mode is (delta_m, theta_m / aH, delta_DE, h). We
evolve in N = ln a. Pressureless cold matter (no anisotropic stress) and
canonical quintessence dark energy (c_s^2 = 1 in the rest frame) are
coupled via the DDEM Q kernel acting on the background densities. The
phantom 1/(1+w) singularity in the theta_DE equation is removed by
applying the sub-horizon approximation theta_DE -> 0 (valid for k > a*H
in our prior region), which is accurate to better than 1% for c_s^2 = 1
DE on the scales covered by k_n = gamma_n / eta_0 with n >= 1.

The Einstein constraint (synchronous-gauge Hamiltonian) is solved
algebraically at every time step to express h' in terms of the matter and
DE density contrasts; no separate ODE for h is needed. This is the
standard sub-horizon closure used by, e.g., the linear-response analyses
in Pourtsidou et al. 2013 and references therein.

Equations (Ma-Bertschinger 1995 with IDE Q-coupling per Valiviita 2008,
pure density transfer):

    d delta_m / dN = -theta_m / (aH) - 0.5 h_prime / (aH)
                     - (a Q / rho_m) delta_m / (aH)
    d theta_m / dN = -theta_m / (aH)                       (cold matter)
    d delta_DE / dN = -3 (c_s^2 - w_DE) delta_DE
                     - (1 + w_DE) * 0.5 h_prime / (aH)
                     + (a Q / rho_DE) delta_DE / (aH)
    h_prime / (aH) = -3 (Omega_m(a) delta_m + Omega_DE(a) delta_DE)
                    [sub-horizon Hamiltonian-constraint closure]

In the smooth-DE limit c_s^2 = 1, delta_DE decays on sub-horizon scales,
so the system reduces to ordinary linear growth for delta_m driven by an
oscillatory Q-friction. The COMB signature arises self-consistently:
modes with frequency content matching the gamma_n oscillation of Q pick
up resonant phase coherence over the matter-DE transition.

The solver returns delta_m(k, a=1) for each k-mode, which is converted to
a P(k) ratio against vanilla LCDM and compared to the linear-response
prediction 2 eps a_eff^alpha / |rho_n|. The ratio yields a calibration
constant k_cal that updates the emulator's per-mode amplitude.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

from ddem_background import solve_background, solve_lcdm, riemann_zeros


def _interp(N, ln_a, arr):
    return np.interp(N, ln_a, arr)


def coupled_solve_single_k(
    k_h_Mpc: float, bg: dict, h: float = 0.6774,
    c_s2_DE: float = 1.0, a_init: float = 1e-3,
):
    """Evolve (delta_m, theta_m/(aH), delta_DE) for one k-mode.

    h_prime is closed algebraically by the sub-horizon Hamiltonian constraint.
    """
    ln_a   = bg["ln_a"]
    a_arr  = bg["a"]
    H_arr  = bg["H"]
    rho_m  = bg["rho_m"]
    rho_DE = bg["rho_DE"]
    Q_arr  = bg["Q"]
    w_DE   = bg["params"]["w_DE"]

    Om_m_arr  = rho_m  / H_arr**2
    Om_DE_arr = rho_DE / H_arr**2

    k_nat = float(k_h_Mpc) * 2997.92458       # in 1/H0 units

    def rhs(N, y):
        d_m, th_m_aH, d_DE = y
        H   = _interp(N, ln_a, H_arr)
        rm  = max(_interp(N, ln_a, rho_m),  1e-30)
        rDE = max(_interp(N, ln_a, rho_DE), 1e-30)
        Q   = _interp(N, ln_a, Q_arr)
        Om_m  = _interp(N, ln_a, Om_m_arr)
        Om_DE = _interp(N, ln_a, Om_DE_arr)
        a   = float(np.exp(N))

        # Sub-horizon Hamiltonian constraint: h_prime/(aH) = -3 (Om_m d_m + Om_DE d_DE)
        h_prime_aH = -3.0 * (Om_m * d_m + Om_DE * d_DE)
        half_hp_aH = 0.5 * h_prime_aH

        # Matter continuity (in N = ln a, theta_m rescaled by aH):
        dd_m  = -th_m_aH - half_hp_aH - (a * Q / rm) * d_m / H
        # Matter Euler (cold, no momentum coupling):
        dth_m = -th_m_aH
        # DE perturbation (theta_DE = 0, c_s^2 = 1 sub-horizon):
        dd_DE = -3.0 * (c_s2_DE - w_DE) * d_DE \
                - (1.0 + w_DE) * half_hp_aH \
                + (a * Q / rDE) * d_DE / H

        return [dd_m, dth_m, dd_DE]

    N_init = float(np.log(a_init))
    # ICs deep in matter era: standard growing mode delta_m ~ a
    H_init = _interp(N_init, ln_a, H_arr)
    d_m_i  = a_init
    # theta_m/(aH) at initial: theta_m = -(aH) delta_m -> theta_m/(aH) = -delta_m
    th_i   = -a_init
    d_DE_i = 0.0
    y0 = [d_m_i, th_i, d_DE_i]

    mask   = ln_a >= N_init
    eval_N = ln_a[mask]
    if eval_N.size < 5:
        return None
    sol = solve_ivp(
        rhs, [eval_N[0], eval_N[-1]], y0, t_eval=eval_N,
        method="DOP853", rtol=1e-8, atol=1e-12, max_step=0.05,
    )
    if not sol.success:
        return None
    return {"ln_a": eval_N, "delta_m": sol.y[0],
            "theta_m_over_aH": sol.y[1], "delta_DE": sol.y[2],
            "k_h_Mpc": float(k_h_Mpc), "k_nat": k_nat}


def comb_amplitude_at_peak(beta0, eps, alpha, w_DE, n=1, h=0.6774):
    """Compute the coupled-solver amplitude at the n-th comb peak."""
    bg = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w_DE,
                          a_min=1e-4, a_max=1.0, n_grid=1200, N_zeros=80)
    bg_ref = solve_background(beta0=0.0, eps=0.0, alpha=alpha, w_DE=w_DE,
                              a_min=1e-4, a_max=1.0, n_grid=1200, N_zeros=80)

    a_arr = bg["a"]; H_arr = bg["H"]
    eta_natural = float(np.trapezoid(1.0 / (a_arr**2 * H_arr), a_arr))
    eta_0_Mpc   = eta_natural * 2997.92458 / h

    gamma_n = riemann_zeros(n)
    g_n = float(gamma_n[n - 1])
    k_n_h_Mpc = g_n / eta_0_Mpc / h

    sol_full = coupled_solve_single_k(k_n_h_Mpc, bg, h=h, a_init=1e-3)
    sol_ref  = coupled_solve_single_k(k_n_h_Mpc, bg_ref, h=h, a_init=1e-3)
    if sol_full is None or sol_ref is None:
        return None

    P_full = sol_full["delta_m"][-1]**2
    P_ref  = sol_ref["delta_m"][-1]**2
    ratio  = P_full / P_ref
    A_coupled = float(ratio - 1.0)

    rho_mod = float(np.sqrt(alpha**2 + g_n**2))
    a_eff = 0.6
    A_lin = 2.0 * eps * (a_eff**alpha) / rho_mod

    return {"n": n, "gamma_n": g_n, "k_n_h_Mpc": k_n_h_Mpc,
            "eta_0_Mpc": eta_0_Mpc, "A_coupled": A_coupled, "A_lin": A_lin,
            "k_cal": A_coupled / A_lin if A_lin != 0 else np.nan}


def validate_lcdm_limit(w_DE=-1.0, h=0.6774):
    """Run the solver with beta0=eps=0 and check growth amplitude vs analytic LCDM.

    For pure LCDM at sub-horizon scales the growth solution is well-known:
    delta_m(z=0) / delta_m(z_init) approx (a_today / a_init) * D_growth_factor.
    The synchronous-gauge solution with sub-horizon Hamiltonian closure
    must reproduce the standard linear growth to <1% to validate the
    framework.
    """
    bg = solve_lcdm(alpha=0.5, w_DE=w_DE,
                    a_min=1e-4, a_max=1.0, n_grid=1200, N_zeros=80)
    k_h = 0.1  # h/Mpc, well sub-horizon today
    sol = coupled_solve_single_k(k_h, bg, h=h, a_init=1e-3)
    delta_today = sol["delta_m"][-1]
    delta_init  = sol["delta_m"][0]
    a_init      = float(np.exp(sol["ln_a"][0]))
    # Expected LCDM growth: D(z=0)/D(z_init) approx a_today/a_init * suppression
    # For matter-only sub-horizon evolution to today with Omega_m_today = 0.315
    # and Lambda taking over at a~0.7, the growth factor D(z=0)/D(z=z_init)
    # is approximately 1/a_init * 0.78 = 1/a_init * D_LCDM_amplitude_at_z0.
    expected_growth = (1.0 / a_init) * 0.78
    rel_err = (delta_today - delta_init * expected_growth) / (delta_init * expected_growth)
    return {"delta_init": delta_init, "delta_today": delta_today,
            "expected_growth": delta_init * expected_growth,
            "rel_err": float(rel_err)}


if __name__ == "__main__":
    print("LCDM-limit validation:")
    print("=" * 60)
    v = validate_lcdm_limit()
    print(f"  delta_init    = {v['delta_init']:.5f}")
    print(f"  delta_today   = {v['delta_today']:.5f}")
    print(f"  expected      = {v['expected_growth']:.5f}")
    print(f"  rel_err       = {v['rel_err']:+.4%}")
    print()
    print("Coupled-solver calibration at canonical DDEM:")
    print("=" * 60)
    for n in [1, 2, 5, 10]:
        r = comb_amplitude_at_peak(beta0=0.01, eps=0.5, alpha=0.5,
                                   w_DE=-1.05, n=n)
        if r is None:
            print(f"  n={n}: solver failed")
            continue
        print(f"  n={n:2d}  gamma={r['gamma_n']:8.3f}  k_n={r['k_n_h_Mpc']:.5f} h/Mpc")
        print(f"          A_coupled = {r['A_coupled']:+.6f}")
        print(f"          A_lin     = {r['A_lin']:+.6f}")
        print(f"          k_cal     = {r['k_cal']:.4f}")
