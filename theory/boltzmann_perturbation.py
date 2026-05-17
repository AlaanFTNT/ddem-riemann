"""Coupled-Boltzmann perturbation solver for the DDEM matter + dark-energy
system. Per-mode evolution from just after horizon crossing through to today,
under the full DDEM Q kernel and PPF closure for sub-horizon DE.

The physical question: when each Fourier mode k crosses the cosmological
horizon at a_H(k) defined by k = a_H H(a_H), does the oscillatory Q kernel
imprint a phase-coherent feature at k_n = gamma_n / eta_0 in the matter
transfer function T(k)? The §5.2 linear-response formula estimates this
analytically; the §8.2 PPF result captures only the late-time growth
contribution, which is k-independent. Only a per-mode Boltzmann evolution
with the Q-kernel imprint at horizon crossing settles the amplitude.

Strategy:
  - For each k in the comb-relevant window, identify a_H(k) by solving
    k = a H(a) on the background grid. Start evolution at a_start(k) =
    1.5 a_H(k), so the mode is comfortably sub-horizon at integration start.
  - Initial conditions in matter era: standard growing mode delta_m proportional
    to a, with Q-induced phase imprint set by the value of cos(gamma_n ln a_H(k))
    at the time of horizon crossing through the integrated Q-source between
    a_start(k) and a_init.
  - Evolve sub-horizon matter equation with Q-induced friction (gamma_F =
    a Q / (H rho_m)) included. DE perturbation is PPF-smooth (delta_DE = 0).
  - At a = 1, read off delta_m(k). T(k) is delta_m(k, 1) divided by an
    appropriate normalisation (cancels in T_DDEM/T_LCDM ratio).
  - Repeat for the LCDM (beta0=eps=0) reference background to get the ratio.
  - Fit the Fourier coefficients of the residual ln(T_DDEM/T_LCDM) at
    gamma_n positions in ln k to obtain A_n^Boltz.

The sub-horizon matter equation in N = ln a:

  d2 delta_m / dN^2 + (2 + dlnH/dlnN + gamma_F(a)) d delta_m / dN
                    - (1.5 Omega_m(a) - d gamma_F / dN) delta_m = 0

with gamma_F(a) = a Q(a) / (H rho_m), exactly as in the §8.2 PPF solver but
now with k-dependent initial conditions at horizon crossing.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ddem_background import solve_background, solve_lcdm, riemann_zeros


def find_horizon_crossing(k_nat, bg):
    """Find a_H(k) such that k = a H(a) in natural (H_0 = 1) units.
    Returns the scale factor at horizon crossing, or None if the mode is
    super-horizon at a=1 (i.e., k <= H_0)."""
    a   = bg["a"]
    H   = bg["H"]
    aH  = a * H
    if k_nat <= aH[-1]:
        return None
    if k_nat >= aH[0]:
        return float(a[0])      # already sub-horizon at a_init
    return float(np.interp(k_nat, aH, a))


def solve_mode_subhorizon(k_nat, bg, a_start_factor=1.5):
    """Evolve delta_m for one k mode from a_start = a_start_factor * a_H(k)
    to a=1. Sub-horizon matter equation with Q-induced friction."""
    a_H = find_horizon_crossing(k_nat, bg)
    if a_H is None:
        return None
    a_start = min(0.9, a_start_factor * a_H)
    if a_start <= bg["a"][0]:
        return None

    lnA   = bg["ln_a"]
    a_arr = bg["a"]
    H_arr = bg["H"]
    rho_m = bg["rho_m"]
    Q_arr = bg["Q"]
    Om_m  = rho_m / H_arr**2
    gamma_F = a_arr * Q_arr / (H_arr * rho_m)
    dlnH_dlna = np.gradient(np.log(H_arr), lnA)
    dgF_dlna  = np.gradient(gamma_F, lnA)

    def rhs(N, y):
        D, Dp = y
        dlnH = np.interp(N, lnA, dlnH_dlna)
        gF   = np.interp(N, lnA, gamma_F)
        dgF  = np.interp(N, lnA, dgF_dlna)
        Om   = np.interp(N, lnA, Om_m)
        d2D  = -(2.0 + dlnH + gF) * Dp - (dgF - 1.5 * Om) * D
        return [Dp, d2D]

    # Matter-era ICs at a_start: standard growing mode delta_m ~ a, delta_m' ~ a
    N0 = np.log(a_start)
    Nf = lnA[-1]
    y0 = [a_start, a_start]
    eval_N = np.linspace(N0, Nf, 1500)
    sol = solve_ivp(rhs, [N0, Nf], y0, t_eval=eval_N,
                    method="DOP853", rtol=1e-9, atol=1e-14, max_step=0.05)
    if not sol.success:
        return None
    return {"ln_a": eval_N, "delta_m": sol.y[0],
            "a_H": a_H, "a_start": a_start,
            "delta_m_today": float(sol.y[0][-1])}


def compute_transfer(beta0, eps, alpha, w_DE=-1.05, k_h_Mpc_grid=None,
                     a_min=1e-4, n_grid=4000, N_zeros=80, Omega_m0=0.315):
    """Compute delta_m(k, a=1) on a k-grid via per-mode sub-horizon Boltzmann
    evolution. The transfer function T(k) is proportional to delta_m(k, 1)
    up to a normalisation; for our ratio T_DDEM/T_LCDM that normalisation
    cancels."""
    if k_h_Mpc_grid is None:
        k_h_Mpc_grid = np.logspace(-2.5, -0.3, 100)
    Omega_DE0 = 1.0 - Omega_m0 - 9.05e-5
    bg = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w_DE,
                          Omega_m0=Omega_m0, Omega_DE0=Omega_DE0,
                          a_min=a_min, a_max=1.0, n_grid=n_grid, N_zeros=N_zeros)
    # k in natural units: k_nat = k_h_Mpc * c/H_0 / h = k_h_Mpc * 2997.92458
    k_nat_grid = k_h_Mpc_grid * 2997.92458
    delta_m_today = np.zeros_like(k_nat_grid)
    a_H_grid      = np.zeros_like(k_nat_grid)
    for i, k_nat in enumerate(k_nat_grid):
        sol = solve_mode_subhorizon(k_nat, bg)
        if sol is None:
            delta_m_today[i] = np.nan
            a_H_grid[i] = np.nan
        else:
            delta_m_today[i] = sol["delta_m_today"]
            a_H_grid[i] = sol["a_H"]
    return k_h_Mpc_grid, delta_m_today, a_H_grid


def transfer_to_residual(k_h, T_ddem, T_lcdm):
    """ln[T_DDEM / T_LCDM] residual, finite-mask applied."""
    ok = np.isfinite(T_ddem) & np.isfinite(T_lcdm) & (T_lcdm != 0) & (T_ddem != 0)
    res = np.full_like(T_ddem, np.nan)
    res[ok] = np.log(T_ddem[ok] / T_lcdm[ok])
    return res, ok


def fit_comb_modes(k_h, residual, mask, gamma_modes, k_star=0.05):
    """Project the (finite, masked) residual onto cos/sin (gamma_n ln k/k_star)."""
    lnk = np.log(k_h[mask] / k_star)
    res = residual[mask]
    # detrend the residual: subtract linear-in-ln(k) fit so we extract oscillatory part only
    poly = np.polyfit(lnk, res, 2)
    smooth = np.polyval(poly, lnk)
    osc = res - smooth
    L = lnk[-1] - lnk[0]
    A_n   = np.zeros(len(gamma_modes))
    for i, g in enumerate(gamma_modes):
        a_cos = (2.0 / L) * np.trapezoid(osc * np.cos(g * lnk), lnk)
        a_sin = (2.0 / L) * np.trapezoid(osc * np.sin(g * lnk), lnk)
        A_n[i] = np.sqrt(a_cos**2 + a_sin**2)
    return A_n, lnk, osc


def linear_response_amplitude(eps, alpha, gamma_modes, a_eff=0.6):
    rho_mod = np.sqrt(alpha**2 + gamma_modes**2)
    return 2.0 * eps * a_eff**alpha / rho_mod


def main():
    print("DDEM coupled-Boltzmann horizon-crossing transfer function:")
    print("=" * 70)
    k_h = np.logspace(-2.5, -0.3, 100)
    print(f"  k window: [{k_h[0]:.4f}, {k_h[-1]:.4f}] h/Mpc, {k_h.size} modes")
    print("Solving canonical DDEM (beta0=0.005, eps=0.2, alpha=0.5)...")
    _, T_ddem, aH_ddem = compute_transfer(beta0=0.005, eps=0.2, alpha=0.5,
                                          k_h_Mpc_grid=k_h)
    print("Solving LCDM reference (beta0=eps=0)...")
    _, T_lcdm, aH_lcdm = compute_transfer(beta0=0.0, eps=0.0, alpha=0.5,
                                          k_h_Mpc_grid=k_h)
    residual, ok = transfer_to_residual(k_h, T_ddem, T_lcdm)
    print(f"  modes solved (DDEM): {np.isfinite(T_ddem).sum()} / {T_ddem.size}")
    print(f"  modes solved (LCDM): {np.isfinite(T_lcdm).sum()} / {T_lcdm.size}")
    print(f"  residual range over valid k: [{residual[ok].min():.4e}, {residual[ok].max():.4e}]")
    print(f"  residual mean / std: {residual[ok].mean():.4e} / {residual[ok].std():.4e}")
    print()

    gamma_modes = riemann_zeros(50)
    A_n_B, lnk_obs, osc = fit_comb_modes(k_h, residual, ok, gamma_modes)
    A_n_LR = linear_response_amplitude(eps=0.2, alpha=0.5, gamma_modes=gamma_modes)

    print("Boltzmann-derived A_n^Boltz vs §5.2 linear-response (n=1..15):")
    print(f"  {'n':>3} {'gamma_n':>9} {'A_n^Boltz':>12} {'A_n^LR':>12} {'ratio':>8}")
    for i in range(min(15, len(gamma_modes))):
        ratio = A_n_B[i] / A_n_LR[i] if A_n_LR[i] > 0 else np.nan
        print(f"  {i+1:>3} {gamma_modes[i]:>9.3f} {A_n_B[i]:>12.4e} "
              f"{A_n_LR[i]:>12.4e} {ratio:>8.3f}")
    print()
    print(f"Median ratio A_n^Boltz / A_n^LR over first 15 modes: "
          f"{np.median(A_n_B[:15] / A_n_LR[:15]):.3f}")
    print()
    np.savez(HERE.parent / "cobaya" / "boltzmann_comb_template.npz",
             k_h_Mpc=k_h, T_ddem=T_ddem, T_lcdm=T_lcdm,
             residual=residual, mask=ok, gamma_modes=gamma_modes,
             A_n_Boltz=A_n_B, A_n_LR=A_n_LR,
             params={"beta0": 0.005, "eps": 0.2, "alpha": 0.5, "w_DE": -1.05})
    print(f"  template saved: cobaya/boltzmann_comb_template.npz")
    return A_n_B, A_n_LR, gamma_modes


if __name__ == "__main__":
    main()
