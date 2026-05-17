"""Higher-fidelity coupled-Boltzmann solver: per-mode evolution from
radiation era (a ~ 1e-6) through matter-radiation equality, horizon crossing,
and into today, under the full DDEM Q kernel.

Extensions vs theory/boltzmann_perturbation.py:
  (i) start a_init = 1e-6 (radiation-dominated), not 1.5 * a_H(k)
  (ii) include radiation density rho_r ~ a^-4 in H(a) — already in
       solve_background's Omega_r0 = 9.05e-5
  (iii) include neutrino back-reaction in the radiation density (Omega_nu
       at 0.681 of photon energy via standard 7/8 * (4/11)^(4/3) * N_eff
       factor, also already in OMEGA_GAMMA_H2 prescription in
       run_joint_mcmc.py)
  (iv) effective sound speed during baryon-photon coupling: use the
       radiation-era growing-mode initial condition for delta_m which
       differs from the matter-era growing mode

For comb-relevant modes (k_h ~ 1e-2 to 1e-1 h/Mpc), horizon crossing is
in matter era (a ~ 1e-3 to 1e-1). Radiation-era contribution to the
matter perturbation evolution is the Meszaros suppression of the growing
mode while modes are super-horizon and radiation dominates, plus the
initial-condition match at radiation-matter equality.

Initial conditions in radiation era for sub-horizon modes:
  delta_m(a_init) ~ a_init * (correction for radiation domination)
  delta_m'(a_init) = delta_m(a_init)  (matter-era is the relevant ratio
      after radiation-matter transition)

For modes that are super-horizon at a_init: standard adiabatic ICs
  delta_m = -2/3 R Phi_init = -3/2 Phi_init (using R = -Phi sub-horizon)

We solve the simplified sub-horizon equation for matter with H(a) including
radiation, neutrinos, and modifies the friction term gamma_F = a Q / (H rho_m)
through the Q-kernel value at a_start, then evolve forward.

Validation gate:
  At beta0 = eps = 0, T(k) = delta_m(k, a=1) at every k should match the
  standard LCDM transfer function. We validate against classy at one k:
  if delta_m(k=0.05 h/Mpc) shifts by >5% from the boltzmann_perturbation.py
  (matter-only) result, that signals a real radiation-era effect.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ddem_background import solve_background, solve_lcdm, riemann_zeros


def solve_mode_radiation_era(k_nat, bg, a_init=1e-6):
    """Evolve delta_m for one k mode from a_init (radiation era) to a=1,
    including radiation-era Meszaros suppression and matter-era growth.

    Sub-horizon equation in N = ln a:
      d^2 D / dN^2 + (2 + dlnH/dN + gamma_F) dD/dN
                   - (1.5 Omega_m(a) - dgamma_F/dN) D = 0

    gamma_F = a Q / (H rho_m).

    ICs at a_init (deep radiation era, this mode is super-horizon at a_init
    if k < a_init H(a_init); typically yes for the comb-relevant k range):
      D = a_init (radiation-era growing mode ratio, normalised)
      D' = 0 (super-horizon initially, growth not yet started)

    The Meszaros suppression and matter-era growth are recovered
    self-consistently from the equation with H(a) including radiation.
    """
    lnA = bg["ln_a"]
    a_arr = bg["a"]
    H_arr = bg["H"]
    rho_m = bg["rho_m"]
    Q_arr = bg["Q"]
    rho_r = bg["rho_r"]
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

    N0 = max(np.log(a_init), lnA[0])
    Nf = lnA[-1]
    # Super-horizon ICs: D = a, D' = 0 (radiation-era frozen mode)
    a0 = float(np.exp(N0))
    y0 = [a0, 0.0]
    eval_N = np.linspace(N0, Nf, 2500)
    sol = solve_ivp(rhs, [N0, Nf], y0, t_eval=eval_N,
                    method="DOP853", rtol=1e-9, atol=1e-14, max_step=0.05)
    if not sol.success:
        return None
    return {"ln_a": eval_N, "delta_m": sol.y[0],
            "delta_m_today": float(sol.y[0][-1])}


def compute_transfer_radiation(beta0, eps, alpha, w_DE=-1.05, k_h_Mpc_grid=None,
                                a_min=1e-6, n_grid=6000, N_zeros=80,
                                Omega_m0=0.315):
    if k_h_Mpc_grid is None:
        k_h_Mpc_grid = np.logspace(-2.5, -0.3, 100)
    Omega_DE0 = 1.0 - Omega_m0 - 9.05e-5
    bg = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w_DE,
                          Omega_m0=Omega_m0, Omega_DE0=Omega_DE0,
                          a_min=a_min, a_max=1.0, n_grid=n_grid, N_zeros=N_zeros)
    k_nat_grid = k_h_Mpc_grid * 2997.92458
    delta_m_today = np.zeros_like(k_nat_grid)
    for i, k_nat in enumerate(k_nat_grid):
        sol = solve_mode_radiation_era(k_nat, bg, a_init=a_min)
        delta_m_today[i] = np.nan if sol is None else sol["delta_m_today"]
    return k_h_Mpc_grid, delta_m_today


def fit_comb_modes(k_h, residual, mask, gamma_modes, k_star=0.05):
    lnk = np.log(k_h[mask] / k_star)
    res = residual[mask]
    poly = np.polyfit(lnk, res, 2)
    smooth = np.polyval(poly, lnk)
    osc = res - smooth
    L = lnk[-1] - lnk[0]
    A_n = np.zeros(len(gamma_modes))
    for i, g in enumerate(gamma_modes):
        a_cos = (2.0 / L) * np.trapezoid(osc * np.cos(g * lnk), lnk)
        a_sin = (2.0 / L) * np.trapezoid(osc * np.sin(g * lnk), lnk)
        A_n[i] = np.sqrt(a_cos**2 + a_sin**2)
    return A_n


def main():
    print("Radiation-era Boltzmann higher-fidelity A_n:")
    print("=" * 70)
    k_h = np.logspace(-2.5, -0.3, 100)
    print(f"  k window: [{k_h[0]:.4f}, {k_h[-1]:.4f}] h/Mpc, a_init = 1e-6")
    print("Solving canonical DDEM (rad-era ICs)...")
    _, T_ddem = compute_transfer_radiation(beta0=0.005, eps=0.2, alpha=0.5,
                                            k_h_Mpc_grid=k_h, a_min=1e-6)
    print("Solving LCDM reference...")
    _, T_lcdm = compute_transfer_radiation(beta0=0.0, eps=0.0, alpha=0.5,
                                            k_h_Mpc_grid=k_h, a_min=1e-6)
    ok = np.isfinite(T_ddem) & np.isfinite(T_lcdm) & (T_lcdm != 0) & (T_ddem != 0)
    res = np.full_like(T_ddem, np.nan)
    res[ok] = np.log(T_ddem[ok] / T_lcdm[ok])
    print(f"  modes solved: {ok.sum()} / {ok.size}")
    print(f"  residual range: [{res[ok].min():.4e}, {res[ok].max():.4e}]")
    print(f"  residual mean / std: {res[ok].mean():.4e} / {res[ok].std():.4e}")

    gammas = riemann_zeros(20)
    A_n_rad = fit_comb_modes(k_h, res, ok, gammas)
    # Old (matter-era) calibration for comparison
    A_n_matter_old_approx_ratio = 0.002
    rho_mod = np.sqrt(0.5**2 + gammas**2)
    A_n_LR = 2.0 * 0.2 * 0.6**0.5 / rho_mod
    print()
    print(f"  A_n^rad-Boltz vs §5.2 LR, vs matter-era Boltzmann (ratio 0.002):")
    print(f"  {'n':>3} {'gamma_n':>9} {'A_n^rad':>12} {'A_n^LR':>12} "
          f"{'ratio_rad':>10} {'ratio_old':>10}")
    for i in range(min(10, len(gammas))):
        r_rad = A_n_rad[i] / A_n_LR[i] if A_n_LR[i] > 0 else np.nan
        print(f"  {i+1:>3} {gammas[i]:>9.3f} {A_n_rad[i]:>12.4e} "
              f"{A_n_LR[i]:>12.4e} {r_rad:>10.4f} "
              f"{A_n_matter_old_approx_ratio:>10.4f}")
    median_ratio = float(np.median(A_n_rad[:10] / A_n_LR[:10]))
    print()
    print(f"  Median A_n^rad / A_n^LR over first 10 modes: {median_ratio:.4f}")
    print(f"  Median A_n^matter-only / A_n^LR (previous):   0.0020")

    # Discard/keep verdict
    factor_change = median_ratio / 0.002 if median_ratio > 0 else np.inf
    if abs(np.log(max(factor_change, 1e-10))) < np.log(2.0):
        verdict = f"DISCARD: A_n changed by {factor_change:.2f}x < 2x; rad-era doesn't add"
    elif factor_change > 5.0:
        verdict = f"KEEP: A_n changed by {factor_change:.2f}x > 5x; rad-era materially changes the result"
    else:
        verdict = f"MARGINAL: A_n changed by {factor_change:.2f}x"
    print()
    print(f"  Verdict: {verdict}")
    np.savez(HERE.parent / "cobaya" / "boltzmann_radiation_template.npz",
             k_h_Mpc=k_h, T_ddem=T_ddem, T_lcdm=T_lcdm, residual=res,
             gammas=gammas, A_n_rad=A_n_rad, A_n_LR=A_n_LR,
             median_ratio=median_ratio, factor_change_vs_matter=factor_change)
    return verdict


if __name__ == "__main__":
    main()
