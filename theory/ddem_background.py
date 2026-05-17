"""
DDEM modified background solver.

Solves the coupled matter-DE continuity equations under the proposed Q kernel:

    rho_m' + 3 H rho_m = +Q
    rho_DE' + 3 H (1 + w_DE) rho_DE = -Q

    Q(a) = beta0 * H0 * rho_DE * [1 + 2*eps * sum_n a^alpha cos(gamma_n ln a)/|rho_n|]

Convention: rho_DE evolves like a fluid with constant w_DE (phantom region per
§5 stability gate). Energy is exchanged from DE to matter when Q > 0.

This is the *background* sector. Perturbation-level modification is in
ddem_perturbation.py.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from mpmath import zetazero, mp
from functools import lru_cache

mp.dps = 30

@lru_cache(maxsize=4)
def riemann_zeros(N: int) -> np.ndarray:
    return np.array([float(zetazero(n).imag) for n in range(1, N + 1)])


def Q_bracket(a, alpha, eps, gamma_n):
    """[1 + 2*eps * sum a^alpha cos(gamma_n ln a)/|rho_n|]"""
    rho_mod = np.sqrt(alpha**2 + gamma_n**2)
    a_pow = a**alpha
    cos_term = np.cos(np.log(a) * gamma_n)
    return 1.0 + 2.0 * eps * a_pow * np.sum(cos_term / rho_mod)


def solve_background(
    beta0: float,
    eps: float,
    alpha: float,
    w_DE: float = -1.05,
    Omega_m0: float = 0.315,
    Omega_DE0: float = 0.685,
    Omega_r0: float = 9.0e-5,
    N_zeros: int = 200,
    a_min: float = 1e-6,
    a_max: float = 1.0,
    n_grid: int = 4000,
):
    """
    Forward-integrate rho_m(a) and rho_DE(a) in ln(a) from a_min to a_max.
    All densities normalized to rho_crit,0 (units where rho_crit0 = 1, H0 = 1).

    Returns a dict with arrays:
        a, ln_a, rho_m, rho_DE, rho_r, H, Q
    """
    gamma_n = riemann_zeros(N_zeros)

    # ICs at a_min (treat as standard cosmology there since Q is suppressed by a^alpha for alpha>0)
    rho_r_at = lambda a: Omega_r0 * a**-4
    # iterate to find ICs that give Omega_m0 and Omega_DE0 today: shoot backwards from today
    # Simpler closed-form: at a_min, use the unmodified scaling
    rho_m_init = Omega_m0 * a_min**-3
    rho_DE_init = Omega_DE0 * a_min**(-3 * (1 + w_DE))

    def rhs(ln_a, y):
        rho_m, rho_DE = y
        a = np.exp(ln_a)
        rho_r = rho_r_at(a)
        # avoid negative density blowup
        rho_m_safe = max(rho_m, 1e-30)
        rho_DE_safe = max(rho_DE, 1e-30)
        H = np.sqrt(rho_m_safe + rho_r + rho_DE_safe)  # H0 = 1
        bracket = Q_bracket(a, alpha, eps, gamma_n)
        Q = beta0 * 1.0 * rho_DE_safe * bracket  # H0 absorbed (= 1 here)
        drho_m_dlna = -3.0 * rho_m + Q / H
        drho_DE_dlna = -3.0 * (1.0 + w_DE) * rho_DE - Q / H
        return [drho_m_dlna, drho_DE_dlna]

    ln_grid = np.linspace(np.log(a_min), np.log(a_max), n_grid)
    sol = solve_ivp(rhs, [ln_grid[0], ln_grid[-1]], [rho_m_init, rho_DE_init],
                    t_eval=ln_grid, method="DOP853", rtol=1e-8, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Background ODE failed: {sol.message}")

    a = np.exp(sol.t)
    rho_m = sol.y[0]
    rho_DE = sol.y[1]
    rho_r = rho_r_at(a)
    H = np.sqrt(rho_m + rho_r + rho_DE)
    brackets = np.array([Q_bracket(ai, alpha, eps, gamma_n) for ai in a])
    Q = beta0 * rho_DE * brackets

    return {
        "a": a, "ln_a": sol.t,
        "rho_m": rho_m, "rho_DE": rho_DE, "rho_r": rho_r,
        "H": H, "Q": Q, "bracket": brackets,
        "Omega_m_today": rho_m[-1] / H[-1]**2,
        "Omega_DE_today": rho_DE[-1] / H[-1]**2,
        "params": dict(beta0=beta0, eps=eps, alpha=alpha, w_DE=w_DE,
                       Omega_m0=Omega_m0, Omega_DE0=Omega_DE0, N_zeros=N_zeros),
    }


def solve_lcdm(**kwargs):
    """Convenience: same solver with beta0=eps=0 -> pure LCDM."""
    kwargs.setdefault("beta0", 0.0)
    kwargs.setdefault("eps", 0.0)
    kwargs.setdefault("w_DE", -1.0)
    return solve_background(**kwargs)


if __name__ == "__main__":
    print("Sanity test: DDEM background solver")
    bg = solve_background(beta0=0.01, eps=0.5, alpha=0.5, w_DE=-1.05)
    lc = solve_lcdm(alpha=0.5)
    print(f"  DDEM today:  Om_m = {bg['Omega_m_today']:.5f}  Om_DE = {bg['Omega_DE_today']:.5f}")
    print(f"  LCDM today:  Om_m = {lc['Omega_m_today']:.5f}  Om_DE = {lc['Omega_DE_today']:.5f}")
    print(f"  H_DDEM/H_LCDM at z=0: {bg['H'][-1]/lc['H'][-1]:.6f}")
    # closure check: rho_m + rho_DE + rho_r should match H^2
    sum_today = bg['rho_m'][-1] + bg['rho_DE'][-1] + bg['rho_r'][-1]
    print(f"  Friedmann closure today: H^2 - sum = {bg['H'][-1]**2 - sum_today:.3e}")
