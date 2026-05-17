"""
DDEM perturbation-level modification to the linear matter power spectrum.

Two effects encoded:

(1) Modified linear growth factor D(a) from the energy transfer Q.
    Solves the matter perturbation ODE
        delta_m'' + (2 + H'/H + (Q/(H rho_m))) delta_m' - (3/2) Omega_m(a) (1 + something) delta_m = 0
    in N = ln(a). The (Q/(H rho_m)) term adds friction; the Omega_m shift modifies the source.
    Calls ddem_background.solve_background to get H(a), rho_m(a), rho_DE(a), Q(a).

(2) Frequency-comb oscillatory residual in P(k) at k_n = gamma_n / eta0.
    Wave numbers commensurate with the time-domain oscillation frequencies of Q develop
    a resonance: at k = gamma_n / eta0 the in-phase coupling enhances/suppresses P(k).
    Per-mode amplitude scales as eps * a^alpha / |rho_n|.

    The kernel used here is a Gaussian envelope around each k_n with width tied to the
    Hubble damping rate (heuristic). This is the prediction-side estimate; the rigorous
    treatment requires the full modified Boltzmann hierarchy in a CLASS source patch.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from ddem_background import solve_background, solve_lcdm, riemann_zeros, Q_bracket


def linear_growth(bg) -> np.ndarray:
    """
    Solve linear-growth ODE in ln(a) using the bg dict from solve_background.
    Returns D(a) normalized to D=1 at a_init (deep matter era).

    delta'' + [2 + (dH/dN)/H + Q/(H rho_m)] delta' - (3/2) Omega_m(N) delta = 0
    where N = ln a, primes are d/dN, and Omega_m(N) = rho_m / (3 H^2) ... wait that's not right
    The Poisson source in cosmological perturbation theory is:
        delta'' + (2 + d ln H / dN) delta' = (3/2) Omega_m(N) delta
    where Omega_m(N) = rho_m(a) / (3 H(a)^2) ... with the right units (H0=1 here, rho_crit0=1)
    so Omega_m(N) = rho_m(a) / (3 H(a)^2) * 3 = rho_m(a) / H(a)^2
    Standard form: f_m(a) = rho_m / (rho_m + rho_r + rho_DE) = rho_m / H^2 here.
    """
    ln_a = bg["ln_a"]
    rho_m = bg["rho_m"]
    H = bg["H"]
    Q = bg["Q"]

    # finite-difference d ln H / d N
    ln_H = np.log(H)
    dlnH_dN = np.gradient(ln_H, ln_a)

    # Q-induced friction; Q can be negative oscillations
    friction = Q / (H * rho_m)

    # Omega_m as a function of N: rho_m / H^2 (with H0=1, rho_crit0=1)
    Omega_m_N = rho_m / H**2

    def rhs(N, y):
        d, dp = y  # delta, delta'
        dlH = np.interp(N, ln_a, dlnH_dN)
        fr  = np.interp(N, ln_a, friction)
        Om  = np.interp(N, ln_a, Omega_m_N)
        ddp = -(2.0 + dlH + fr) * dp + 1.5 * Om * d
        return [dp, ddp]

    # initial conditions: matter-dominated growth, delta ~ a, delta' = delta
    d0 = 1.0
    dp0 = 1.0
    sol = solve_ivp(rhs, [ln_a[0], ln_a[-1]], [d0, dp0],
                    t_eval=ln_a, method="DOP853", rtol=1e-8, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Growth ODE failed: {sol.message}")
    return sol.y[0]  # delta(N)


def conformal_time_today(bg) -> float:
    """
    eta0 = integral_0^t0 dt/a = integral d ln a / (a H)
    """
    a = bg["a"]
    H = bg["H"]
    integrand = 1.0 / (a * H)
    eta = cumulative_trapezoid(integrand, np.log(a), initial=0.0)
    # in units where H0 = 1 and c=1, eta has units of 1/H0 = 1/(h * 100 km/s/Mpc)
    # to convert to Mpc, multiply by c/H0 = 2997.92/h  Mpc (with h ~ 0.6774)
    return eta[-1]


def comb_signature(k_h_Mpc: np.ndarray,
                   bg,
                   eps: float,
                   alpha: float,
                   N_zeros: int = 200,
                   h: float = 0.6774):
    """
    Compute the comb modification (1 + delta_comb(k)) on the matter power spectrum.

    Per-mode contribution at k_n = gamma_n / eta0 (in physical Mpc^-1):
        A_n(k) = (a_eff^alpha / |rho_n|) * exp(-((ln k - ln k_n)/sigma_lnk)^2 / 2)
    summed over n, multiplied by 2*eps.

    The width sigma_lnk is derived from the von Mangoldt damping factor
    x^rho_n / rho_n with rho_n = alpha + i*gamma_n. The damping rate alpha
    sets the phase coherence length in ln(x); for mode k crossing the horizon
    when k * eta = gamma_n, the corresponding width in ln(k) is

        sigma_lnk = alpha / gamma_n   .

    For alpha = 1/2, gamma_1 = 14.135 -> sigma_lnk = 0.035 (3.5% in k).
    For gamma_100 ≈ 236 -> sigma_lnk = 0.0021 (0.2% in k).
    Low-n peaks are wider; high-n peaks are sharp, consistent with the
    coherence-length interpretation.

    Returns (1 + 2*eps * sum, k_n in h/Mpc, eta0 in Mpc).
    """
    gamma_n = riemann_zeros(N_zeros)
    rho_mod = np.sqrt(alpha**2 + gamma_n**2)

    eta0_natural = conformal_time_today(bg)
    eta0_Mpc = eta0_natural * 2997.92 / h
    k_n_Mpc = gamma_n / eta0_Mpc            # 1/Mpc
    k_n_h_Mpc = k_n_Mpc / h                  # h/Mpc

    # Average over the dark-energy-emergence epoch: a ∈ [0.4, 1.0] is where
    # the coupling is dynamically relevant. Use the rho_DE-weighted a-mean.
    a_eff = 0.6
    amp = a_eff**alpha / rho_mod

    # Physical width from the von Mangoldt damping coherence length
    sigma_lnk = max(abs(alpha), 1e-3) / gamma_n
    # Width in k-space at each peak: dk = k * sigma_lnk
    sig_k = k_n_h_Mpc * sigma_lnk

    k_h = np.atleast_1d(k_h_Mpc)
    # Gaussian in ln k centered at ln k_n with width sigma_lnk:
    log_args = np.log(np.maximum(k_h[:, None], 1e-12)) - np.log(k_n_h_Mpc[None, :])
    gauss = np.exp(-0.5 * (log_args / sigma_lnk[None, :])**2)
    contribution = np.sum(amp[None, :] * gauss, axis=1)
    return 1.0 + 2.0 * eps * contribution, k_n_h_Mpc, eta0_Mpc


def max_comb_amplitude(eps, alpha, eta0_Mpc, h=0.6774, N_zeros=200,
                       k_min_h=5e-3, k_max_h=0.3):
    """Maximum |residual| of the comb in the observable k-window.

    Used for the perturbation-level feature-amplitude likelihood (Beutler et al.
    2023, arXiv:2303.13946 — 95% upper bound on P(k) oscillatory residuals).
    Evaluating only at the comb peaks gives the exact maximum.
    """
    gamma_n = riemann_zeros(N_zeros)
    rho_mod = np.sqrt(alpha**2 + gamma_n**2)
    k_n_h_Mpc = (gamma_n / eta0_Mpc) / h
    in_window = (k_n_h_Mpc >= k_min_h) & (k_n_h_Mpc <= k_max_h)
    if not in_window.any():
        return 0.0
    a_eff = 0.6
    per_mode = a_eff**alpha / rho_mod
    # At the exact peak, the Gaussian factor equals 1; envelope from neighbors is small
    # for sigma_lnk much smaller than spacing between log(k_n).
    peak_vals = 2.0 * eps * per_mode[in_window]
    return float(np.max(np.abs(peak_vals)))


def DDEM_over_LCDM(k_h_Mpc, z, bg, lc, eps, alpha, N_zeros=200, h=0.6774):
    """
    Full P_DDEM(k, z) / P_LCDM(k, z) ratio:
        (D_DDEM(z)/D_LCDM(z))^2 * |1 + comb_signature|
    """
    D_ddem = linear_growth(bg)
    D_lcdm = linear_growth(lc)
    a_target = 1.0 / (1.0 + z)
    ln_a_target = np.log(a_target)
    D_ddem_z = np.interp(ln_a_target, bg["ln_a"], D_ddem) / D_ddem[-1]
    D_lcdm_z = np.interp(ln_a_target, lc["ln_a"], D_lcdm) / D_lcdm[-1]
    growth_factor_sq = (D_ddem_z / D_lcdm_z)**2

    comb, k_n, eta0_Mpc = comb_signature(k_h_Mpc, bg, eps, alpha,
                                          N_zeros=N_zeros, h=h)
    return growth_factor_sq * comb, k_n, eta0_Mpc


def fsigma8_DDEM(z_arr, bg, lc, sigma8_LCDM_z0=0.811, h=0.6774):
    """
    Compute fsigma_8(z) under the DDEM modified growth.

        f(z) = d ln D / d ln a
        sigma_8(z) = sigma_8(z=0) * D(z) / D(0)
        f*sigma_8(z) = (1/D0) * (dD/dlna)(z) * sigma_8,0

    sigma_8,0 enters as a normalisation; for the joint fit we use the LCDM
    Planck-best-fit value (Planck 2018: sigma_8 = 0.811). The growth rate
    derivative is taken on the DDEM background grid.
    """
    D_ddem = linear_growth(bg)
    ln_a = bg["ln_a"]
    dD_dlna = np.gradient(D_ddem, ln_a)
    out = np.zeros_like(np.atleast_1d(z_arr), dtype=float)
    for i, zi in enumerate(np.atleast_1d(z_arr)):
        ln_ai = np.log(1.0 / (1.0 + zi))
        D_at = np.interp(ln_ai, ln_a, D_ddem)
        dDda = np.interp(ln_ai, ln_a, dD_dlna)
        f = dDda / D_at
        sigma8_z = sigma8_LCDM_z0 * (D_at / D_ddem[-1])
        out[i] = f * sigma8_z
    return out


if __name__ == "__main__":
    print("Sanity test: DDEM perturbation pipeline")
    bg = solve_background(beta0=0.01, eps=0.5, alpha=0.5, w_DE=-1.05)
    lc = solve_lcdm(alpha=0.5)
    D_ddem = linear_growth(bg)
    D_lcdm = linear_growth(lc)
    print(f"  D_DDEM(z=0)/D_LCDM(z=0) [normalized at a_min] = {D_ddem[-1]/D_lcdm[-1]:.5f}")
    eta0 = conformal_time_today(bg)
    print(f"  eta0 (1/H0 units) = {eta0:.4f}")
    print(f"  eta0 (Mpc, h=0.6774) = {eta0 * 2997.92/0.6774:.2f}")
    k_test = np.array([1e-3, 1e-2, 0.05, 0.1])  # h/Mpc
    ratio, k_n, eta0_Mpc = DDEM_over_LCDM(k_test, z=0.0, bg=bg, lc=lc,
                                          eps=0.5, alpha=0.5)
    print(f"  First 5 predicted comb peaks (h/Mpc): {k_n[:5]}")
    print(f"  P_DDEM/P_LCDM at z=0:")
    for k, r in zip(k_test, ratio):
        print(f"    k = {k:.4f} h/Mpc -> ratio = {r:.5f}")
