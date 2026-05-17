"""Sign-changeable Interacting Dark Energy (S-IDE) competitor model.

Reference: Li et al. 2025, "Probing the sign-changeable interaction in the
dark sector using DESI BAO + DES SN" (arXiv:2501.07361). Their IDE3 variant
is the best-evidenced competitor for our oscillatory model:

    Q(a) = beta(a) * H0 * rho_DE(a),    beta(a) = beta0 * a + beta_e * (1 - a)

with w_DE fixed at -1 (cosmological constant). The sign change happens when
beta0 and beta_e have opposite signs, which Li et al. find detected with
beta0 < 0 (late-time, DE -> DM) and beta_e > 0 (early-time, DM -> DE).

This module provides a drop-in coupled background solver matching the
existing solve_background interface, plus a chi^2 evaluator over the
6-likelihood pipeline (BAO + Planck + fsigma8 + Beutler null + Qu lensing
+ Pantheon+). The Beutler bound is trivially satisfied (no comb), so it
contributes zero at any S-IDE parameter point.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def solve_background_side(beta0, beta_e,
                          Omega_m0=0.315, Omega_DE0=0.685, Omega_r0=9.05e-5,
                          a_min=1e-4, a_max=1.0, n_grid=2000):
    """Forward-integrate (rho_m, rho_DE) under the Li-2025 IDE3 Q-kernel."""
    lnA = np.linspace(np.log(a_min), np.log(a_max), n_grid)

    def rhs(ln_a, y):
        rho_m, rho_DE = y
        a = np.exp(ln_a)
        rho_r = Omega_r0 * a**-4
        rho_m_safe = max(rho_m, 1e-30)
        rho_DE_safe = max(rho_DE, 1e-30)
        H = np.sqrt(rho_m_safe + rho_r + rho_DE_safe)
        beta_a = beta0 * a + beta_e * (1.0 - a)
        Q = beta_a * 1.0 * rho_DE_safe                  # H0=1 natural units
        # dot(rho_m) + 3H rho_m = +Q  ->  d rho_m / d ln a = -3 rho_m + Q/H
        # dot(rho_DE) + 3H (1+w_DE) rho_DE = -Q;  w_DE = -1 so 1+w_DE = 0
        # ->  d rho_DE / d ln a = -Q/H
        drho_m  = -3.0 * rho_m_safe + Q / H
        drho_DE = -Q / H
        return [drho_m, drho_DE]

    # ICs at a_min (deep matter era; coupling small): standard scaling
    rho_m_init  = Omega_m0 * a_min**-3
    rho_DE_init = Omega_DE0
    sol = solve_ivp(rhs, [lnA[0], lnA[-1]], [rho_m_init, rho_DE_init],
                    t_eval=lnA, method="DOP853", rtol=1e-9, atol=1e-12)
    if not sol.success:
        return None

    a = np.exp(lnA)
    rho_m  = sol.y[0]
    rho_DE = sol.y[1]
    rho_r  = Omega_r0 * a**-4
    H = np.sqrt(np.maximum(rho_m, 1e-30) + rho_r + np.maximum(rho_DE, 1e-30))
    beta_a = beta0 * a + beta_e * (1.0 - a)
    Q = beta_a * np.maximum(rho_DE, 1e-30)
    return {"a": a, "ln_a": lnA, "rho_m": rho_m, "rho_DE": rho_DE,
            "rho_r": rho_r, "H": H, "Q": Q,
            "params": {"beta0": beta0, "beta_e": beta_e, "w_DE": -1.0,
                       "Omega_m0": Omega_m0, "Omega_DE0": Omega_DE0}}


def linear_growth_side(bg):
    """Modified linear growth under the Li-2025 friction term."""
    a_arr  = bg["a"]
    H_arr  = bg["H"]
    rho_m  = bg["rho_m"]
    Q_arr  = bg["Q"]
    lnA    = bg["ln_a"]
    Om_m   = rho_m / H_arr**2
    gamma_F  = a_arr * Q_arr / (H_arr * rho_m)
    dlnH_dlna = np.gradient(np.log(H_arr), lnA)
    dgF_dlna  = np.gradient(gamma_F, lnA)

    def rhs(N, y):
        D, Dp = y
        gF   = np.interp(N, lnA, gamma_F)
        dlnH = np.interp(N, lnA, dlnH_dlna)
        dgF  = np.interp(N, lnA, dgF_dlna)
        Om   = np.interp(N, lnA, Om_m)
        d2D  = -(2.0 + dlnH + gF) * Dp - (dgF - 1.5 * Om) * D
        return [Dp, d2D]

    N0, Nf = lnA[0], lnA[-1]
    sol = solve_ivp(rhs, [N0, Nf], [np.exp(N0), np.exp(N0)],
                    t_eval=lnA, method="DOP853", rtol=1e-9, atol=1e-12)
    return sol.y[0]


def chi2_side_joint(beta0, beta_e, h, omega_b, omega_cdm):
    """Compute the joint chi^2 for the Li 2025 S-IDE model under the same
    6-likelihood pipeline. Returns total chi^2 and component breakdown.

    No oscillatory feature -> Beutler chi^2 = 0 by construction.
    """
    sys.path.insert(0, str(HERE.parent / "cobaya"))
    from desi_dr2_bao_data import chi2_desi_bao
    from growth_data import chi2_fsigma8
    from cmb_lensing_qu25 import chi2_cmb_lensing
    from pantheon_plus import chi2_pantheon_plus
    from run_joint_mcmc import chi2_planck, rd_aubourg, DM_Mpc, OMEGA_GAMMA_H2
    SIGMA8_LCDM_Z0 = 0.811

    Om_m  = (omega_b + omega_cdm) / h**2
    Om_r  = OMEGA_GAMMA_H2 / h**2 * (1.0 + 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * 3.046)
    Om_DE = 1.0 - Om_m - Om_r

    try:
        bg = solve_background_side(beta0=beta0, beta_e=beta_e,
                                   Omega_m0=Om_m, Omega_DE0=Om_DE, Omega_r0=Om_r,
                                   a_min=1e-4, a_max=1.0, n_grid=2000)
        # For LCDM reference (growth comparison):
        from ddem_background import solve_lcdm
        lc = solve_lcdm(alpha=0.5, w_DE=-1.0,
                        Omega_m0=Om_m, Omega_DE0=1.0 - Om_m - Om_r,
                        Omega_r0=Om_r, a_min=1e-4, a_max=1.0, n_grid=2000,
                        N_zeros=20)
    except Exception:
        return np.inf, {}

    a = bg["a"]; H = bg["H"]
    rd = rd_aubourg(omega_b, omega_cdm)
    def DM_over_rd(z): return DM_Mpc(bg, z, h) / rd
    def DH_over_rd(z): return 2997.92458 / (np.interp(1.0/(1.0+z), a, H) * h * rd)
    def DV_over_rd(z):
        DM = DM_Mpc(bg, z, h)
        DH = 2997.92458 / (np.interp(1.0/(1.0+z), a, H) * h)
        return (z * DM**2 * DH)**(1.0/3.0) / rd
    chi2_BAO = chi2_desi_bao(DM_over_rd, DH_over_rd, DV_over_rd)
    chi2_Pl  = chi2_planck(h, omega_b, omega_cdm)

    # fsigma8 prediction: need linear-growth D_DDEM-equivalent and D_LCDM
    D_side = linear_growth_side(bg)
    from ddem_perturbation import linear_growth
    D_lcdm = linear_growth(lc)
    # f(a) = d ln D / d ln a
    def fsigma8_side(z):
        a_z = 1.0/(1.0+z)
        lna = np.log(a_z)
        D_at  = np.interp(lna, bg["ln_a"], D_side)
        Dl_at = np.interp(lna, lc["ln_a"], D_lcdm)
        sigma8_a = SIGMA8_LCDM_Z0 * (D_at / D_lcdm[-1])
        f_at = np.interp(lna, bg["ln_a"], np.gradient(D_side, bg["ln_a"]) / np.maximum(D_side, 1e-30))
        return f_at * sigma8_a
    chi2_fs8 = chi2_fsigma8(fsigma8_side)

    # Beutler: S-IDE has no comb -> 0
    chi2_F = 0.0

    # Lensing: sigma_8(z=0) from D_DDEM(z=0)/D_LCDM(z=0)
    sigma8_z0 = SIGMA8_LCDM_Z0 * (D_side[-1] / D_lcdm[-1])
    chi2_Lens = chi2_cmb_lensing(sigma8_z0, Om_m)

    # Pantheon+: use the bg H(z) through luminosity distance
    # Reuse the chi2_pantheon_plus path with a custom theta-equivalent: but
    # the function expects DDEM theta, not S-IDE. We need a small adapter.
    from pantheon_plus import _PP
    C_KMS = 299792.458
    H0_kms_Mpc = 100.0 * h
    z_grid = 1.0/a - 1.0
    H_kms = H / H[-1] * H0_kms_Mpc
    order = np.argsort(z_grid)
    z_sorted = z_grid[order]; H_sorted = H_kms[order]
    def H_of_z(z):
        z = np.atleast_1d(z)
        return np.interp(z, z_sorted, H_sorted, left=H_sorted[0], right=H_sorted[-1])
    # luminosity distance
    def mu_th(z_arr):
        d_L = np.empty_like(z_arr, dtype=float)
        for i, zi in enumerate(z_arr):
            zp = np.linspace(0.0, zi, 256)
            Hp = np.maximum(H_of_z(zp), 1e-12)
            d_L[i] = (1.0 + zi) * C_KMS * np.trapezoid(1.0/Hp, zp)
        return 5.0 * np.log10(d_L) + 25.0
    mu_pred = mu_th(_PP["z"])
    r0 = _PP["m_b"] - mu_pred
    chi2_SN = float(r0 @ _PP["cov_marg_inv"] @ r0)

    total = chi2_BAO + chi2_Pl + chi2_fs8 + chi2_F + chi2_Lens + chi2_SN
    return total, {"BAO": chi2_BAO, "Planck": chi2_Pl, "fsigma8": chi2_fs8,
                   "feature": chi2_F, "lensing": chi2_Lens, "Pantheon+": chi2_SN,
                   "sigma8_z0": sigma8_z0, "Om_m": Om_m}


if __name__ == "__main__":
    print("Li et al. 2025 IDE3 head-to-head against DDEM")
    print("=" * 60)
    # Li et al. best-fit IDE3 values
    print("At Li+2025 IDE3 best-fit (beta0=-2.69, beta_e=+4.35):")
    chi2_li, parts = chi2_side_joint(beta0=-2.69, beta_e=+4.35,
                                     h=0.6685, omega_b=0.02236, omega_cdm=0.1202)
    print(f"  total chi^2 = {chi2_li:.2f}")
    for k, v in parts.items():
        print(f"    {k:12s} = {v:.4f}")
    print()
    # ΛCDM control (β=0)
    print("At LCDM control (beta0=0, beta_e=0):")
    chi2_lcdm, parts = chi2_side_joint(beta0=0.0, beta_e=0.0,
                                       h=0.6727, omega_b=0.02236, omega_cdm=0.1202)
    print(f"  total chi^2 = {chi2_lcdm:.2f}")
    for k, v in parts.items():
        print(f"    {k:12s} = {v:.4f}")
