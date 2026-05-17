"""Pantheon+ SH0ES Type Ia supernova likelihood.

Source: Brout et al. 2022 (ApJ 938 110, arXiv:2202.04077) and Scolnic et al. 2022
(arXiv:2112.03863). Public data products live at PantheonPlusSH0ES/DataRelease.

Implementation follows Cobaya's sn.pantheonplus reference. For cosmology-only
fits we drop SH0ES Cepheid hosts (IS_CALIBRATOR == 1) and apply zHD > 0.01 to
remove peculiar-velocity-dominated objects, leaving 1590 SNe.

The absolute magnitude M_B is analytically marginalized in closed form
(Goliath et al. 2001 / Bridle et al. 2002):

    chi^2_marg = A - B^2 / D
    A = r0^T C^{-1} r0
    B = r0^T C^{-1} 1
    D = 1^T C^{-1} 1
    r0 = m_b_corr - mu_th(z; theta)

Published flat-LCDM benchmark: chi^2 = 1523 / 1580 dof at Om = 0.334
(Brout et al. 2022 Table 3). The pipeline target at fiducial LCDM is to
reproduce this benchmark within +/- a few tens of chi^2 units.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

C_KMS = 299792.458  # km/s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "pantheon_plus"
DAT_FILE = DATA_DIR / "Pantheon+SH0ES.dat"
COV_FILE = DATA_DIR / "Pantheon+SH0ES_STAT+SYS.cov"
sys.path.insert(0, str(ROOT / "theory"))


def _load_pantheonplus():
    """Load Pantheon+ data + STAT+SYS covariance with cosmology cut applied."""
    raw = np.genfromtxt(DAT_FILE, names=True, dtype=None, encoding="utf-8")
    z_HD          = raw["zHD"]
    m_b_corr      = raw["m_b_corr"]
    is_calibrator = raw["IS_CALIBRATOR"].astype(int)
    n_all = z_HD.size  # 1701

    # Load STAT+SYS covariance: first line N=1701, then N*N floats row-major
    with open(COV_FILE, "r") as f:
        n_cov = int(f.readline().strip())
        assert n_cov == n_all, f"cov dim {n_cov} != data rows {n_all}"
        cov_full = np.fromfile(f, sep=" ", count=n_all * n_all).reshape(n_all, n_all)

    # Cosmology mask: drop Cepheid calibrators, drop low-z noise objects.
    mask = (is_calibrator == 0) & (z_HD > 0.01)
    idx = np.where(mask)[0]
    z_cut   = z_HD[idx]
    mb_cut  = m_b_corr[idx]
    cov_cut = cov_full[np.ix_(idx, idx)]

    inv_cov = np.linalg.inv(cov_cut)
    ones    = np.ones(idx.size)
    D       = float(ones @ inv_cov @ ones)
    Cinv_1  = inv_cov @ ones
    # Pre-compute the M_B-marginalized inverse covariance:
    #   chi^2_marg(theta) = r0^T C_marg^{-1} r0
    # with C_marg^{-1} = C^{-1} - (C^{-1} 1)(C^{-1} 1)^T / D
    cov_marg_inv = inv_cov - np.outer(Cinv_1, Cinv_1) / D

    return {"z": z_cut, "m_b": mb_cut, "n": idx.size,
            "cov_marg_inv": cov_marg_inv, "D_norm": D}


_PP = _load_pantheonplus()
N_SNE = _PP["n"]  # 1590


def _luminosity_distance_Mpc(z, H_of_z_kms_per_Mpc, n_int=512):
    """d_L = (1+z) * c * int_0^z dz'/H(z'). Vectorised trapezoidal integration."""
    z_arr = np.atleast_1d(z)
    d_L = np.empty_like(z_arr, dtype=float)
    for i, zi in enumerate(z_arr):
        zp = np.linspace(0.0, zi, n_int)
        Hp = H_of_z_kms_per_Mpc(zp)
        Hp = np.maximum(Hp, 1e-12)
        d_C = C_KMS * np.trapezoid(1.0 / Hp, zp)
        d_L[i] = (1.0 + zi) * d_C
    return d_L


def _mu_theory(z, H_of_z_kms_per_Mpc):
    d_L = _luminosity_distance_Mpc(z, H_of_z_kms_per_Mpc)
    return 5.0 * np.log10(d_L) + 25.0


def chi2_pantheon_plus(theta, bg=None):
    """Pantheon+ chi^2 (M_B analytically marginalized) for DDEM parameters.

    theta = (beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld).
    bg may be supplied (already-solved background dict) to skip the solve.
    Returns float chi^2.
    """
    from ddem_background import solve_background
    beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld = theta

    Om_m  = (omega_b + omega_cdm) / h**2
    Om_r  = 2.473e-5 / h**2 * (1.0 + 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * 3.046)
    Om_DE = 1.0 - Om_m - Om_r

    if bg is None:
        bg = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w0_fld,
                              Omega_m0=Om_m, Omega_DE0=Om_DE, Omega_r0=Om_r,
                              a_min=1e-3, a_max=1.0, n_grid=600, N_zeros=80)

    # Build H(z) in km/s/Mpc from the natural-units background table.
    a_arr = bg["a"]
    H_nat = bg["H"]                                    # H in units of H0_nat
    H0_kms_Mpc = 100.0 * h                             # km/s/Mpc
    z_grid = 1.0 / a_arr - 1.0
    H_kms_Mpc = H_nat / H_nat[-1] * H0_kms_Mpc         # normalised to H0 today
    # bg["a"] runs low to high (a_min->1); z_grid runs high to low.
    order = np.argsort(z_grid)
    z_sorted = z_grid[order]
    H_sorted = H_kms_Mpc[order]

    def H_of_z(z):
        z = np.atleast_1d(z)
        H = np.interp(z, z_sorted, H_sorted,
                      left=H_sorted[0], right=H_sorted[-1])
        return H

    mu_th = _mu_theory(_PP["z"], H_of_z)
    r0    = _PP["m_b"] - mu_th
    return float(r0 @ _PP["cov_marg_inv"] @ r0)


def chi2_lcdm_pantheon(h, Omega_m, Omega_r=0.0):
    """Pantheon+ chi^2 for flat LCDM benchmark check (no DDEM bg needed)."""
    H0_kms_Mpc = 100.0 * h
    Omega_L = 1.0 - Omega_m - Omega_r
    def H_of_z(z):
        return H0_kms_Mpc * np.sqrt(Omega_m * (1+z)**3
                                     + Omega_r * (1+z)**4 + Omega_L)
    mu_th = _mu_theory(_PP["z"], H_of_z)
    r0    = _PP["m_b"] - mu_th
    return float(r0 @ _PP["cov_marg_inv"] @ r0)


def info():
    return {"n_sne": _PP["n"], "z_min": float(_PP["z"].min()),
            "z_max": float(_PP["z"].max()), "data_file": str(DAT_FILE),
            "cov_file": str(COV_FILE)}


if __name__ == "__main__":
    info_d = info()
    print(f"Pantheon+ loaded: {info_d['n_sne']} SNe, "
          f"z in [{info_d['z_min']:.4f}, {info_d['z_max']:.4f}]")

    # LCDM benchmark: Brout et al. 2022 Pantheon+-only best fit
    # Om = 0.334, h arbitrary (M_B marginalized).
    chi2_lcdm = chi2_lcdm_pantheon(h=0.674, Omega_m=0.334)
    print(f"  chi^2 (flat LCDM, Om=0.334) = {chi2_lcdm:.1f}  "
          f"(published benchmark: 1523 / 1580 dof)")
    chi2_lcdm_plk = chi2_lcdm_pantheon(h=0.674, Omega_m=0.315)
    print(f"  chi^2 (flat LCDM, Om=0.315 Planck) = {chi2_lcdm_plk:.1f}")

    # DDEM canonical
    theta = np.array([0.005, 0.2, 0.5, 0.6774, 0.02236, 0.1188, -1.05])
    chi2_ddem = chi2_pantheon_plus(theta)
    print(f"  chi^2 (DDEM canonical) = {chi2_ddem:.1f}")
