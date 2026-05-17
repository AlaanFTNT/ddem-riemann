"""DESI DR1 LRG1 broadband P(k) likelihood.

Source: DESI Collaboration 2024 ("DESI 2024 V: Full-Shape Galaxy Clustering"),
arXiv:2411.12021, with public data products in the DR1 full-shape-bao-clustering
VAC (data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/).

This module loads the LRG1 tracer bin (0.4 < z < 0.6, z_eff = 0.51) compressed
covariance file, which contains the binned P(k) multipoles ell = 0, 2, 4 on
an 80-point k-grid with the full multipole-multipole covariance from 1000
EZmock realisations. The theory model uses an Eisenstein-Hu no-wiggle
transfer function multiplied by the DDEM modified-growth ratio, with the
DDEM comb feature applied multiplicatively. Kaiser linear RSD is applied
with one nuisance parameter (the linear bias b1) per tracer bin.

We use a simple top-hat scale cut k in [0.02, 0.20] h/Mpc rather than the
DESI window-matrix convolution, sacrificing roughly 5-10% in absolute amplitude
calibration in exchange for self-contained likelihood evaluation. This is
the proof-of-concept implementation for the broadband-feature search; a
window-convolved version using pypower BaseMatrix is the natural follow-up.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "theory"))
sys.path.insert(0, str(ROOT / "synthetic_tests"))

from ddem_perturbation import linear_growth, comb_signature, conformal_time_today
from predict_signature import Eisenstein_Hu_pk as _Eisenstein_Hu_pk_raw


def _sigma8_eh(h, Omega_m, Omega_b, n_s, sigma8_target):
    """Return the multiplicative rescaling that sets sigma_8 = sigma8_target
    for the EH no-wiggle P(k)."""
    k_arr = np.logspace(-4, 1, 2000)            # h/Mpc
    P_unnorm = _Eisenstein_Hu_pk_raw(k_arr, h=h, Omega_m=Omega_m,
                                     Omega_b=Omega_b, sigma8=1.0, n_s=n_s)
    R = 8.0  # Mpc/h
    kR = k_arr * R
    W2 = (3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3)**2
    integrand = P_unnorm * W2 * k_arr**2
    var = np.trapezoid(integrand, k_arr) / (2.0 * np.pi**2)
    sig8 = np.sqrt(var)
    return (sigma8_target / sig8)**2            # factor on P(k)


def Eisenstein_Hu_pk(k_h_Mpc, h, Omega_m, Omega_b, sigma8, n_s):
    """sigma_8-normalised no-wiggle Eisenstein-Hu linear matter P(k) at z=0."""
    rescale = _sigma8_eh(h, Omega_m, Omega_b, n_s, sigma8)
    return _Eisenstein_Hu_pk_raw(k_h_Mpc, h=h, Omega_m=Omega_m,
                                 Omega_b=Omega_b, sigma8=sigma8, n_s=n_s) * rescale

# ---------------------------------------------------------------------------
# Data: DESI DR1 LRG1 at z_eff = 0.5096
# ---------------------------------------------------------------------------
DATA_DIR  = ROOT / "desi_dr1" / "lrg1"
COV_FILE  = DATA_DIR / "covariance_spectrum-poles+bao-recon_LRG_GCcomb_z0.4-0.6.h5"
Z_EFF     = 0.5096288678782911            # value verbatim from file attrs

K_MIN     = 0.02         # h/Mpc -- DESI DR1 baseline lower cut
K_MAX     = 0.20         # h/Mpc -- DESI DR1 baseline upper cut


def _load_dr1_lrg1():
    """Load LRG1 compressed multipoles + covariance with scale cut applied."""
    with h5py.File(COV_FILE, "r") as f:
        k0 = np.array(f["observable/spectrum/0/k"])
        k2 = np.array(f["observable/spectrum/2/k"])
        k4 = np.array(f["observable/spectrum/4/k"])
        P0 = np.array(f["observable/spectrum/0/value"])
        P2 = np.array(f["observable/spectrum/2/value"])
        P4 = np.array(f["observable/spectrum/4/value"])
        cov_full = np.array(f["value"])      # 242 x 242 (3 multipoles x 80 + 2 BAO recon)

    n_per_ell = len(k0)                       # 80
    # Scale cut indices
    m0 = (k0 >= K_MIN) & (k0 <= K_MAX)
    m2 = (k2 >= K_MIN) & (k2 <= K_MAX)
    m4 = (k4 >= K_MIN) & (k4 <= K_MAX)

    # Block ordering in the covariance file: [P0(80), P2(80), P4(80), qpar, qper]
    idx0 = np.where(m0)[0]
    idx2 = n_per_ell + np.where(m2)[0]
    idx4 = 2 * n_per_ell + np.where(m4)[0]
    idx_all = np.concatenate([idx0, idx2, idx4])

    k_cut    = np.concatenate([k0[m0], k2[m2], k4[m4]])
    P_cut    = np.concatenate([P0[m0], P2[m2], P4[m4]])
    cov_cut  = cov_full[np.ix_(idx_all, idx_all)]
    inv_cov  = np.linalg.inv(cov_cut)

    ells     = np.concatenate([np.full(m0.sum(), 0),
                               np.full(m2.sum(), 2),
                               np.full(m4.sum(), 4)])
    return {
        "k": k_cut, "P": P_cut, "ells": ells,
        "k0": k0[m0], "k2": k2[m2], "k4": k4[m4],
        "P0": P0[m0], "P2": P2[m2], "P4": P4[m4],
        "cov": cov_cut, "inv_cov": inv_cov,
        "n_data": len(P_cut),
    }


_DR1 = _load_dr1_lrg1()


def predict_lrg1_multipoles(theta, b1, bg=None, lc=None):
    """Predict P_0, P_2, P_4 at the DR1 LRG1 k-bins for parameter vector theta.

    theta = (beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld)
    b1 = linear galaxy bias for LRG1.
    bg, lc = optional precomputed background dicts (DDEM and LCDM) to save work.
    """
    from ddem_background import solve_background, solve_lcdm
    beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld = theta

    Om_m  = (omega_b + omega_cdm) / h**2
    Om_r  = 2.473e-5 / h**2 * (1.0 + 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * 3.046)
    Om_DE = 1.0 - Om_m - Om_r

    if bg is None:
        bg = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w0_fld,
                              Omega_m0=Om_m, Omega_DE0=Om_DE, Omega_r0=Om_r,
                              a_min=1e-3, a_max=1.0, n_grid=600, N_zeros=80)
    if lc is None:
        lc = solve_lcdm(alpha=alpha, w_DE=-1.0,
                        Omega_m0=Om_m, Omega_DE0=1.0 - Om_m - Om_r,
                        Omega_r0=Om_r, a_min=1e-3, a_max=1.0, n_grid=600,
                        N_zeros=80)

    # Growth factors
    D_ddem = linear_growth(bg)
    D_lcdm = linear_growth(lc)
    a_z = 1.0 / (1.0 + Z_EFF)
    lnaz = np.log(a_z)
    D_ddem_z = np.interp(lnaz, bg["ln_a"], D_ddem) / D_ddem[-1]
    D_lcdm_z = np.interp(lnaz, lc["ln_a"], D_lcdm) / D_lcdm[-1]
    growth_sq = D_ddem_z**2          # absolute D^2 (z_eff) relative to z=0 DDEM

    # Logarithmic growth rate f(z_eff) = dlnD/dlna evaluated at z_eff
    dD_dlna_ddem = np.gradient(D_ddem, bg["ln_a"])
    D_at_z   = np.interp(lnaz, bg["ln_a"], D_ddem)
    dD_at_z  = np.interp(lnaz, bg["ln_a"], dD_dlna_ddem)
    f_at_z   = dD_at_z / D_at_z

    # Linear matter P(k) at z = 0 (no-wiggle EH), scaled to z_eff
    pk_smooth_z0 = Eisenstein_Hu_pk(_DR1["k0"], h=h, Omega_m=Om_m,
                                    Omega_b=omega_b/h**2, sigma8=0.811,
                                    n_s=0.9665)
    pk_smooth_z = pk_smooth_z0 * growth_sq

    # Comb modification (1 + delta_comb(k))
    comb_factor, _, _ = comb_signature(_DR1["k0"], bg, eps=eps, alpha=alpha,
                                       N_zeros=80, h=h)
    pk_lin_z = pk_smooth_z * comb_factor

    # Kaiser RSD multipoles
    beta_k = f_at_z / b1
    A0 = 1.0 + (2.0/3.0) * beta_k + (1.0/5.0) * beta_k**2
    A2 = (4.0/3.0) * beta_k + (4.0/7.0) * beta_k**2
    A4 = (8.0/35.0) * beta_k**2
    pre = (b1**2) * pk_lin_z

    P0_th = A0 * pre
    P2_th = A2 * pre
    P4_th = A4 * pre
    return P0_th, P2_th, P4_th


def chi2_dr1_lrg1(theta, b1, bg=None, lc=None):
    P0_th, P2_th, P4_th = predict_lrg1_multipoles(theta, b1, bg=bg, lc=lc)
    pred = np.concatenate([P0_th, P2_th, P4_th])
    diff = _DR1["P"] - pred
    return float(diff @ _DR1["inv_cov"] @ diff)


def info():
    return {
        "n_data":       _DR1["n_data"],
        "k_min":        K_MIN,
        "k_max":        K_MAX,
        "z_eff":        Z_EFF,
        "n_per_ell":    [(_DR1["ells"] == ell).sum() for ell in (0, 2, 4)],
    }


if __name__ == "__main__":
    print("DESI DR1 LRG1 broadband P(k) likelihood")
    print(f"  z_eff = {Z_EFF:.4f}")
    info_d = info()
    print(f"  k-cut [{K_MIN}, {K_MAX}] h/Mpc")
    print(f"  n_data = {info_d['n_data']}  (ell=0:{info_d['n_per_ell'][0]}, "
          f"ell=2:{info_d['n_per_ell'][1]}, ell=4:{info_d['n_per_ell'][2]})")
    print()
    theta = np.array([0.005, 0.2, 0.5, 0.6774, 0.02236, 0.1188, -1.05])
    for b1 in [1.5, 2.0, 2.5]:
        chi2 = chi2_dr1_lrg1(theta, b1)
        print(f"  theta = canonical, b1 = {b1}:  chi^2 = {chi2:.2f}  (DOF ~ {info_d['n_data']-1})")
