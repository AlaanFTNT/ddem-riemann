"""Joint MCMC: DESI DR2 BAO + Planck distance priors + compiled fsigma_8 + Beutler feature.

This is the perturbation-level real-data test. The comb prediction enters
the likelihood via:

  (a) Modified linear growth -> shift in fsigma_8(z), constrained by the
      compiled RSD compilation (13 measurements: 6dFGRS, BOSS DR12, eBOSS
      DR16, DESI DR1 full-shape).
  (b) Per-mode comb amplitude in P(k) -> constrained by the Beutler et al.
      2023 95% upper bound on log-oscillation features (arXiv:2303.13946,
      |A_log|_95 = 0.04 for omega_log in [10, 360]). Each gamma_n inside
      the omega_log window contributes independently.

Background distances and the sound horizon are computed from the modified
background solver in theory/ddem_background.py.

Posterior is saved to chain_joint.npy.

All numerical inputs (Planck distance priors, fsigma_8 measurements, Aubourg
r_d fit, Beutler feature bound) are verified against the cited primary
publications.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import emcee

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "theory"))
sys.path.insert(0, str(HERE))

from ddem_background import solve_background, solve_lcdm
from ddem_perturbation import linear_growth, fsigma8_DDEM
from desi_dr2_bao_data import chi2_desi_bao
from growth_data import chi2_fsigma8
from feature_amplitude import chi2_feature_per_mode
from cmb_lensing_qu25 import chi2_cmb_lensing as chi2_act_lensing
from pantheon_plus import chi2_pantheon_plus, N_SNE

CHAIN_PATH = HERE / "chain_joint.npy"
LNP_PATH   = HERE / "chain_joint_lnp.npy"

# ---------------------------------------------------------------------------
# Planck 2018 marginal constraints on (h, omega_b, omega_cdm).
# Source: Planck Collaboration 2020 (arXiv:1807.06209), Table 2,
# TT,TE,EE+lowE column (NO lensing -- switched from +lensing column to avoid
# double-counting the Planck PR4 lensing maps now entering the pipeline via
# the Qu et al. 2025 joint CMB-lensing likelihood).
# ---------------------------------------------------------------------------
PLANCK_H_MEAN     = 0.6727
PLANCK_H_SIG      = 0.0060
PLANCK_WB_MEAN    = 0.02236
PLANCK_WB_SIG     = 0.00015
PLANCK_WCDM_MEAN  = 0.1202
PLANCK_WCDM_SIG   = 0.0014

T_CMB_K        = 2.7255       # K
OMEGA_GAMMA_H2 = 2.473e-5     # Omega_gamma h^2 at T_CMB = 2.7255 K


def chi2_planck(h, omega_b, omega_cdm):
    return (((h - PLANCK_H_MEAN) / PLANCK_H_SIG)**2
          + ((omega_b - PLANCK_WB_MEAN) / PLANCK_WB_SIG)**2
          + ((omega_cdm - PLANCK_WCDM_MEAN) / PLANCK_WCDM_SIG)**2)


# ---------------------------------------------------------------------------
# Drag-epoch sound horizon -- Aubourg et al. 2015, arXiv:1411.1074 Eq. (16).
# r_d = 55.154 * exp[-72.3*(omega_nu + 0.0006)^2]
#       * (omega_b + omega_cdm)^(-0.25351) * omega_b^(-0.12807)   [Mpc]
# Accurate to 0.021% for N_eff = 3.046 and m_nu in {0, 0.06 eV}.
# We assume massless neutrinos: omega_nu = 0.
# ---------------------------------------------------------------------------
def rd_aubourg(omega_b, omega_cdm, omega_nu=0.0):
    omh2 = omega_b + omega_cdm
    return 55.154 * np.exp(-72.3 * (omega_nu + 0.0006)**2) \
           * omh2**(-0.25351) * omega_b**(-0.12807)


# ---------------------------------------------------------------------------
# Sound horizon at recombination r_s(z_*) -- numerical integration.
# c_s(a) = 1 / sqrt(3 (1 + R_b(a)))  in units of c
# R_b(a) = (3/4) * omega_b / Omega_gamma_h2 * a
# Then r_s(z_*) = c/H0 * int_0^a_* da / (a^2 H(a) sqrt(3(1+R_b(a))))   [Mpc]
# ---------------------------------------------------------------------------
def rs_at_z_star(bg, omega_b, h, Omega_r0):
    """Sound horizon at recombination by numerical integration plus an
    analytic radiation-era tail correction for [0, a_min] which the
    background grid does not cover.

    In the radiation-dominated era H(a) = H0 * sqrt(Omega_r) * a^-2, so
    1/(a^2 H) is constant in a. With c_s = 1/sqrt(3(1+R_b)) and
    R_b(a) = R_b0 * a, the closed-form integral is

        int_0^a_min  c_s / (a^2 H) da
          = (1 / (H0 sqrt(3 Omega_r))) * (2 / R_b0)
            * [ sqrt(1 + R_b0 * a_min) - 1 ].
    """
    a_star = 1.0 / (1.0 + Z_STAR_PLANCK)
    a = bg["a"]
    H = bg["H"]                      # in H0 units
    mask = a <= a_star
    a_int = a[mask]
    H_int = H[mask]
    if a_int.size < 5:
        return np.nan
    a_grid_min = float(a_int[0])
    R_b_grid = (3.0 / 4.0) * omega_b / OMEGA_GAMMA_H2 * a_int
    cs_grid  = 1.0 / np.sqrt(3.0 * (1.0 + R_b_grid))
    integrand = cs_grid / (a_int**2 * H_int)
    r_s_num = float(np.trapezoid(integrand, a_int))

    # Analytic radiation-era tail from a = 0 to a_grid_min
    R_b0 = (3.0 / 4.0) * omega_b / OMEGA_GAMMA_H2
    tail = (1.0 / np.sqrt(3.0 * Omega_r0)) * (2.0 / R_b0) \
           * (np.sqrt(1.0 + R_b0 * a_grid_min) - 1.0)
    r_s_natural = r_s_num + tail
    return r_s_natural * 2997.92458 / h         # Mpc


# ---------------------------------------------------------------------------
# Comoving angular diameter distance D_A(z) and D_M(z) = (1+z) D_A(z).
# In a flat universe D_M(z) = c * integral_{a(z)}^{1} da / (a^2 H(a)).
# ---------------------------------------------------------------------------
def DM_Mpc(bg, z, h):
    a_z = 1.0 / (1.0 + z)
    a = bg["a"]
    H = bg["H"]
    idx = a >= a_z
    a_int = a[idx]
    H_int = H[idx]
    if a_int.size < 2:
        return 0.0
    a_grid = np.concatenate([[a_z], a_int])
    H_grid = np.concatenate([[np.interp(a_z, a, H)], H_int])
    integrand = 1.0 / (a_grid**2 * H_grid)
    I = float(np.trapezoid(integrand, a_grid))
    return I * 2997.92458 / h


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
def compute_predictions(theta):
    beta0, eps, alpha, h, omega_b, omega_cdm, w0_fld = theta

    Om_m  = (omega_b + omega_cdm) / h**2
    Om_r  = OMEGA_GAMMA_H2 / h**2 * (1.0 + 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * 3.046)
    Om_DE = 1.0 - Om_m - Om_r

    try:
        # Push a_min to ~1e-4 so the recombination epoch (a ~ 9.17e-4) is inside.
        bg = solve_background(
            beta0=beta0, eps=eps, alpha=alpha, w_DE=w0_fld,
            Omega_m0=Om_m, Omega_DE0=Om_DE, Omega_r0=Om_r,
            a_min=1e-4, a_max=1.0, n_grid=800, N_zeros=80,
        )
        lc = solve_lcdm(
            alpha=alpha, w_DE=-1.0,
            Omega_m0=Om_m, Omega_DE0=1.0 - Om_m - Om_r, Omega_r0=Om_r,
            a_min=1e-4, a_max=1.0, n_grid=800, N_zeros=80,
        )
    except Exception:
        return np.inf, {}

    a = bg["a"]
    H = bg["H"]

    # BAO predictions
    rd = rd_aubourg(omega_b, omega_cdm)
    def DM_over_rd(z): return DM_Mpc(bg, z, h) / rd
    def DH_over_rd(z): return 2997.92458 / (np.interp(1.0/(1.0+z), a, H) * h * rd)
    def DV_over_rd(z):
        DM = DM_Mpc(bg, z, h)
        DH = 2997.92458 / (np.interp(1.0/(1.0+z), a, H) * h)
        return (z * DM**2 * DH)**(1.0/3.0) / rd
    chi2_BAO = chi2_desi_bao(DM_over_rd, DH_over_rd, DV_over_rd)

    # Planck marginals on (h, omega_b, omega_cdm)
    chi2_Pl = chi2_planck(h, omega_b, omega_cdm)

    # Growth rate fsigma_8(z) -- LCDM-anchored sigma_8,0 = 0.811 (Planck 2018).
    SIGMA8_LCDM_Z0 = 0.811
    def pred_fs8(z): return fsigma8_DDEM(z, bg, lc, sigma8_LCDM_z0=SIGMA8_LCDM_Z0, h=h)
    chi2_fs8 = chi2_fsigma8(pred_fs8)

    # Per-mode Beutler feature constraint
    chi2_F, A_max = chi2_feature_per_mode(eps=eps, alpha=alpha, a_eff=0.6,
                                           beta0=beta0, use_boltzmann=True)

    # ACT DR6 lensing: S_8^CMBL = sigma_8(z=0) * (Omega_m/0.3)^0.25
    D_ddem = linear_growth(bg)
    D_lcdm = linear_growth(lc)
    sigma8_z0 = SIGMA8_LCDM_Z0 * (D_ddem[-1] / D_lcdm[-1])
    chi2_ACT = chi2_act_lensing(sigma8_z0, Om_m)

    # Pantheon+ SNe Ia (Brout et al. 2022). M_B is analytically marginalized.
    chi2_SN = chi2_pantheon_plus(theta, bg=bg)

    total = chi2_BAO + chi2_Pl + chi2_fs8 + chi2_F + chi2_ACT + chi2_SN
    return total, {
        "chi2_BAO": chi2_BAO, "chi2_Planck": chi2_Pl,
        "chi2_fsigma8": chi2_fs8, "chi2_feature": chi2_F,
        "chi2_ACT": chi2_ACT, "chi2_Pantheon+": chi2_SN,
        "max_comb_amp": A_max, "r_d_Mpc": rd,
        "sigma8_z0": sigma8_z0, "S8_CMBL": sigma8_z0 * (Om_m/0.3)**0.25,
    }


PRIORS = {
    "beta0":     (0.0, 0.02),
    "eps":       (0.0, 2.0),
    "alpha":     (-1.0, 2.0),
    "h":         (0.55, 0.85),
    "omega_b":   (0.018, 0.026),
    "omega_cdm": (0.08, 0.16),
    "w0_fld":    (-1.5, -1.01),
}
LO = np.array([v[0] for v in PRIORS.values()])
HI = np.array([v[1] for v in PRIORS.values()])


def log_prior(theta):
    if np.any(theta < LO) or np.any(theta > HI):
        return -np.inf
    return 0.0


def log_post(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    chi2, _ = compute_predictions(theta)
    if not np.isfinite(chi2):
        return -np.inf
    return lp - 0.5 * chi2


def main(nwalkers=64, nsteps=3000, burn=600, nproc=12):
    ndim = 7
    init_center = np.array([0.005, 0.3, 0.5, 0.6774, 0.0223, 0.1188, -1.05])
    init_scale  = np.array([0.003, 0.2, 0.3, 0.005, 0.0005, 0.002, 0.02])
    p0 = init_center + init_scale * np.random.randn(nwalkers, ndim)
    p0 = np.clip(p0, LO + 1e-6, HI - 1e-6)

    print(f"  warming Riemann-zero cache + checking pipeline...")
    chi2_init, parts_init = compute_predictions(init_center)
    print(f"  init chi^2 = {chi2_init:.3f}")
    for k, v in parts_init.items():
        print(f"    {k:15s} = {v:.4f}")

    print(f"Joint MCMC: BAO + Planck + 13 fsigma_8 + per-mode Beutler + ACT DR6 lensing + Pantheon+ ({N_SNE} SNe)")
    print(f"  walkers={nwalkers}  steps={nsteps}  burn={burn}  procs={nproc}")
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=False)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s  ({dt/nsteps*1000:.1f}ms/step/walker)")
    print(f"  acceptance fraction = {np.mean(sampler.acceptance_fraction):.3f}")

    chain = sampler.get_chain(discard=burn, flat=True)
    lnp = sampler.get_log_prob(discard=burn, flat=True)
    np.save(CHAIN_PATH, chain)
    np.save(LNP_PATH, lnp)
    print(f"  chain shape: {chain.shape}")
    print(f"  saved: {CHAIN_PATH.name}")

    print()
    print("Posterior quantiles (16/50/84) and (2.5/97.5):")
    labels = list(PRIORS.keys())
    for i, lab in enumerate(labels):
        q025, q16, q50, q84, q975 = np.percentile(chain[:, i], [2.5, 16, 50, 84, 97.5])
        print(f"  {lab:10s}  med={q50:+.5f}  68%=[{q16:+.5f}, {q84:+.5f}]  "
              f"95%=[{q025:+.5f}, {q975:+.5f}]")

    i_map = int(np.argmax(lnp))
    theta_map = chain[i_map]
    _, parts = compute_predictions(theta_map)
    print()
    print("MAP chi^2 decomposition:")
    for k, v in parts.items():
        print(f"  {k:15s} = {v:.4f}")

    return chain, lnp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps",   type=int, default=3000)
    ap.add_argument("--burn",     type=int, default=600)
    ap.add_argument("--nproc",    type=int, default=12)
    a = ap.parse_args()
    main(nwalkers=a.nwalkers, nsteps=a.nsteps, burn=a.burn, nproc=a.nproc)
