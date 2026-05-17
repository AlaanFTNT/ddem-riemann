"""CMB lensing C_L^kappa-kappa prediction for DDEM, using Limber projection
of the DDEM-modified linear matter power spectrum.

The CMB lensing convergence power spectrum is

    C_L^{kk} = int_0^z* dz [W_kappa(z)]^2 / (chi(z)^2 H(z)) P(k=(L+0.5)/chi(z), z)

with

    W_kappa(z) = (3/2) Omega_m H_0^2 (1+z) chi(z) (chi* - chi(z)) / chi*

where chi* is the comoving distance to last scattering (z* ~ 1090).

DDEM modifies (i) the background H(z) through the IDE coupling, (ii) the
linear growth D(a) through the Q-friction term, and (iii) the matter
power spectrum through the per-mode comb amplitude A_n(beta0, eps, alpha)
from the §8.2 Boltzmann solver. The first two enter via the LCDM-anchored
P(k, z) = D(z)^2 P_LCDM(k, 0) ratio; the third via the multiplicative
comb modulation P_DDEM/P_LCDM = 1 + sum_n A_n cos(gamma_n ln(k/k_*)) plus
phase, with the dynamically-derived A_n from theory/boltzmann_perturbation.py.

The residual C_L^{kk,DDEM} / C_L^{kk,LCDM} - 1 is the new observable. The
multi-frequency comb in P(k) maps to a multi-frequency oscillation in L
through the Limber projection k = (L + 1/2) / chi(z) at peak z ~ 2; the
finite kernel width spreads each k_n = gamma_n/eta_0 peak across a range
of L by Delta L / L ~ Delta chi / chi (the kernel width).

We compare against the ACT DR6 binned bandpowers L in [40, 1300] with
fractional uncertainty 0.5-5%, and forecast against SO / CMB-S4.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ddem_background import solve_background, solve_lcdm, riemann_zeros


C_KMS = 299792.458    # km/s
H0_KMS_MPC = 67.27    # Planck 2018 marginals (TT,TE,EE+lowE column)
H_PARAM    = 0.6727
OMEGA_M0   = 0.315
Z_STAR     = 1090.0


def chi_of_z(z, H_of_z_kms_per_Mpc, n_int=512):
    """Comoving distance chi(z) = c int_0^z dz'/H(z')  [Mpc]."""
    z_arr = np.atleast_1d(z)
    chi = np.empty_like(z_arr, dtype=float)
    for i, zi in enumerate(z_arr):
        zp = np.linspace(0.0, zi, n_int)
        Hp = H_of_z_kms_per_Mpc(zp)
        Hp = np.maximum(Hp, 1e-12)
        chi[i] = C_KMS * np.trapezoid(1.0 / Hp, zp)
    return chi


def lensing_kernel_W(chi_z, chi_star, z, H0=H0_KMS_MPC, Omega_m=OMEGA_M0):
    """W_kappa(z) = (3/2) Omega_m H_0^2 (1+z) chi(z) (chi* - chi(z)) / chi*
    in units of Mpc^-1.  H_0 in km/s/Mpc, chi in Mpc."""
    H0_over_c = H0 / C_KMS   # 1/Mpc
    return 1.5 * Omega_m * H0_over_c**2 * (1 + z) * chi_z * (chi_star - chi_z) / chi_star


def build_growth_factor(bg):
    """Linear-growth D(a) from background ddem_perturbation.linear_growth."""
    from ddem_perturbation import linear_growth
    D = linear_growth(bg)
    return bg["ln_a"], D / D[-1]  # normalised to D(a=1) = 1


def power_spectrum_z(k_h, z, bg, P_LCDM_z0_fn, comb_modifier_fn=None):
    """Compute P(k, z) for the DDEM background. P_LCDM_z0_fn(k_h) is the
    LCDM P(k) at z=0 (Eisenstein-Hu or similar). comb_modifier_fn(k_h)
    multiplies P_LCDM by the DDEM comb factor."""
    lnA, D_norm = build_growth_factor(bg)
    a_z = 1.0 / (1.0 + z)
    lna_z = np.log(a_z)
    Dz = float(np.interp(lna_z, lnA, D_norm))
    P0 = P_LCDM_z0_fn(k_h)
    if comb_modifier_fn is not None:
        P0 = P0 * comb_modifier_fn(k_h)
    return Dz**2 * P0


def _eh_no_wiggle_P(k_h, h=H_PARAM, Omega_m=OMEGA_M0, Omega_b=0.0494,
                     sigma8=0.811, n_s=0.9665):
    """Wrapper for the existing Eisenstein-Hu no-wiggle P(k) module."""
    sys.path.insert(0, str(HERE.parent / "synthetic_tests"))
    sys.path.insert(0, str(HERE.parent / "cobaya"))
    from dr1_lrg1_pk import Eisenstein_Hu_pk
    return Eisenstein_Hu_pk(k_h, h=h, Omega_m=Omega_m, Omega_b=Omega_b,
                              sigma8=sigma8, n_s=n_s)


def ddem_comb_modifier(k_h, beta0, eps, alpha, h=H_PARAM,
                        eta_0_Mpc=14136.0, N_zeros=80, R_calib=0.002):
    """Multiplicative comb factor on P(k) at z=0 in the Boltzmann-calibrated
    amplitude. The per-mode amplitude is

        A_n = R_calib * 2 eps a_eff^alpha / |rho_n| * (beta0 / 0.005)

    Peak positions k_n = gamma_n / eta_0. Width sigma_{ln k} = alpha / gamma_n.
    """
    gammas = riemann_zeros(N_zeros)
    rho_mod = np.sqrt(alpha**2 + gammas**2)
    k_n_h_Mpc = np.array([g / eta_0_Mpc / h for g in gammas])
    a_eff = 0.6
    A_n = R_calib * 2.0 * eps * (a_eff**alpha) / rho_mod * (beta0 / 0.005)
    sigma_lnk = alpha / gammas
    lnk = np.log(k_h)
    modifier = np.ones_like(k_h)
    for i, kn in enumerate(k_n_h_Mpc):
        if kn < k_h[0] or kn > k_h[-1]:
            continue
        gauss = np.exp(-(lnk - np.log(kn))**2 / (2 * sigma_lnk[i]**2))
        modifier = modifier * (1 + A_n[i] * np.cos(gammas[i] * lnk) * gauss)
    return modifier


def compute_clkk_residual(beta0=0.005, eps=0.2, alpha=0.5, w_DE=-1.05,
                           L_grid=None, n_z=80, z_max=5.0):
    """Compute C_L^{kappa-kappa} residual: DDEM vs LCDM, via Limber.

    Returns (L_grid, residual_array) where residual = C_DDEM/C_LCDM - 1.
    """
    if L_grid is None:
        L_grid = np.logspace(np.log10(40), np.log10(1300), 50)

    bg_ddem = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w_DE,
                                 a_min=1e-4, a_max=1.0, n_grid=2000, N_zeros=80)
    bg_lcdm = solve_lcdm(alpha=alpha, w_DE=-1.0, a_min=1e-4, a_max=1.0,
                          n_grid=2000, N_zeros=20)
    # H(z) for chi computation: use DDEM's H for DDEM, LCDM's H for LCDM
    def make_H_of_z(bg):
        a_arr = bg["a"]
        H_nat = bg["H"]
        H_kms = H_nat / H_nat[-1] * H0_KMS_MPC
        z_arr = 1.0/a_arr - 1.0
        order = np.argsort(z_arr)
        return lambda z: np.interp(z, z_arr[order], H_kms[order],
                                    left=H_kms[order][0], right=H_kms[order][-1])
    H_ddem = make_H_of_z(bg_ddem)
    H_lcdm = make_H_of_z(bg_lcdm)

    z_int  = np.linspace(0.01, z_max, n_z)
    chi_ddem  = chi_of_z(z_int, H_ddem)
    chi_lcdm  = chi_of_z(z_int, H_lcdm)
    chi_star_ddem = chi_of_z([Z_STAR], H_ddem)[0]
    chi_star_lcdm = chi_of_z([Z_STAR], H_lcdm)[0]

    W_ddem = lensing_kernel_W(chi_ddem, chi_star_ddem, z_int)
    W_lcdm = lensing_kernel_W(chi_lcdm, chi_star_lcdm, z_int)

    def comb_fn(k_h):
        return ddem_comb_modifier(k_h, beta0=beta0, eps=eps, alpha=alpha)

    C_ddem = np.zeros_like(L_grid)
    C_lcdm = np.zeros_like(L_grid)
    for j, L in enumerate(L_grid):
        # k(z) = (L + 0.5) / chi(z)  in 1/Mpc, convert to h/Mpc
        k_ddem_h = (L + 0.5) / chi_ddem / H_PARAM
        k_lcdm_h = (L + 0.5) / chi_lcdm / H_PARAM
        # Mask k_h in valid range
        ok = (k_ddem_h > 1e-4) & (k_ddem_h < 10)
        # P(k, z) for DDEM (with comb) and LCDM (no comb)
        P_ddem = np.zeros_like(k_ddem_h)
        P_lcdm = np.zeros_like(k_lcdm_h)
        for i in range(n_z):
            if not ok[i]:
                continue
            P_ddem[i] = power_spectrum_z(np.array([k_ddem_h[i]]), z_int[i], bg_ddem,
                                          _eh_no_wiggle_P, comb_fn)[0]
            P_lcdm[i] = power_spectrum_z(np.array([k_lcdm_h[i]]), z_int[i], bg_lcdm,
                                          _eh_no_wiggle_P, None)[0]
        # Limber integrand: W^2 / (chi^2 H) * P
        # H in km/s/Mpc, chi in Mpc, P in (Mpc/h)^3 -> need consistent units
        # Standard convention: C_L^kk = int dz W^2(z) / (chi^2(z) H(z)/c) P(k=(L+0.5)/chi, z)
        # with W in 1/Mpc, chi in Mpc, P in Mpc^3 (NOT h^-3 Mpc^3)
        P_ddem_Mpc3 = P_ddem / H_PARAM**3
        P_lcdm_Mpc3 = P_lcdm / H_PARAM**3
        H_ddem_z = np.array([H_ddem(zi) for zi in z_int])
        H_lcdm_z = np.array([H_lcdm(zi) for zi in z_int])
        ig_ddem = W_ddem**2 / (chi_ddem**2 * H_ddem_z / C_KMS) * P_ddem_Mpc3
        ig_lcdm = W_lcdm**2 / (chi_lcdm**2 * H_lcdm_z / C_KMS) * P_lcdm_Mpc3
        C_ddem[j] = np.trapezoid(ig_ddem[ok], z_int[ok])
        C_lcdm[j] = np.trapezoid(ig_lcdm[ok], z_int[ok])

    residual = C_ddem / C_lcdm - 1.0
    return L_grid, residual, C_ddem, C_lcdm


def evaluate_against_act_dr6(L_grid, residual):
    """ACT DR6 lensing bandpowers fractional uncertainty is approximately
    0.5% (broad band) to 5% (high-L individual bins). SO/CMB-S4 forecast
    is ~0.3% at L ~ 200.

    Discard criterion: max |residual| < 0.0001 (0.01%) over L in [40, 1300].
    Keep criterion:    max |residual| > 0.001 (0.1%).
    """
    valid = np.isfinite(residual)
    if not valid.any():
        return "no valid points", 0.0
    amp_max = float(np.max(np.abs(residual[valid])))
    amp_rms = float(np.sqrt(np.mean(residual[valid]**2)))
    if amp_max < 1e-4:
        verdict = "DISCARD: residual < 0.01% (below ACT DR6 + SO precision)"
    elif amp_max > 1e-3:
        verdict = "KEEP: residual > 0.1% (forecastable signal)"
    else:
        verdict = "MARGINAL: 0.01% < residual < 0.1%"
    return verdict, amp_max, amp_rms


def main():
    print("DDEM CMB lensing C_L^{kk} comb prediction:")
    print("=" * 70)
    L, res, C_d, C_l = compute_clkk_residual(beta0=0.005, eps=0.2, alpha=0.5)
    print(f"  L grid:  {L[0]:.0f} to {L[-1]:.0f}, {L.size} bins")
    print(f"  residual max:  {np.max(np.abs(res)):.4e}")
    print(f"  residual rms:  {np.sqrt(np.mean(res**2)):.4e}")
    print()
    verdict, amp_max, amp_rms = evaluate_against_act_dr6(L, res)
    print(f"  Verdict: {verdict}")
    print(f"  amp_max = {amp_max:.4e}, amp_rms = {amp_rms:.4e}")
    print()
    print("Residual by L (every 5th point):")
    for i in range(0, len(L), 5):
        print(f"  L={L[i]:6.1f}  residual={res[i]:+.4e}  C_DDEM/C_LCDM={1+res[i]:.6f}")
    np.savez(HERE.parent / "cobaya" / "clkk_comb_residual.npz",
             L=L, residual=res, C_DDEM=C_d, C_LCDM=C_l,
             params={"beta0": 0.005, "eps": 0.2, "alpha": 0.5})
    print(f"\n  saved: cobaya/clkk_comb_residual.npz")
    return verdict


if __name__ == "__main__":
    main()
