"""CMB-lensing constraint on modified growth.

Primary source: Qu et al. 2025, "Unified and consistent structure growth
measurements from joint ACT, SPT and Planck CMB lensing" (Phys. Rev. Lett. 136,
021001; arXiv:2504.20038). Joint Planck PR4 + ACT DR6 + SPT-3G CMB lensing
reconstruction, lensing-only (no primary TT/TE/EE), with only weak BBN /
n_s priors.

Headline lensing-only constraint:

    S_8^{CMBL} = sigma_8 (Omega_m / 0.3)^{0.25} = 0.825 +0.015 / -0.013

The lensing kernel peaks near z ~ 2, complementing galaxy RSD f sigma_8 at
late time. Same alpha = 0.25 eigendirection as Madhavacheril et al. 2024
(ACT DR6 alone), so this is a drop-in replacement that tightens the lensing
constraint by ~1.6x. The Gaussian symmetrisation uses the mean of the
asymmetric 1-sigma bounds.

Because Qu et al. 2025 uses Planck PR4 lensing maps, run_joint_mcmc.py
switches the Planck 2018 marginals from the "TT,TE,EE+lowE+lensing" column
to the "TT,TE,EE+lowE" column of Planck 2018 Table 2, to avoid double-counting
the Planck lensing piece.
"""
import numpy as np

S8_CMBL_VAL    = 0.825
S8_CMBL_SIG    = 0.014           # symmetric approx of +0.015/-0.013
S8_CMBL_ALPHA  = 0.25


def chi2_cmb_lensing(sigma8_z0, Omega_m):
    """Predict S_8^{CMBL} = sigma_8 (Omega_m/0.3)^0.25 and return chi^2."""
    pred = sigma8_z0 * (Omega_m / 0.3)**S8_CMBL_ALPHA
    return ((pred - S8_CMBL_VAL) / S8_CMBL_SIG)**2


# Backward-compatible alias for the existing run_joint_mcmc.py import path.
chi2_act_lensing = chi2_cmb_lensing


if __name__ == "__main__":
    print(f"Qu+ 2025 joint Planck PR4 + ACT DR6 + SPT-3G CMB lensing:")
    print(f"  S_8^CMBL = {S8_CMBL_VAL} +- {S8_CMBL_SIG}  (alpha = {S8_CMBL_ALPHA})")
    print()
    for s8, Om in [(0.811, 0.315), (0.820, 0.305), (0.825, 0.300), (0.800, 0.320)]:
        pred = s8 * (Om/0.3)**0.25
        chi2 = chi2_cmb_lensing(s8, Om)
        print(f"  sigma_8={s8}, Omega_m={Om}:  pred = {pred:.4f}, chi^2 = {chi2:.3f}")
