"""ACT DR6 CMB lensing reconstruction constraint on the modified growth.

Source: Madhavacheril et al. 2024, "The Atacama Cosmology Telescope: DR6
gravitational lensing map and cosmological parameters" (arXiv:2304.05203,
ApJ 962, 113). Companion power-spectrum paper: Qu et al. 2024 (arXiv:2304.05202).

The lensing-only headline constraint reported in the abstract is

    S_8^{CMBL} = sigma_8 (Omega_m / 0.3)^{0.25} = 0.818 +- 0.022

with the alpha = 0.25 exponent set by the eigendirection of the DR6 lensing
likelihood. The kernel peaks near z ~ 2 and is sensitive over z ~ 0.5-5,
i.e., a higher-redshift complement to galaxy RSD fsigma_8.

The DR6-alone value (rather than DR6 + Planck NPIPE lensing or DR6 + BAO)
is used here because it is independent of the Planck marginal prior and the
BAO likelihood already in the joint pipeline.
"""
import numpy as np

S8_CMBL_VAL    = 0.818
S8_CMBL_SIG    = 0.022
S8_CMBL_ALPHA  = 0.25


def chi2_act_lensing(sigma8_z0, Omega_m):
    """Predict S_8^{CMBL} from (sigma_8(z=0), Omega_m) and return chi^2.

    S_8^{CMBL} = sigma_8 * (Omega_m / 0.3)^0.25
    chi^2      = (predicted - measured)^2 / sigma^2
    """
    pred = sigma8_z0 * (Omega_m / 0.3)**S8_CMBL_ALPHA
    return ((pred - S8_CMBL_VAL) / S8_CMBL_SIG)**2


if __name__ == "__main__":
    print(f"ACT DR6 CMB lensing alone: S_8^CMBL = {S8_CMBL_VAL} +- {S8_CMBL_SIG}")
    print(f"  alpha exponent: {S8_CMBL_ALPHA}")
    print()
    for s8, Om in [(0.811, 0.315), (0.820, 0.305), (0.800, 0.320)]:
        pred = s8 * (Om/0.3)**0.25
        chi2 = chi2_act_lensing(s8, Om)
        print(f"  sigma_8={s8}, Omega_m={Om}:  S_8^CMBL pred = {pred:.4f}, chi^2 = {chi2:.3f}")
