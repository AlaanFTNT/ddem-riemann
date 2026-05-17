"""Compiled fsigma_8(z) measurements from RSD analyses.

Published growth-rate measurements from independent surveys. Values verified
against the original publications as of 2026-05-15.

Diagonal-Gaussian likelihood is used here. The BOSS DR12 consensus values
have non-zero inter-bin correlations (rho ~ 0.17, 0.20, 0.06 from
Alam et al. 2017 Table 8) which are not included; the effective uncertainty
on the BOSS-only contribution is therefore slightly underestimated.
"""
import numpy as np

# (label, z_eff, fsigma8, sigma_fsigma8, reference)
FSIGMA8_DATA = [
    ("6dFGRS",           0.067, 0.423, 0.055, "Beutler et al. 2012, arXiv:1204.4725"),
    ("BOSS DR12 LOWZ",   0.38,  0.497, 0.045, "Alam et al. 2017, arXiv:1607.03155 Table 7 (stat+sys)"),
    ("BOSS DR12 CMASS",  0.51,  0.458, 0.038, "Alam et al. 2017, arXiv:1607.03155 Table 7 (stat+sys)"),
    ("BOSS DR12 high-z", 0.61,  0.436, 0.034, "Alam et al. 2017, arXiv:1607.03155 Table 7 (stat+sys)"),
    ("eBOSS LRG",        0.698, 0.473, 0.044, "Bautista et al. 2021, arXiv:2007.08993"),
    ("eBOSS ELG",        0.85,  0.315, 0.095, "Tamone et al. 2020, arXiv:2007.09009 (consensus)"),
    ("eBOSS QSO",        1.48,  0.464, 0.045, "Hou et al. 2021, arXiv:2007.08998 (consensus)"),
    # DESI DR1 full-shape, ShapeFit-only; DESI DR2 has not yet released RSD.
    ("DESI DR1 BGS",     0.295, 0.3772, 0.0941, "DESI Collaboration 2024, arXiv:2411.12021 App. A"),
    ("DESI DR1 LRG1",    0.510, 0.5136, 0.0643, "DESI Collaboration 2024, arXiv:2411.12021 App. A"),
    ("DESI DR1 LRG2",    0.706, 0.4836, 0.0530, "DESI Collaboration 2024, arXiv:2411.12021 App. A"),
    ("DESI DR1 LRG3",    0.919, 0.4222, 0.0473, "DESI Collaboration 2024, arXiv:2411.12021 App. A"),
    ("DESI DR1 ELG2",    1.317, 0.3767, 0.0374, "DESI Collaboration 2024, arXiv:2411.12021 App. A"),
    ("DESI DR1 QSO",     1.491, 0.4349, 0.0445, "DESI Collaboration 2024, arXiv:2411.12021 App. A"),
]

Z_FSIGMA8     = np.array([d[1] for d in FSIGMA8_DATA])
FSIGMA8_VALS  = np.array([d[2] for d in FSIGMA8_DATA])
FSIGMA8_SIGS  = np.array([d[3] for d in FSIGMA8_DATA])


def chi2_fsigma8(predict_fsigma8):
    """Compute the compiled-RSD chi^2 against the published measurements.

    predict_fsigma8 : callable
        Function that takes a 1D array of z and returns fsigma_8(z).
    """
    pred = predict_fsigma8(Z_FSIGMA8)
    resid = FSIGMA8_VALS - pred
    return float(np.sum((resid / FSIGMA8_SIGS)**2))


if __name__ == "__main__":
    print(f"Compiled fsigma_8 measurements: {len(FSIGMA8_DATA)} points")
    for d in FSIGMA8_DATA:
        print(f"  z={d[1]:.3f}  f*sigma8={d[2]:.4f} +/- {d[3]:.4f}   [{d[0]}]")
