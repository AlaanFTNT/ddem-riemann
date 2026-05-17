"""DESI DR2 BAO data, from arXiv:2503.14738 Table 1 (BAO-only consensus).

For each tracer bin we store the effective redshift, the measurements
(D_M/r_d and/or D_H/r_d and/or D_V/r_d), their 1-sigma uncertainties, and the
cross-correlation coefficient between D_M and D_H within the same bin.

Reference: DESI Collaboration 2025, arXiv:2503.14738. Diagonal-Gaussian
approximation used here (full covariance with off-diagonal entries would
require the published .ini files from DESI; using r ≈ -0.4 between D_M and
D_H within a bin per the DR2 paper Table 1)."""
import numpy as np

# Each entry: (z_eff, [(obs_type, value, sigma)...], r_DM_DH)
# obs_type: "DM_over_rd", "DH_over_rd", "DV_over_rd"
DESI_DR2_BAO = [
    {"name": "BGS",       "z": 0.295,
     "meas": [("DV_over_rd", 7.944, 0.075)],
     "r_DM_DH": 0.0},
    {"name": "LRG1",      "z": 0.510,
     "meas": [("DM_over_rd", 13.587, 0.169),
              ("DH_over_rd", 21.863, 0.425)],
     "r_DM_DH": -0.45},
    {"name": "LRG2",      "z": 0.706,
     "meas": [("DM_over_rd", 17.351, 0.177),
              ("DH_over_rd", 19.455, 0.330)],
     "r_DM_DH": -0.42},
    {"name": "LRG3+ELG1", "z": 0.934,
     "meas": [("DM_over_rd", 21.576, 0.152),
              ("DH_over_rd", 17.641, 0.193)],
     "r_DM_DH": -0.39},
    {"name": "ELG2",      "z": 1.321,
     "meas": [("DM_over_rd", 27.601, 0.318),
              ("DH_over_rd", 14.176, 0.221)],
     "r_DM_DH": -0.44},
    {"name": "QSO",       "z": 1.484,
     "meas": [("DV_over_rd", 26.067, 0.659)],
     "r_DM_DH": 0.0},
    {"name": "Lya QSO",   "z": 2.330,
     "meas": [("DM_over_rd", 38.988, 0.531),
              ("DH_over_rd", 8.632, 0.101)],
     "r_DM_DH": -0.48},
]


def chi2_desi_bao(predict_DM_over_rd, predict_DH_over_rd, predict_DV_over_rd):
    """Compute the DESI DR2 BAO chi^2 given prediction functions of z."""
    chi2 = 0.0
    for bin_ in DESI_DR2_BAO:
        z = bin_["z"]
        types = [m[0] for m in bin_["meas"]]
        vals = [m[1] for m in bin_["meas"]]
        sigs = [m[2] for m in bin_["meas"]]
        if "DV_over_rd" in types:
            i = types.index("DV_over_rd")
            pred = predict_DV_over_rd(z)
            chi2 += ((vals[i] - pred) / sigs[i])**2
        else:
            # DM_over_rd and DH_over_rd jointly; use correlation
            iM = types.index("DM_over_rd")
            iH = types.index("DH_over_rd")
            r = bin_["r_DM_DH"]
            d = np.array([vals[iM] - predict_DM_over_rd(z),
                          vals[iH] - predict_DH_over_rd(z)])
            cov = np.array([[sigs[iM]**2, r * sigs[iM] * sigs[iH]],
                            [r * sigs[iM] * sigs[iH], sigs[iH]**2]])
            inv = np.linalg.inv(cov)
            chi2 += float(d @ inv @ d)
    return chi2


if __name__ == "__main__":
    # Sanity: print the data
    for b in DESI_DR2_BAO:
        print(f"{b['name']:12s}  z={b['z']:.3f}", end="  ")
        for m in b["meas"]:
            print(f"{m[0]}={m[1]:.3f}+/-{m[2]:.3f}", end="  ")
        print()
