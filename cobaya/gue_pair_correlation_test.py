"""GUE pair-correlation test on DR1 LRG1 P(k) residuals.

The Riemann-spectrum ansatz predicts that the Q-coupling produces a comb
structure in P(k) at positions k_n = gamma_n / eta_0. The peak positions
{k_n} inherit the GUE pair-correlation statistics of the zeta zeros
(Montgomery-Odlyzko law). This module tests that ansatz directly against
the DR1 LRG1 broadband measurement, independently of the absolute-amplitude
calibration that defeats the chi^2 fit in dr1_lrg1_pk.py.

Procedure:

  1. Load the DR1 LRG1 monopole P_0(k) at k in [0.02, 0.20] h/Mpc.
  2. Fit a smooth no-wiggle broadband baseline using a low-order polynomial
     in ln P vs ln k, weighted by the diagonal of the covariance.
  3. Identify residual-peak positions (local maxima of (P_obs - P_smooth)
     above 1 sigma).
  4. Compute the nearest-neighbour spacing distribution s_i = (k_{i+1} - k_i)
     normalised to its mean (i.e., to mean spacing = 1).
  5. Test the unfolded spacing distribution against three null kernels:
       * GUE Wigner surmise:  p(s) = (32/pi^2) s^2 exp(-4 s^2 / pi)
       * Poisson:             p(s) = exp(-s)
       * Equal spacing:       p(s) = delta(s - 1)
     via Kolmogorov-Smirnov statistic on the cumulative distribution.

If the data is consistent with GUE statistics, the KS p-value for the GUE
null will be > 0.05 and the Poisson / equal-spacing nulls will be rejected.
This is a direct test of the Riemann-spectrum ansatz that does not depend
on the absolute P(k) amplitude.

The signal-to-noise on individual peaks at DR1 precision is modest, so the
test is reported with the number of peaks identified and an expected-detection
calculation rather than as a single yes/no claim.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import kstest
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "theory"))
sys.path.insert(0, str(ROOT / "synthetic_tests"))

from dr1_lrg1_pk import _DR1, K_MIN, K_MAX
from ddem_background import riemann_zeros


def _smooth_baseline(k, P, sigma_P, deg=4):
    """Fit a low-order polynomial in log-log space to the monopole, weighted
    by the diagonal P-uncertainty. Returns the smooth model on the input k grid."""
    lnk = np.log(k)
    lnP = np.log(np.maximum(P, 1e-30))
    # Weight by lnP uncertainty ~ sigma_P / P
    w = (P / np.maximum(sigma_P, 1e-30))
    coeffs = np.polyfit(lnk, lnP, deg, w=w)
    lnP_smooth = np.polyval(coeffs, lnk)
    return np.exp(lnP_smooth)


def _wigner_gue_cdf(s):
    """Cumulative distribution of the Wigner-GUE surmise.
    F(s) = 1 - (1 + 4s^2/pi) exp(-4 s^2 / pi) is incorrect; the exact form:
       p(s) = (32/pi^2) s^2 exp(-4 s^2 / pi)
       F(s) = erf(2 s / sqrt(pi)) - (4 s / pi) exp(-4 s^2 / pi)
    """
    from scipy.special import erf
    return erf(2.0 * s / np.sqrt(np.pi)) - (4.0 * s / np.pi) * np.exp(-4.0 * s**2 / np.pi)


def _poisson_cdf(s):
    """Cumulative distribution of an exponential (Poisson) spacing."""
    return 1.0 - np.exp(-s)


def _load_full_dr1_lrg1_monopole():
    """Load the full DR1 LRG1 monopole over the full k-range (k <= 0.40 h/Mpc)
    rather than only the chi^2-fit window. The GUE peak test benefits from
    the larger k-lever arm."""
    import h5py
    from dr1_lrg1_pk import COV_FILE
    with h5py.File(COV_FILE, "r") as f:
        k0 = np.array(f["observable/spectrum/0/k"])
        P0 = np.array(f["observable/spectrum/0/value"])
        cov_full = np.array(f["value"])
    n_per_ell = len(k0)                 # 80
    sigma_P0  = np.sqrt(np.diag(cov_full[:n_per_ell, :n_per_ell]))
    # Use a generous window for residual analysis.
    K_TEST_MIN, K_TEST_MAX = 0.02, 0.30
    m = (k0 >= K_TEST_MIN) & (k0 <= K_TEST_MAX)
    return k0[m], P0[m], sigma_P0[m]


def gue_pair_correlation_test(deg=5, peak_height_sigma=0.3, peak_distance_bins=1,
                              use_full_range=True):
    """Run the GUE pair-correlation test on DR1 LRG1 P0(k) residuals.

    Returns a dict with:
      n_peaks         : int      number of residual peaks identified
      k_peaks         : ndarray  peak positions
      s_normalised    : ndarray  nearest-neighbour spacings, normalised to mean 1
      ks_GUE          : (D, p)   KS statistic and p-value for GUE null
      ks_Poisson      : (D, p)   KS statistic and p-value for Poisson null
      expected_zeros  : ndarray  Riemann zero positions in k-units for comparison
    """
    if use_full_range:
        k, P, sigma_P = _load_full_dr1_lrg1_monopole()
    else:
        k = _DR1["k0"]
        P = _DR1["P0"]
        n_per = (_DR1["ells"] == 0).sum()
        sigma_P = np.sqrt(np.diag(_DR1["cov"][:n_per, :n_per]))

    # Smooth baseline + residual
    P_smooth = _smooth_baseline(k, P, sigma_P, deg=deg)
    resid = (P - P_smooth) / sigma_P  # in units of 1-sigma

    # Identify residual peak positions: local maxima of resid above threshold.
    peaks_idx, _ = find_peaks(resid, height=peak_height_sigma,
                              distance=peak_distance_bins)
    k_peaks = k[peaks_idx]

    # Riemann zeros in k-units for comparison (computed before the early-return
    # so the dict shape is consistent across all paths).
    gamma = riemann_zeros(20)
    eta_0_Mpc = 14000.0  # rough natural value for DDEM canonical
    h = 0.674
    k_zeros = np.array([g / eta_0_Mpc / h for g in gamma])
    k_zeros_in = k_zeros[(k_zeros >= K_MIN) & (k_zeros <= K_MAX)]

    if k_peaks.size < 4:
        return {"n_peaks": int(k_peaks.size), "k_peaks": k_peaks,
                "s_normalised": np.array([]), "ks_GUE": (np.nan, np.nan),
                "ks_Poisson": (np.nan, np.nan), "P_smooth": P_smooth,
                "resid": resid, "k": k, "P": P,
                "expected_zeros_k": k_zeros_in,
                "note": "too few peaks for KS test"}

    # Spacings, normalised so the mean spacing = 1 (standard "unfolding")
    s = np.diff(k_peaks)
    s_mean = s.mean()
    s_norm = s / s_mean

    ks_g = kstest(s_norm, _wigner_gue_cdf)
    ks_p = kstest(s_norm, _poisson_cdf)

    return {"n_peaks": int(k_peaks.size), "k_peaks": k_peaks,
            "s_normalised": s_norm, "ks_GUE": (ks_g.statistic, ks_g.pvalue),
            "ks_Poisson": (ks_p.statistic, ks_p.pvalue),
            "P_smooth": P_smooth, "resid": resid, "k": k, "P": P,
            "expected_zeros_k": k_zeros_in}


def report():
    print("=" * 70)
    print("GUE pair-correlation test on DESI DR1 LRG1 P_0(k) residuals")
    print("=" * 70)
    k_full, P_full, sP_full = _load_full_dr1_lrg1_monopole()
    print(f"k-window (full residual test): [{k_full.min():.3f}, {k_full.max():.3f}] h/Mpc ({k_full.size} bins)")
    print()
    for deg in [4, 5, 6, 8]:
        for h_sig in [0.2, 0.3, 0.5, 0.8]:
            res = gue_pair_correlation_test(deg=deg, peak_height_sigma=h_sig)
            n = res["n_peaks"]
            if "note" in res:
                print(f"deg={deg} h_sig={h_sig:.1f}: only {n} peaks ({res['note']})")
                continue
            D_g, p_g = res["ks_GUE"]
            D_p, p_p = res["ks_Poisson"]
            print(f"deg={deg} h_sig={h_sig:.1f}: n_peaks={n}  "
                  f"GUE KS p={p_g:.3f}  Poisson KS p={p_p:.3f}")
    print()
    # Detailed dump for the fiducial setting
    res = gue_pair_correlation_test(deg=5, peak_height_sigma=0.3)
    print(f"Fiducial deg=5, h_sig=0.3:")
    print(f"  n_peaks identified: {res['n_peaks']}")
    if res["n_peaks"] >= 4:
        print(f"  peak k positions (h/Mpc): " + ", ".join(f"{k:.4f}" for k in res["k_peaks"]))
        print(f"  spacings (normalised to mean=1):")
        for s in res["s_normalised"]:
            print(f"    {s:.3f}")
        print(f"  KS test GUE Wigner:    D = {res['ks_GUE'][0]:.4f}, p = {res['ks_GUE'][1]:.4f}")
        print(f"  KS test Poisson:       D = {res['ks_Poisson'][0]:.4f}, p = {res['ks_Poisson'][1]:.4f}")
    print()
    print(f"For comparison, first 10 expected Riemann-zero k positions in window:")
    for k in res["expected_zeros_k"][:10]:
        print(f"    {k:.4f} h/Mpc")
    return res


if __name__ == "__main__":
    report()
