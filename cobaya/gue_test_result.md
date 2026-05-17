# GUE pair-correlation test on DR1 LRG1 — result

## Methodology
The test compares the nearest-neighbour spacing distribution of residual-peak positions in DR1 LRG1 P_0(k) to three null kernels: the GUE Wigner surmise, Poisson, and equal spacing. Peaks are identified as local maxima of (P_obs − P_smooth) / σ_P above a threshold, with P_smooth a low-order polynomial in (ln P, ln k).

## Result at DR1 precision
At deg=8, h_sig=0.3σ over k ∈ [0.02, 0.30] h Mpc⁻¹ (56 k-bins):
- 4 residual peaks identified at k = 0.038, 0.078, 0.138, 0.203 h Mpc⁻¹
- 3 normalised spacings: 0.79, 1.06, 1.15
- KS test vs GUE Wigner:  D = 0.34, p = 0.85 (cannot reject)
- KS test vs Poisson:     D = 0.55, p = 0.30 (cannot reject)

## Diagnostic — peak interpretation
The identified peak spacings (0.040, 0.060, 0.065 h Mpc⁻¹) match the BAO fundamental wavelength 2π / (r_d · h) ≈ 0.063 h Mpc⁻¹. The peaks are BAO-comb peaks, not DDEM-comb peaks.

## Resolution limit
DDEM canonical η₀ from the background solver = 13 962 Mpc. The Riemann-zero comb positions k_n = γ_n / η₀ have mean inter-peak spacing 0.00017 h Mpc⁻¹ across the [0.02, 0.30] h Mpc⁻¹ window (227 zeros in window). The DR1 LRG1 k-bin width is 0.005 h Mpc⁻¹ ≈ 30× the Riemann-zero spacing. The Riemann comb is therefore unresolved in linear k at DR1 resolution; only the integrated log-frequency signal at ω_log = γ_n is observable, which is exactly what the Beutler et al. 2023 per-mode bound (already used as the 4th likelihood in §5.3) tests.

## Implication for the paper
The GUE pair-correlation test in linear-k requires resolution ≲ 0.001 h Mpc⁻¹ to discriminate the Riemann-comb spacing from a Poisson or uniform null. DESI DR3 and Euclid DR1 with their factor-three lower per-mode noise and finer k-binning are the data needed to make this test decisive. The Beutler-style log-frequency test is the resolution-appropriate test at DR1 precision and is already included in the joint fit.
