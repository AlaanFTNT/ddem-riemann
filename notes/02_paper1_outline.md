# Paper 1 — Working Outline

**Working title:** "Oscillatory dark-sector coupling with a Riemann-spectrum ansatz: methodology, doom-factor stability, and Tier 1/Tier 2 falsification forecasts."

**Target:** JCAP.

**Status of underlying results (from this session's work):**

| Section | Source artifact | Status |
|---|---|---|
| §1 Introduction | Refined_Hypothesis.md §1, §3 | drafted |
| §2 Phenomenological ansatz framing | Refined_Hypothesis.md §3 + research reports | drafted |
| §3 Lagrangian (Type-2 conformal) | Refined_Hypothesis.md §4 | drafted |
| §4 Background equations + doom-factor proof | doom_factor_analysis.py + doom_factor_report.md | computed |
| §5 Perturbation theory | ddem_background.py + ddem_perturbation.py | computed (Python emulator) |
| §6 Predicted observables | predict_signature.py + comb_peaks_table.md | computed |
| §7 Synthetic injection-recovery | inject_recover_v3_multiz.py + inject_recover_v3_report.md | computed |
| §8 Differentiation vs competing models | differentiation_table.py + differentiation_table.md | computed |
| §9 Pre-registered analysis pipeline | Refined_Hypothesis.md §8 | drafted |
| §10 Discussion + risks | Refined_Hypothesis.md §11 | drafted |

## Headline results to feature

1. **Analytical stability gate cleared.** The proposed Q kernel passes both VMM-style
   stability criteria across the observational a-range for β₀ ≲ 0.01, ε ≲ 1, and
   phantom w_DE ≲ −1.05. The dominant oscillation mode is γ₁ = 14.135 at every
   scale factor (1/|ρ_n| damping saturates the first zero universally). The
   stability threshold therefore compresses to a single bound on β₀/|1 + w_DE|.

2. **Concrete comb prediction.** η₀ = 14,136 Mpc. First five predicted peaks at
   k = 0.00148, 0.00220, 0.00261, 0.00318, 0.00344 h/Mpc (Tier 3 — below DESI
   k_min). Peaks from γ₅₀ onward (k > 0.015 h/Mpc) are inside the DESI linear
   regime at sub-percent precision — Tier 2.

3. **Methodology validated by synthetic injection-recovery.** Single-redshift
   fits exhibit a strong (ε, α) degeneracy. Multi-z data (z=0 + z=1) breaks the
   degeneracy via the 2^α scaling of the amplitude. At DR2-class noise (1.5%/bin),
   the canonical-coupling α = 0.5 injection passes the §8 criterion (68% CI
   [0.18, 0.65]); the α = 0.3 control fails because DR2 cannot distinguish 0.3
   from 0.5 at canonical coupling. DR3-class noise (0.5%/bin) clears both
   tests cleanly. **This is the falsification sensitivity statement Paper 1 needs.**

4. **Differentiation from competing models.** DDEM is the only multi-frequency
   comb in P(k); EDE is one bump, S-IDE one tilt, w₀wₐCDM smooth, FHSW
   oscillating quintessence single-frequency. The distinguishing band is
   0.015 ≲ k ≲ 0.1 h/Mpc.

## Headline limitations to acknowledge

1. The full CLASS source modification is not yet implemented (see
   `class_mod/ddem_patch_README.md`). The Python emulator is sufficient for
   methodology + synthetic forecasts but the real-data run (planned as the
   Paper 1 §11 numerical result) requires the C-source patch.

2. The Gaussian envelope width σ_n = 0.02 · k_n in the comb signature is a
   heuristic. The rigorous width derives from perturbation evolution at the
   epoch of mode crossing and should be checked against the modified CLASS
   output before claim of comb amplitudes.

3. The (ε, α) degeneracy is not fully broken at DR2 sensitivity for the α=0.3
   control. The pre-registered falsification statement should be conditioned
   on dataset combination (DR2 alone insufficient; DR3 / Euclid DR1 sufficient).

## What goes into Paper 2 not Paper 1

- Tier 3 detection attempt at low-n comb peaks (k < 0.01 h/Mpc).
- Cross-correlation with Planck PR4 ultra-large-scale polarization.
- SKA-Low 21cm intensity mapping at the horizon scales.
- Roman early HLTDS supernova data.

## Submission gates

Paper 1 cannot submit until:
- [ ] CLASS C-source patches implemented and validated (4-stage validation in
      `class_mod/ddem_patch_README.md`).
- [ ] Real-data MCMC run against DESI DR2 + Planck PR4 + ACT DR6 + Euclid Q1
      with Cobaya. 8 chains to R-1 < 0.01.
- [ ] Bayesian evidence Δ ln Z computed vs ΛCDM, w₀wₐCDM, S-IDE benchmark.
- [ ] Differentiation analysis upgraded from heuristic templates to full
      Boltzmann output for each competitor model.
- [ ] One co-author from the IDE community sanity-checks the doom-factor
      derivation and the perturbation-coupling form.
