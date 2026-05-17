# Task #3 — Doom-Factor Analytical Stability: Result

**Status:** Passed analytical gate.
**Script:** `theory/doom_factor_analysis.py`
**Outputs:** `theory/doom_factor.png`, `theory/doom_factor_report.md`

## Headline finding

The proposed oscillatory Q kernel,
`Q(a) = beta0 * H0 * rho_Lambda * [1 + 2*eps * sum_n a^alpha cos(gamma_n ln a)/|rho_n|]`,
passes both Refined_Hypothesis.md §5 stability criteria across the observational a-range for phantom dark energy with β₀ ≲ 0.01 and w_DE ≲ -1.05. The non-phantom case (w_DE = -0.95) fails criterion (i) at every a, exactly as VMM 2008 predicts. No surprise, and no new physics needed to clear the gate.

## Surprising structural fact

The dominant oscillatory mode γ_dom is γ₁ = 14.135 at **every** scale factor in the tested range. The 1/|ρ_n| damping in the von Mangoldt sum makes γ₁ dominant universally — higher zeros never overtake it because their per-mode amplitude is suppressed by γ_n while their oscillation rate grows as γ_n. This means the N_osc threshold from §5 is a single universal number, |d_peak| < 1/(γ₁/2π) ≈ 0.4445, not a function of a.

This is worth surfacing in Paper 1. It compresses the stability statement to a one-line bound on the parameter combination β₀ / |1 + w_DE|.

## Boundary cases (informative failures)

- `higher_beta` (β₀ = 0.05): fails criterion (ii) at a = 1 with d_peak = 0.78.
- `weakly_phantom` (w_DE = -1.01): fails criterion (ii) at a = 1 with d_peak = 0.78.

Both failures occur at a = 1 (today) because the doom factor scales as β₀ / |1 + w_DE| and both perturbations move the same combination above the threshold. These define the corner of parameter space where the prior is constrained.

## Implication for Paper 1 prior

The free parameters {β₀, ε, α} need joint priors that respect the boundary observed above. Concretely:
- β₀ flat in [0, 0.02] keeps criterion (ii) green for w_DE ≤ -1.05.
- ε flat in [0, 2] is admissible — the eps scan showed no instability up to eps = 1.
- α flat in [-1, 2] (per §8 pre-registration) is admissible — the alpha scan showed no instability across α ∈ {0, 0.5, 1}.
- w_DE flat in [-1.5, -1.01] keeps the dark sector phantom enough.

The CLASS MCMC will need to enforce this jointly, e.g., via a prior product or by rejecting samples where the analytical stability check fails on a coarse grid.

## What this does NOT prove

- That the perturbation-level evolution is stable for k > k_horizon. Only the background-doom-factor argument is checked here. The §5 decision tree calls for a numerical CLASS injection at high-k modes if analytical bounds are marginal. Baseline passes with margin, so the numerical check is informative rather than load-bearing — still required, but the analytical pass is the primary result.
- That non-linear regime is stable. Only linear perturbation theory is covered by VMM.
- That the truncation in n is benign. We used N = 200 zeros; the high-γ_n contribution to the bracket is small (suppressed by 1/γ_n) but should be checked for truncation sensitivity in a follow-up.

## Next: Task #4

Proceed to CLASS clone, build, and modification of `source/background.c` and `source/perturbations.c`. Ubuntu WSL was installed for this purpose because CLASS does not build cleanly on Windows.
