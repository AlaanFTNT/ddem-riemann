# Synthetic Injection-Recovery Results (Task #5)

Pre-registered per Refined_Hypothesis.md §8.
Data: 80 k-bins log-spaced from 0.005 to 0.3 h/Mpc, sigma_P/P = 1.5% per bin.
Priors: eps in [0, 2], alpha in [-1, 2].
MCMC: emcee, 48 walkers, 2000 steps, 500 burn.

## Experiment A: inject alpha = 0.5 (Riemann critical-line)

- **eps**: median = +0.7447; 68% CI = [+0.3548, +1.1671]; 95% CI = [+0.1963, +1.4603]
- **alpha**: median = +1.1554; 68% CI = [+0.0877, +1.7808]; 95% CI = [-0.7718, +1.9673]

**Pass criterion (§8):** 68% CI of alpha encloses 0.5 AND excludes 0 and 1.
68% CI of alpha: [+0.0877, +1.7808]  ->  **FAIL**

## Experiment B: inject alpha = 0.3 (non-Riemann control)

- **eps**: median = +0.7396; 68% CI = [+0.3585, +1.1439]; 95% CI = [+0.1990, +1.4148]
- **alpha**: median = +1.1870; 68% CI = [+0.1390, +1.7837]; 95% CI = [-0.7112, +1.9665]

**Pass criterion (§8):** 95% CI of alpha EXCLUDES 0.5 (no spurious recovery).
95% CI of alpha: [-0.7112, +1.9665]  ->  **FAIL**

## Overall: AT LEAST ONE FAILED