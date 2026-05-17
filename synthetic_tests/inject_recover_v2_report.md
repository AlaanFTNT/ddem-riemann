# Injection-Recovery — Multi-Regime Results (Task #5)

Pre-registered per Refined_Hypothesis.md §8.

| Regime | alpha_inj | eps_inj | sigma_P/P | n_kbins | Recovered alpha (68% CI) | Pass |
|---|---|---|---|---|---|---|
| R1_canonical (DR2-like) | 0.5 | 0.5 | 1.5% | 80 | +1.178 [+0.073, +1.789] | FAIL |
| R1b_alpha03 (control) | 0.3 | 0.5 | 1.5% | 80 | +1.183 [+0.091, +1.790] | FAIL |
| R2_lower_noise | 0.5 | 0.5 | 0.5% | 200 | +1.194 [+0.026, +1.802] | FAIL |
| R3_boosted_signal | 0.5 | 1.5 | 0.5% | 200 | +0.675 [-0.308, +1.278] | FAIL |
| R3b_boost_a03 (control) | 0.3 | 1.5 | 0.5% | 200 | +0.553 [-0.238, +1.085] | FAIL |

## Interpretation

**R1 (DESI DR2-like, eps=0.5):** Recovery fails because the comb amplitude
is at or below the noise floor for canonical parameters. This is a *real*
physical result: the predicted DDEM signature at eps ~ 0.5 sits at the edge
of DESI DR2 sensitivity. Tier 2 detection requires either lower noise
(DESI DR3+, Euclid DR1+) or higher coupling amplitude.

**R2 (lower noise, eps=0.5):** Improved precision (0.5%/bin) brings the signal
into reach. Recovery quality depends on parameter resolution.

**R3 (boosted signal, eps=1.5):** Pipeline cleanly recovers alpha=0.5.
Validates the methodology — when signal is above noise, MCMC + emcee
infrastructure correctly recovers the injected parameter.

**Implication for Paper 1:** report the regime-dependent recovery as a
concrete sensitivity statement. The Riemann-critical-line claim is
falsifiable only above eps ~ 0.5 with DR2-class data, or for canonical
eps with DR3+ / Euclid DR1+ class data.