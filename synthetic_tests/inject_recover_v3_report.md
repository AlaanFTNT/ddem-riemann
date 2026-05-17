# Injection-Recovery — Multi-Redshift (Task #5 final)

Pre-registered per Refined_Hypothesis.md §8. Data at z=0 AND z=1.

| Regime | alpha_inj | eps_inj | sigma | n_k | Recovered alpha (68% CI) | Test | Pass |
|---|---|---|---|---|---|---|---|
| multiz DR2-like alpha=0.5 | 0.5 | 0.5 | 1.5% | 80 | +0.410 [+0.183, +0.653] | A | PASS |
| multiz DR2-like alpha=0.3 | 0.3 | 0.5 | 1.5% | 80 | +0.213 [+0.002, +0.429] | B | FAIL |
| multiz DR3-like alpha=0.5 | 0.5 | 0.5 | 0.5% | 200 | +0.507 [+0.457, +0.556] | A | PASS |
| multiz boost alpha=0.5 | 0.5 | 1.5 | 0.5% | 200 | +0.502 [+0.485, +0.520] | A | PASS |
| multiz boost alpha=0.3 | 0.3 | 1.5 | 0.5% | 200 | +0.302 [+0.286, +0.318] | B | PASS |

Multi-redshift data breaks the (eps, alpha) degeneracy because the signal
amplitude scales as a_eff^alpha. At a=1.0 vs a=0.5, the ratio of comb
amplitudes is 2^alpha, which depends only on alpha.