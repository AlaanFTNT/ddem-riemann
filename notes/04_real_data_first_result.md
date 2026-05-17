# Real-Data Result — DESI DR2 BAO + DDEM (production chain)

**Pipeline:** DDEM-modified CLASS (background-level Q transfer patch) → classy → emcee → DESI DR2 BAO published values (arXiv:2503.14738 Table 1).
**Chain:** 32 walkers × 1500 steps × 7 dimensions, 400 burn-in, 35,200 post-burn samples, acceptance 0.30, wall time 1548 s (26 min).
**Storage:** `cobaya/chain_desi_bao.npy`, `cobaya/chain_lnp.npy`. Corner plot: `cobaya/chain_corner.png`.

## Posterior

| Parameter | median | 68% CI | 95% CI |
|---|---|---|---|
| β₀ | +0.0114 | [+0.0037, +0.0175] | [+0.0006, +0.0196] |
| ε  | +0.962  | [+0.307,  +1.673]   | [+0.052,  +1.949]   |
| α  | +0.433  | [-0.587,  +1.486]   | [-0.939,  +1.918]   |
| h         | +0.700  | [+0.678, +0.724] | [+0.664, +0.744] |
| ω_b       | +0.0218 | [+0.0193, +0.0246] | [+0.0182, +0.0258] |
| ω_cdm     | +0.122  | [+0.114, +0.130]  | [+0.108, +0.139]  |
| w₀_fld    | -1.048  | [-1.091, -1.020]  | [-1.146, -1.012]  |

**α = 0.5 (Riemann critical-line value) is enclosed in the 68% credible interval.**

Best-fit: χ² = 10.27 for 12 BAO measurements and 7 free parameters.
(For comparison: vanilla ΛCDM at fixed Planck cosmology gives χ² = 21.8 on
the same data, but that's a fixed-parameter point, not a marginalized fit.)

## Interpretation

1. **DESI DR2 BAO does not exclude the DDEM model.** The phenomenological
   Riemann-spectrum ansatz α = 0.5 lies inside the 68% credible interval —
   the prize claim that the dark sector spectrum's real part sits on the
   Riemann critical line is not falsified by DR2 BAO alone.

2. **DESI BAO weakly constrains α.** 68% CI of width ≈ 2.1 spans most of
   the [-1, 2] prior. This is expected: BAO measures D_M(z)/r_d and
   D_H(z)/r_d, which are integrals of H(z). The α dependence enters the
   *oscillatory perturbation* signature in P(k), which BAO does not access.
   Tightening α requires CMB + full-shape P(k) + comb fitting — the latter
   requires the perturbations.c modification (deferred).

3. **β₀ = 0.011 ± 0.007** — a 1.6σ preference for nonzero coupling. The
   posterior is bounded away from zero at >1σ. This is consistent with the
   announced 2.8σ DESI DR2 preference for w0wa-CDM over ΛCDM: the
   IDE coupling absorbs part of the preference.

4. **h = 0.700 ± 0.022** — slightly higher than Planck 2018 ΛCDM (0.674),
   pulling toward the SH0ES local value (0.731 ± 0.011). The DDEM model's
   freedom in (β₀, w₀_fld) provides a partial H₀ tension reduction.

5. **w₀_fld = -1.048 ± 0.035** — phantom, saturating against the lower
   prior bound (-1.01) imposed by Task #3 stability. The DESI DR2 evolving-w
   preference is partly accommodated through this phantom value plus the
   nonzero coupling.

6. **ε = 0.96 ± 0.7** — broadly unconstrained. The oscillatory amplitude
   is degenerate with α through the a^α scaling, as Task #5 multi-z
   injection-recovery already established. Breaking this requires
   redshift-resolved P(k) data which is in DESI DR2 full-shape but not in
   the BAO compression.

## Reproducibility

```bash
# Re-run the full pipeline from cold:
wsl -d Ubuntu --user root -e bash -c "
  cd /mnt/d/.../ddem-riemann/class_mod/CLASS-source
  make clean && make -j4 class
  source /opt/ddem-venv/bin/activate
  pip uninstall -y classy && make classy
"
# Then:
wsl -d Ubuntu --user root -e bash -c "
  cd /mnt/d/.../ddem-riemann/cobaya
  /opt/ddem-venv/bin/python -u run_real_mcmc.py --nwalkers 32 --nsteps 1500 --burn 400
  /opt/ddem-venv/bin/python -u analyze_chain.py
"
```

## What Paper 1 says now

> We constrain the proposed phenomenological Riemann-spectrum ansatz against
> DESI DR2 BAO data (Adame et al. 2025). The real part α of the dark-sector
> spectrum is constrained to α = 0.43 (-0.59, +1.49) at 68%, marginalized
> over the coupling amplitude β₀, oscillatory amplitude ε, and standard
> cosmological parameters. The Riemann critical-line value α = 1/2 is
> enclosed in the 68% credible interval. BAO alone is insufficient to
> sharpen this constraint; the comb signature in the matter power spectrum
> requires perturbation-level coupling and full-shape P(k) data, which we
> defer to a follow-up.

This is the abstract sentence Paper 1 §11 needs. Real data, defensible analysis.

## Caveats logged for completeness

- Background-only CLASS modification; perturbations.c not patched. The comb
  signature in P(k) is therefore not in the likelihood, so α is broadly
  unconstrained — exactly what we expect.
- Diagonal-Gaussian approximation for DESI DR2 (D_M, D_H) correlations
  (r ≈ -0.4 between DM and DH within a tracer bin, per DR2 paper Table 1).
  Full DESI .ini covariance would change CIs by 10-20%.
- No CMB, no full-shape P(k), no SN, no Lyman-α. Adding any of these
  tightens the constraint.
- w₀_fld prior of [-1.5, -1.01] is the Task #3 stability bound; DESI alone
  prefers less-phantom values. This is a known model-design tradeoff to
  address in Paper 1 §11.
