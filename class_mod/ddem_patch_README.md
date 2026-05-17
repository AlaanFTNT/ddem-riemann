# DDEM Patch for CLASS — Implementation Plan

This directory holds the modifications to CLASS (Blas, Lesgourgues, Tram 2011)
that wire the DDEM oscillatory Q kernel into the background and perturbation
modules.

## Status

- [x] CLASS upstream cloned at `CLASS-source/` (commit at clone time).
- [x] Vanilla build verified: `class explanatory.ini` runs and produces sane LCDM output.
- [x] `classy` Python wrapper built and importable from `/opt/ddem-venv` in WSL Ubuntu.
- [ ] **Q kernel added to `source/background.c`** — see `patches/01_background_Q_kernel.diff` (stub).
- [ ] **Coupling to `source/perturbations.c`** — `patches/02_perturbations_delta_Q.diff` (stub).
- [ ] **Input parameters added to `source/input.c`** — `patches/03_input_DDEM_params.diff` (stub).
- [ ] Recompile classy wrapper, rerun sanity tests with DDEM disabled (`eps=0`) and verify identity to LCDM.

## Why the Python emulator first

The Python emulator under `theory/` (ddem_background.py, ddem_perturbation.py)
gives us:
1. A working end-to-end pipeline for synthetic injection-recovery (Task #5, done).
2. A concrete falsification prediction (predict_signature.py output, Task #7, done).
3. A differentiation comparison vs competing models (Task #8, done).
4. A validated MCMC infrastructure (Task #5 multi-z showed the methodology works).

These deliverables form Paper 1 §3–§7 content. The full CLASS C-source
modification is required for the paper's final MCMC run against real CMB+LSS
data (Paper 1 §8 result table). That work is multi-day: it requires careful
modification of the Boltzmann hierarchy, validation of energy conservation,
high-k stability tests, and recompile/test cycles.

## Where the modification goes in CLASS source

### `source/background.c`

Locate the function `background_derivs`. This is the RHS of the background ODE
system. Currently it integrates the standard FLRW continuity equations for each
component. Add:

```c
/* DDEM oscillatory matter-DE coupling */
double Q_ddem = 0.0;
if (pba->ddem_enabled == _TRUE_) {
    Q_ddem = ddem_Q_kernel(a, pba->ddem_beta0, pba->ddem_eps,
                           pba->ddem_alpha, pba->Omega0_lambda, H);
    /* Add to matter continuity */
    dy[pba->index_bg_rho_cdm] += Q_ddem / H;
    /* Subtract from DE continuity (energy conservation) */
    dy[pba->index_bg_rho_lambda] -= Q_ddem / H;
}
```

`ddem_Q_kernel(...)` is the helper defined in `source/ddem.c` (new file). It
implements

```c
Q(a) = beta0 * H0 * rho_DE * [1 + 2*eps * sum_n a^alpha cos(gamma_n ln a) / |rho_n|]
```

with a hard-coded table of the first N = 200 imaginary parts of the non-trivial
zeta zeros (provided in `source/ddem_zeros.h` — pre-tabulated to 15 digits via
mpmath).

### `source/perturbations.c`

In `perturb_derivs`, add the perturbed coupling:

```c
if (pba->ddem_enabled == _TRUE_) {
    /* delta_Q from variation of Q wrt rho_DE, with bracket-perturbation absorbed
       into the bracket evaluated at the same a (no spatial variation of bracket
       at this order — geometric/temporal coupling only). */
    double bracket = ddem_bracket(a, pba->ddem_eps, pba->ddem_alpha);
    double delta_Q = pba->ddem_beta0 * pba->H0 * y[pv->index_pt_delta_de] * bracket;
    /* Source into the matter delta and theta equations */
    dy[pv->index_pt_delta_cdm] += delta_Q / (H * y[pv->index_pt_rho_cdm_background]);
    dy[pv->index_pt_delta_de]  -= delta_Q / (H * y[pv->index_pt_rho_de_background]);
    /* theta_cdm receives the velocity-divergence source per VMM 2008 eq. (something) */
    dy[pv->index_pt_theta_cdm] += -ppr->ddem_drag * Q_ddem * y[pv->index_pt_theta_cdm] / rho_cdm;
}
```

The `ddem_drag` coefficient is the new effective drag from the coupling. For
the safe regime (phantom w_DE), it stays negative (decelerating).

### `source/input.c`

Add four new parameters under `&pba->`:

```c
class_read_double("ddem_beta0", pba->ddem_beta0);
class_read_double("ddem_eps",   pba->ddem_eps);
class_read_double("ddem_alpha", pba->ddem_alpha);
class_read_int(   "ddem_N",     pba->ddem_N);    /* truncation; default 100 */
class_call(input_string_to_int("ddem_enabled", input, &pba->ddem_enabled),
           errmsg, errmsg);
```

And the matching default in `input_read_parameters_background`.

## Validation steps after recompile

1. **Identity test:** `eps = 0` should give bit-identical output to vanilla CLASS
   (within floating-point tolerance). Compare P(k, z) and Cl from both runs.
2. **Background closure:** verify `H^2(a) - (rho_m + rho_r + rho_DE)(a)` stays
   at machine precision across all a.
3. **High-k stability:** inject perturbations at k = 1 h/Mpc and confirm no
   unphysical blowup at recombination epoch (this is the §5 decision-tree
   numerical stability check).
4. **Comparison to Python emulator:** for matched parameters, CLASS-derived
   P_DDEM(k)/P_LCDM(k) should agree with the Python emulator within ~few percent
   in the k > 0.01 h/Mpc band where the emulator is reliable. Discrepancies
   below k = 0.005 are expected and indicate the emulator's limitations.

## Next-engineer hand-off

The Python emulator under `theory/` is the trusted reference for what the C
patch should produce. After implementing the patches above, re-run
`synthetic_tests/inject_recover_v3_multiz.py` with the modified classy in place
of `Eisenstein_Hu_pk` — recovery quality should match or exceed the emulator
results.

The four-stage validation above (identity, closure, stability, emulator match)
is the gate before any real-data MCMC. Skipping it risks wasting cluster hours
on a buggy implementation.
