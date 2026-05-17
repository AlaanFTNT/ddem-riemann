# DDEM Build Session — Summary

## What was built

```
ddem-riemann/
├── .venv/                              # Python 3.12 venv (Windows)
├── theory/
│   ├── doom_factor_analysis.py         # §5 analytical stability (Task #3)
│   ├── doom_factor_report.md           # Pass/fail table per parameter set
│   ├── doom_factor.png                 # Plot of d(a) for 4 parameter sets
│   ├── ddem_background.py              # Modified background ODE solver
│   ├── ddem_perturbation.py            # Linear growth + comb signature
│   ├── differentiation_table.py        # DDEM vs EDE/S-IDE/w0waCDM/FHSW
│   ├── differentiation_residuals.png   # Visual comparison plot
│   └── differentiation_table.md        # Comparison table
├── class_mod/
│   ├── CLASS-source/                   # Vanilla CLASS, built + classy installed
│   └── ddem_patch_README.md            # C-source modification hand-off doc
├── synthetic_tests/
│   ├── predict_signature.py            # Falsifiable comb prediction (Task #7)
│   ├── predict_signature.png           # P(k) residual + peak locations
│   ├── comb_peaks_table.md             # k_n table for n in [1, 200]
│   ├── inject_recover.py               # v1 single-z (initial attempt)
│   ├── inject_recover_v2.py            # Multi-regime S/N exploration
│   ├── inject_recover_v3_multiz.py     # Two-redshift fit (validated)
│   ├── inject_recover_v3_*.png         # Corner plots, 5 regimes
│   └── inject_recover_v3_report.md     # Final pass/fail summary
└── notes/
    ├── 01_doom_factor_result.md
    ├── 02_paper1_outline.md
    └── 03_session_summary.md           # This file
```

## Task status

| Task | Status | Artifact |
|---|---|---|
| #1 Refined hypothesis | DONE | `Refined_Hypothesis.md` |
| #2 Mechanism commitment | DONE | Type-2 conformal, GUE ansatz (in §1, §4) |
| #3 Doom-factor analytical | DONE | `theory/doom_factor_*` |
| #4 CLASS C-source mod | PARTIAL | `class_mod/ddem_patch_README.md` scaffolding; Python emulator complete |
| #5 Synthetic injection-recovery | DONE | `synthetic_tests/inject_recover_v3_*` |
| #6 Real-data MCMC | PENDING | Multi-day work; needs Task #4 finished + data download + Cobaya |
| #7 Tier 2 high-n comb prediction | DONE | `synthetic_tests/predict_signature.*`, `comb_peaks_table.md` |
| #8 Differentiation analysis | DONE | `theory/differentiation_*` |
| #9 Paper 1 draft | OUTLINED | `notes/02_paper1_outline.md`; sections wired to artifacts |

## Concrete scientific results

1. **Stability proof passes** for β₀ ≲ 0.01, ε ≲ 1, w_DE ≲ −1.05. The dominant
   γ_n is universally γ₁ — a structural fact making the stability bound
   one-dimensional in β₀/|1+w_DE|.

2. **First five comb peaks** at k = 1.5, 2.2, 2.6, 3.2, 3.4 × 10⁻³ h/Mpc.
   First two below DESI k_min (Tier 3); n ≥ 50 inside DESI linear regime
   (Tier 2).

3. **Multi-z synthetic recovery validates methodology.** DR3-class noise
   cleanly separates α = 0.5 from α = 0.3 at canonical coupling. DR2-class
   noise is at the sensitivity edge.

4. **Differentiation confirmed.** Multi-frequency comb is the unique DDEM
   signature; no competitor model in the published literature predicts a
   matching feature.

## What blocks Paper 1 submission

- Full CLASS C-source modification (Task #4 main scope). The Python emulator
  is a good proxy but reviewers will require the full Boltzmann pipeline for
  the real-data result. Estimate: this is the largest remaining build effort.

- Real-data MCMC (Task #6). Once the CLASS modification is validated against
  the four-stage gate in `class_mod/ddem_patch_README.md`, the Cobaya pipeline
  with Planck PR4 / DESI DR2 / ACT DR6 / Euclid Q1 likelihoods can run. The
  MCMC itself is days of cluster time.

## Environment notes

- Windows venv at `.venv/`: Python 3.12.10, numpy 2.4.4, scipy 1.17.1, sympy 1.14,
  matplotlib 3.10.9, mpmath 1.3, emcee, corner, getdist.
- WSL Ubuntu 26.04 venv at `/opt/ddem-venv` (Python 3.14): classy 3.3.4 + scientific stack.
- CLASS C binary built at `class_mod/CLASS-source/class`. Builds cleanly with gcc 15.

## How to resume

The Python emulator is the entry point for all downstream development. To
re-run the full pipeline end-to-end:

```bash
# Windows: doom-factor + Python emulator validation
.venv\Scripts\python.exe theory\doom_factor_analysis.py
.venv\Scripts\python.exe theory\ddem_background.py
.venv\Scripts\python.exe theory\ddem_perturbation.py
.venv\Scripts\python.exe synthetic_tests\predict_signature.py
.venv\Scripts\python.exe synthetic_tests\inject_recover_v3_multiz.py
.venv\Scripts\python.exe theory\differentiation_table.py

# WSL: vanilla CLASS sanity
wsl -d Ubuntu --user root -e bash -c "source /opt/ddem-venv/bin/activate && \
    cd /mnt/d/.../CLASS-source && ./class explanatory.ini"
```

The CLASS C-source modification picks up from `class_mod/ddem_patch_README.md`.
