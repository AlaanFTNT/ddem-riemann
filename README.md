# Multi-Frequency Dark-Sector Comb Structure: A Riemann-Zeros Realization and Joint Cosmological Constraints

Alaan Franklin (2026). Consortium for Space Mobility and ISAM Capabilities (COSMIC); Tactical Nexus Technologies LLC, Sheridan, Wyoming.

This repository implements a Pourtsidou Type-2 conformal coupled quintessence model in which the dark-sector coupling kernel carries a GUE-class multi-frequency spectrum, realized as a tabulation of the non-trivial Riemann zeros. The code computes background evolution, scalar perturbations through a PPF closure and a per-mode Boltzmann solve, joint MCMC posteriors against Planck, DESI, ACT, and Pantheon+, Bayesian evidence under both the full and canonical-reduction parameterizations, and a GUE pair-correlation Tier 1 test on the recovered comb frequencies. Synthetic injection-recovery harnesses cover the single-redshift and multi-redshift cases. A patch to CLASS is provided for users who need a coupled background solve outside the Python emulator path used in production.

## Citation

If you use this code, cite the paper (citation TBD on submission to JCAP) and the Zenodo DOI (TBD).

## Repository structure

```
ddem-riemann/
├── theory/            background ODE, PPF and per-mode Boltzmann solvers, Riemann-zero tabulation
├── cobaya/            joint MCMC pipeline, individual likelihoods, GUE pair-correlation test, dynesty evidence
├── synthetic_tests/   injection-recovery forecasts at single and multiple redshifts
├── figures/           figure generation scripts and rendered PNGs
├── class_mod/         patch file modifying CLASS at the background level (upstream CLASS not shipped)
├── desi_dr1/          public DESI DR1 LRG1 P(k) covariance for the Appendix A diagnostic
├── pantheon_plus/     public Pantheon+ SNe Ia data release
├── notes/             session notes from the analysis
└── paper/             paper PDF and markdown source
```

## Installation

Python 3.11 or newer is required. A virtual environment is recommended:

```
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

Key dependencies pinned in `requirements.txt`: numpy, scipy, emcee, dynesty, mpmath, getdist, matplotlib, h5py.

## Reproducing the paper's results

Each headline result maps to a single entry point:

- Section 3 doom-factor stability proof: `theory/ddem_background.py` followed by `figures/build_all_figures.py::figure2_doom_factor`
- Section 6 injection-recovery forecast: `synthetic_tests/multi_redshift_recovery.py`
- Section 7 joint MCMC posterior: `cobaya/run_joint_mcmc.py`
- Section 7.2 Bayesian evidence (full parameterization and canonical reduction): `cobaya/bayesian_evidence.py [--trimmed]`
- Section 8.1 GUE pair-correlation Tier 1 test: `cobaya/gue_pair_correlation_test.py`
- Section 8.2 per-mode Boltzmann calibration: `theory/boltzmann_perturbation.py`
- All paper figures: `figures/build_all_figures.py`

## CLASS modification

The file `class_mod/ddem_patch.c` patches the coupled-quintessence Q kernel into the CLASS background module. Users must clone the upstream CLASS repository separately and apply the patch against that tree; the patched CLASS source is not redistributed here. The production likelihoods do not invoke the patched CLASS build. They use the faster Python emulator under `theory/`, which is calibrated against the patched CLASS at the percent level in the uncoupled limit. The CLASS path is retained for users who want a fully coupled background solve.

## License

MIT. See [LICENSE](LICENSE).

## Contact

alaan.franklin@tacnextech.com
