"""Perturbation-level feature-amplitude constraint on P(k) oscillations.

Data source: Beutler, Biagetti, Green, Slosar, and Wallisch (2023),
"Constraints on primordial features from BOSS and eBOSS" (arXiv:2303.13946,
JCAP 2023). Combined BOSS DR12 + eBOSS DR16 95% upper bound |A_log| < 0.04
over the omega_log window [10, 360]. The DDEM comb maps to a log oscillation
of frequency omega_log = gamma_n at the n-th Riemann zero, with peak position
k_n = gamma_n / eta_0.

Per-mode amplitude — dynamical derivation: the §5.2 linear-response formula
A_n^LR = 2 eps a_eff^alpha / |rho_n| was found to over-estimate the comb
amplitude by a factor of ~500 once the per-mode transfer function is
computed dynamically by the coupled-Boltzmann solver in
`theory/boltzmann_perturbation.py`. The rapid oscillation of Q(a) during
matter-era growth integration averages out most of the imprint that the LR
formula assumes is instantaneously deposited. From the canonical (beta0 =
0.005, eps = 0.2, alpha = 0.5) Boltzmann run, the median ratio
A_n^Boltz / A_n^LR across the first 15 Riemann modes is 0.002. We adopt the
calibrated form

    A_n^Boltz(beta0, eps, alpha) = R_calib * A_n^LR(eps, alpha) * (beta0 / 0.005)

with R_calib = 0.002, treating the scaling as linear in beta0 and eps
(verified in the small-Q regime per `theory/linear_response_validation.py`)
and absorbing weak alpha dependence into the residual modest variation
across modes. This is the per-mode amplitude entered into the Beutler
likelihood; the LR formula is retained for reference and as a conservative
upper bound (see §8.2 of the paper).
"""
import numpy as np

# Beutler et al. 2023 95% credible bound on |A_log| in the log-oscillation
# frequency window of interest. From Figure 9 of the paper.
LIMIT_95   = 0.04
SIGMA_LOG  = LIMIT_95 / 1.96      # 1-sigma one-sided Gaussian width

# Beutler omega_log prior range (Table 1): only modes with gamma_n in this
# window are bounded by the published analysis. gamma_1 = 14.135 is inside.
OMEGA_LOG_MIN = 10.0
OMEGA_LOG_MAX = 360.0

# Boltzmann/LR calibration ratio derived from the canonical run.
R_CALIB_BOLTZMANN = 0.002
BETA0_CANONICAL   = 0.005


def chi2_feature_per_mode(eps, alpha, a_eff=0.6, N_zeros=200, beta0=None,
                           use_boltzmann=True):
    """Sum of per-mode chi^2 contributions from the Beutler log-oscillation bound.

    With use_boltzmann=True (default), the amplitude is

        A_n = R_calib * 2 eps a_eff^alpha / |rho_n| * (beta0 / 0.005)

    matching the Boltzmann-derived transfer-function comb (theory/
    boltzmann_perturbation.py). If beta0 is None, beta0 = 0.005 is assumed
    (i.e., uses the canonical calibration without rescaling for the
    independent beta0 prior). With use_boltzmann=False, returns the
    conservative §5.2 LR upper bound for comparison.

    Returns (chi2, A_max).
    """
    from ddem_background import riemann_zeros          # lazy import to keep module light
    gamma_n = riemann_zeros(N_zeros)
    rho_mod = np.sqrt(alpha**2 + gamma_n**2)
    constrained = (gamma_n >= OMEGA_LOG_MIN) & (gamma_n <= OMEGA_LOG_MAX)
    if not constrained.any():
        return 0.0, 0.0
    A_n_LR = 2.0 * abs(eps) * (max(a_eff, 1e-6)**alpha) / rho_mod
    if use_boltzmann:
        scale = R_CALIB_BOLTZMANN * ((beta0 if beta0 is not None else BETA0_CANONICAL)
                                      / BETA0_CANONICAL)
        A_n = A_n_LR * abs(scale)
    else:
        A_n = A_n_LR
    A_constrained = A_n[constrained]
    chi2 = float(np.sum((A_constrained / SIGMA_LOG)**2))
    return chi2, float(np.max(A_constrained))


if __name__ == "__main__":
    print(f"Beutler+2023 log-oscillation 95% bound: |A_log| <= {LIMIT_95:.3f}")
    print(f"  1-sigma equivalent: {SIGMA_LOG:.4f}")
    print(f"  omega_log window: [{OMEGA_LOG_MIN}, {OMEGA_LOG_MAX}]")
    print()
    print("DDEM per-mode amplitudes at canonical (eps=0.5, alpha=0.5, a_eff=0.6):")
    chi2, A_max = chi2_feature_per_mode(eps=0.5, alpha=0.5)
    print(f"  total chi^2 = {chi2:.3f}, A_max = {A_max:.4f}")
    print()
    for eps_try in [0.0, 0.1, 0.2, 0.5, 1.0]:
        chi2, A_max = chi2_feature_per_mode(eps=eps_try, alpha=0.5)
        print(f"  eps={eps_try:.2f}:  chi^2 = {chi2:7.3f},  A_max = {A_max:.4f}")
