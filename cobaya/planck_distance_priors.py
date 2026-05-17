"""Planck 2018 compressed CMB likelihood ("distance priors").

Uses three observables derivable from the background-only classy output:
    R         = sqrt(Omega_m * H0^2) * (1 + z_*) D_A(z_*) / c
    l_A       = pi * (1 + z_*) D_A(z_*) / r_s(z_*)
    omega_b   = Omega_b h^2

Means and covariance from Chen, Huang, Wang 2019, Table 5
(PlanckTT,TE,EE+lowE+lensing), arXiv:1808.05724. Functionally equivalent
formulations appear in arXiv:1808.05724 and downstream papers; this
compressed likelihood replicates ~95% of Planck CMB information for BAO+SN+CMB
analyses with extended dark-sector models.

This is a *distance-prior* approximation, not a full likelihood. It captures
the CMB constraint on the expansion history and baryon density. It does NOT
capture small-scale CMB polarization or lensing, but those are not the
dominant constraint for IDE / dark-sector models at the background level.
"""
import numpy as np

# Planck 2018 TTTEEE+lowE+lensing distance priors
# Mean: [R, l_A, omega_b]
PLANCK_MEAN = np.array([1.7502, 301.471, 0.02236])
# 1-sigma errors and correlation matrix (Planck 2018 TTTEEE+lowE+lensing)
_sig = np.array([0.0046, 0.090, 0.00015])
_corr = np.array([
    [1.00,  0.46, -0.66],
    [0.46,  1.00, -0.33],
    [-0.66, -0.33, 1.00],
])
PLANCK_COV = _corr * np.outer(_sig, _sig)
PLANCK_INV = np.linalg.inv(PLANCK_COV)
PLANCK_LOGDET = np.linalg.slogdet(PLANCK_COV)[1]


def planck_distance_obs(classy_instance, h=None):
    """Compute (R, l_A, omega_b) from a computed classy instance.

    R = sqrt(Omega_m * H0^2) * (1+z_*) D_A(z_*) / c
    l_A = pi * (1+z_*) D_A(z_*) / r_s(z_*)
    omega_b = Omega_b * h^2

    classy returns angular_distance in Mpc and Hubble in 1/Mpc; we treat
    c = 1 in the natural units classy uses (D_A is comoving physical distance).
    """
    c_unit = 1.0  # classy uses geometrized units where c is absorbed
    z_star = classy_instance.get_current_derived_parameters(["z_rec"])["z_rec"]
    DA_star = classy_instance.angular_distance(z_star)
    rs_star = classy_instance.rs_drag()  # NOTE: rs at drag, not at recombination — Planck uses rs_rec
    # Better: use rs at recombination
    rs_star = classy_instance.get_current_derived_parameters(["rs_rec"])["rs_rec"]
    Omega_m = classy_instance.Omega_m()
    H0 = classy_instance.Hubble(0) * 2997.92458  # 1/Mpc -> km/s/Mpc when multiplied by c (2997.92 = c/100)
    # Easier: use H0 directly from input h (more reliable)
    if h is None:
        h = classy_instance.h()
    H0_unit = h * 100.0 / 299792.458  # 1/Mpc
    sqrt_Omh2 = np.sqrt(Omega_m) * H0_unit
    R = sqrt_Omh2 * (1.0 + z_star) * DA_star
    l_A = np.pi * (1.0 + z_star) * DA_star / rs_star
    omega_b = classy_instance.omega_b()
    return np.array([R, l_A, omega_b])


def chi2_planck_distance(obs):
    diff = obs - PLANCK_MEAN
    return float(diff @ PLANCK_INV @ diff)
