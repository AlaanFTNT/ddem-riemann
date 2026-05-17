"""Numerical inversion of the canonical-DDEM Q kernel to scalar-field
functions f(phi) and V(phi).

Background: §2.1 of the paper specifies the Pourtsidou Type-2 conformal
coupled-quintessence action

    S = int d^4x sqrt(-g) [R/16piG + L_DE - rho_m f(phi)] + S_r ,

with L_DE the dark-sector scalar Lagrangian and the coupling

    Q = -(d ln f / dphi) phi-dot rho_m .

For phantom equation of state w_DE < -1, the canonical kinetic term cannot
deliver the required rho_DE+p_DE = phi-dot^2 < 0. We therefore adopt a
phantom kinetic term

    L_DE = -(1/2)(partial phi)^2 - V(phi) ,

which gives rho_DE = -(1/2) phi-dot^2 + V and p_DE = -(1/2) phi-dot^2 - V,
so

    (1 + w_DE) = -phi-dot^2 / rho_DE       (negative for phi-dot^2 > 0)
    phi-dot^2(a) = - rho_DE(a) (1 + w_DE(a))

For constant w_DE = -1.05 in the §3 prior region, phi-dot^2(a) = 0.05 rho_DE(a).

Inversion procedure for canonical DDEM (beta0 = 0.005, eps = 0.2,
alpha = 0.5, w_DE = -1.05):

  (i)   Solve the background to get rho_m(a), rho_DE(a), H(a), Q(a).
  (ii)  Compute phi-dot(a) = sqrt(-rho_DE(1+w_DE)). Sign convention: choose
        phi-dot > 0 (monotone phi growth).
  (iii) Integrate phi(a) - phi(a_init) = int (phi-dot / H) d(ln a).
  (iv)  Compute d ln f / dphi = -Q / (phi-dot rho_m) and integrate to get
        ln f(phi), then f(phi) = exp(ln f(phi)).
  (v)   Compute V(phi) = rho_DE + (1/2) phi-dot^2 along the trajectory.

Phantom-scalar pathologies: a phantom kinetic term is a ghost field. We
treat the phantom realisation as an effective parametrisation; promoting Q
itself to an effective evolving-w mechanism with a canonical-kinetic scalar
is the natural follow-up but is not done here.

This script demonstrates that f(phi) is bounded and slowly varying and that
V(phi) is bounded below over the observable range a in [a_min, 1], so the
action is admissible at the level required for the §7 analysis.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "figures"))

from ddem_background import solve_background, Q_bracket, riemann_zeros


def invert_scalar(beta0=0.005, eps=0.2, alpha=0.5, w_DE=-1.05,
                  Omega_m0=0.315, Omega_DE0=0.685, Omega_r0=9.05e-5,
                  a_min=1e-4, a_max=1.0, n_grid=4000, N_zeros=200):
    """Run the §2.1 inversion. All densities in natural units (rho_crit,0 = 1,
    H0 = 1)."""
    bg = solve_background(beta0=beta0, eps=eps, alpha=alpha, w_DE=w_DE,
                          Omega_m0=Omega_m0, Omega_DE0=Omega_DE0,
                          Omega_r0=Omega_r0, a_min=a_min, a_max=a_max,
                          n_grid=n_grid, N_zeros=N_zeros)
    a   = bg["a"]
    H   = bg["H"]
    rho_m  = bg["rho_m"]
    rho_DE = bg["rho_DE"]
    Q   = bg["Q"]
    lnA = bg["ln_a"]

    # Phantom-scalar field velocity: phi-dot^2 = -rho_DE * (1+w_DE)
    # Constant w_DE here; positive for phantom (1+w<0).
    phidot2 = -rho_DE * (1.0 + w_DE)
    phidot2 = np.maximum(phidot2, 0.0)
    phidot  = np.sqrt(phidot2)          # natural units; H0 = 1

    # phi(a) - phi(a_init) = int_{a_init}^{a} (phi-dot / H) da/a
    integrand_phi = phidot / H
    phi = np.zeros_like(a)
    for i in range(1, a.size):
        phi[i] = phi[i-1] + 0.5 * (integrand_phi[i] + integrand_phi[i-1]) \
                          * (lnA[i] - lnA[i-1])

    # d ln f / dphi = -Q / (phi-dot * rho_m)
    eps_floor = 1e-30
    dlnf_dphi = -Q / (np.maximum(phidot, eps_floor) * np.maximum(rho_m, eps_floor))
    # ln f(phi) - ln f(phi_init) = int dphi dlnf/dphi = int dlna (dphi/dlna) dlnf/dphi
    integrand_lnf = dlnf_dphi * (phidot / H)
    ln_f = np.zeros_like(a)
    for i in range(1, a.size):
        ln_f[i] = ln_f[i-1] + 0.5 * (integrand_lnf[i] + integrand_lnf[i-1]) \
                            * (lnA[i] - lnA[i-1])
    f_phi = np.exp(ln_f)

    # V(phi) = rho_DE + (1/2) phi-dot^2
    V_phi = rho_DE + 0.5 * phidot2

    return {"a": a, "ln_a": lnA, "phi": phi,
            "phidot": phidot, "phidot2": phidot2,
            "f_phi": f_phi, "ln_f": ln_f,
            "V_phi": V_phi, "rho_DE": rho_DE, "rho_m": rho_m,
            "Q": Q, "H": H}


def summarise(r):
    a = r["a"]
    print(f"a-range: [{a[0]:.2e}, {a[-1]:.2e}]  ({a.size} points)")
    print(f"phi(a_min)  = {r['phi'][0]:+.4f}  natural units")
    print(f"phi(a=1)    = {r['phi'][-1]:+.4f}")
    print(f"Delta phi   = {r['phi'][-1] - r['phi'][0]:+.4f}")
    print()
    print(f"f(phi_init) = {r['f_phi'][0]:.6f}")
    print(f"f(phi=phi_today)  = {r['f_phi'][-1]:.6f}")
    print(f"max |ln f - ln f_init|  = {np.max(np.abs(r['ln_f'])):.4e}")
    print(f"max |1 - f|             = {np.max(np.abs(1 - r['f_phi'])):.4e}")
    print()
    V = r["V_phi"]
    print(f"V(phi)  range: [{V.min():+.5f}, {V.max():+.5f}]  (natural units)")
    print(f"V(phi=phi_today) = {V[-1]:+.5f}")
    print(f"V(phi)  bounded below: {V.min() > -1e-6}  (min = {V.min():.3e})")


def figure_scalar_inversion(out_path):
    """Plot f(phi) and V(phi) for the canonical DDEM inversion."""
    r = invert_scalar()
    summarise(r)

    fig, (ax_f, ax_V) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    phi = r["phi"]
    # f(phi): deviation from unity, log scale
    ax_f.plot(phi, r["f_phi"] - 1.0, lw=1.8, color="#1f4068")
    ax_f.axhline(0.0, color="#888888", lw=0.6, ls="--")
    ax_f.set_xlabel(r"$\phi$  (natural units, $H_0 = 1$)")
    ax_f.set_ylabel(r"$f(\phi) - 1$")
    ax_f.set_title(r"Conformal coupling function $f(\phi)$")
    ax_f.grid(alpha=0.3)

    ax_V.plot(phi, r["V_phi"], lw=1.8, color="#ad3b3b")
    ax_V.axhline(0.0, color="#888888", lw=0.6, ls="--")
    ax_V.set_xlabel(r"$\phi$  (natural units, $H_0 = 1$)")
    ax_V.set_ylabel(r"$V(\phi)$  (in $\rho_{\rm crit,0}$ units)")
    ax_V.set_title(r"Scalar potential $V(\phi)$")
    ax_V.grid(alpha=0.3)

    for ax in (ax_f, ax_V):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    OUT = HERE.parent / "figures" / "figure9_scalar_inversion.png"
    figure_scalar_inversion(OUT)
