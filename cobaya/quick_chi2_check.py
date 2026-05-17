"""Quick chi^2 check: how does vanilla LCDM compare to DDEM-baseline on DESI DR2 BAO?
Computes chi^2 for several parameter points without MCMC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from desi_dr2_bao_data import chi2_desi_bao, DESI_DR2_BAO
from classy import Class
import numpy as np

def chi2_point(name, ddem_enabled, beta0=0.0, eps=0.0, alpha=0.5,
               h=0.6774, omega_b=0.0223, omega_cdm=0.1188, w0_fld=-1.0):
    p = {
        "output": "", "h": h, "omega_b": omega_b, "omega_cdm": omega_cdm,
        "n_s": 0.9667, "tau_reio": 0.066, "ln10^{10}A_s": 3.064,
        "Omega_Lambda": 0.0, "fluid_equation_of_state": "CLP",
        "w0_fld": w0_fld, "wa_fld": 0.0, "cs2_fld": 1.0, "use_ppf": "yes",
    }
    if ddem_enabled:
        p.update({"ddem_enabled": "yes", "ddem_beta0": beta0,
                  "ddem_eps": eps, "ddem_alpha": alpha, "ddem_N": 100})
    c = Class()
    c.set(p); c.compute()
    rd = c.rs_drag()
    DM = {b["z"]: (1+b["z"])*c.angular_distance(b["z"])/rd for b in DESI_DR2_BAO}
    DH = {b["z"]: 1.0/(c.Hubble(b["z"])*rd) for b in DESI_DR2_BAO}
    DV = {b["z"]: (b["z"]*((1+b["z"])*c.angular_distance(b["z"]))**2/c.Hubble(b["z"]))**(1/3)/rd
          for b in DESI_DR2_BAO}
    chi2 = chi2_desi_bao(
        lambda z: DM[z], lambda z: DH[z], lambda z: DV[z])
    c.struct_cleanup()
    print(f"{name:35s}  chi2={chi2:7.3f}  rd={rd:.2f} Mpc  Om={c.Omega_m() if False else (omega_b+omega_cdm)/h/h:.4f}")
    return chi2

print("DESI DR2 BAO data points (Ndof = sum of measurements):")
ndata = sum(len(b["meas"]) for b in DESI_DR2_BAO)
print(f"  Ndata = {ndata}")
print()
print("chi^2 at various parameter points:")
chi2_point("vanilla LCDM (w=-1)",    False, w0_fld=-1.0)
chi2_point("phantom w=-1.05",         False, w0_fld=-1.05)
chi2_point("DDEM baseline (b=.01,e=.5,a=.5,w=-1.05)", True,
           beta0=0.01, eps=0.5, alpha=0.5, w0_fld=-1.05)
chi2_point("DDEM small (b=.001,e=.5,a=.5,w=-1.05)", True,
           beta0=0.001, eps=0.5, alpha=0.5, w0_fld=-1.05)
chi2_point("DDEM tiny  (b=.0001,e=.5,a=.5,w=-1.05)", True,
           beta0=0.0001, eps=0.5, alpha=0.5, w0_fld=-1.05)
chi2_point("DDEM alpha=0  (b=.001,e=.5,a=0)", True,
           beta0=0.001, eps=0.5, alpha=0.0, w0_fld=-1.05)
chi2_point("DDEM alpha=1  (b=.001,e=.5,a=1)", True,
           beta0=0.001, eps=0.5, alpha=1.0, w0_fld=-1.05)
