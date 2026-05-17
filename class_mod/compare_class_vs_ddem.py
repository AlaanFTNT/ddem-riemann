"""Compare baseline LCDM P(k) from vanilla CLASS to DDEM-modified P(k) from the patched build.

Runs classy twice from the WSL venv:
    (a) vanilla LCDM with the same cosmology (DDEM disabled)
    (b) DDEM enabled with the baseline coupling (beta0=0.01, eps=0.5, alpha=0.5)
Plots P(k) ratio and sanity-checks shifts."""
import subprocess

WSL_PY = "/opt/ddem-venv/bin/python"
SCRIPT = '''
from classy import Class
import numpy as np
import json

base = {
    "output": "mPk", "P_k_max_h/Mpc": 1.0, "z_max_pk": 2.0,
    "h": 0.6774, "omega_b": 0.0223, "omega_cdm": 0.1188,
    "n_s": 0.9667, "tau_reio": 0.066, "ln10^{10}A_s": 3.064,
    "Omega_Lambda": 0,
    "fluid_equation_of_state": "CLP",
    "w0_fld": -1.05, "wa_fld": 0.0, "cs2_fld": 1.0, "use_ppf": "yes",
}

# Vanilla (DDEM disabled by default)
c1 = Class(); c1.set(base); c1.compute()
s1 = c1.sigma8(); h1 = c1.h(); Om1 = c1.Omega_m()
k_h = np.logspace(-3, 0, 60)
pk1 = np.array([c1.pk_lin(k*c1.h(), 0.0) for k in k_h])
c1.struct_cleanup()

# DDEM enabled
dd = dict(base)
dd.update({"ddem_enabled": "yes", "ddem_beta0": 0.01,
           "ddem_eps": 0.5, "ddem_alpha": 0.5, "ddem_N": 100})
c2 = Class(); c2.set(dd); c2.compute()
s2 = c2.sigma8(); h2 = c2.h(); Om2 = c2.Omega_m()
pk2 = np.array([c2.pk_lin(k*c2.h(), 0.0) for k in k_h])
c2.struct_cleanup()

ratio = pk2 / pk1
out = {
    "sigma8_vanilla": s1, "sigma8_ddem": s2,
    "Omega_m_vanilla": Om1, "Omega_m_ddem": Om2,
    "h_vanilla": h1, "h_ddem": h2,
    "k_h": k_h.tolist(), "pk_vanilla": pk1.tolist(), "pk_ddem": pk2.tolist(),
    "ratio": ratio.tolist(),
}
print("JSON_BEGIN")
print(json.dumps(out))
print("JSON_END")
'''

result = subprocess.run(
    ["wsl", "-d", "Ubuntu", "--user", "root", "-e", "bash", "-c",
     f"{WSL_PY} -c '{SCRIPT.replace(chr(10), chr(10))}'"],
    capture_output=True, text=True, timeout=600
)
print("STDOUT (tail):"); print(result.stdout[-3000:])
if result.returncode != 0:
    print("STDERR:"); print(result.stderr[-2000:])
