"""Posterior analysis for the DESI DR2 BAO MCMC chain.
Produces corner plot, 1D marginals, and a markdown report.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import corner

OUT = Path(__file__).parent
chain = np.load(OUT / "chain_desi_bao.npy")
lnp   = np.load(OUT / "chain_lnp.npy")
labels = ["beta0", "eps", "alpha", "h", "omega_b", "omega_cdm", "w0_fld"]
truths = [0.0, None, 0.5, 0.6774, 0.0223, 0.1188, -1.0]

print(f"chain shape: {chain.shape}")
print(f"acceptance / log-posterior range: ln_p min={lnp.min():.2f}  max={lnp.max():.2f}")

# Posterior summary
lines = ["# DESI DR2 BAO posterior — DDEM model (background-only patch)", "",
         f"Chain size: {chain.shape[0]} samples post-burn.", "",
         "| Parameter | median | 68% CI | 95% CI |",
         "|---|---|---|---|"]
for i, lab in enumerate(labels):
    q025, q16, q50, q84, q975 = np.percentile(chain[:, i], [2.5, 16, 50, 84, 97.5])
    lines.append(f"| {lab} | {q50:+.5f} | [{q16:+.5f}, {q84:+.5f}] | [{q025:+.5f}, {q975:+.5f}] |")

# Check whether alpha=0.5 is enclosed in the 68% interval for alpha
i = labels.index("alpha")
q16, q50, q84 = np.percentile(chain[:, i], [16, 50, 84])
enc05 = q16 <= 0.5 <= q84
lines += ["",
          f"**alpha = 0.5 enclosed in 68% CI:** {enc05}",
          f"alpha median = {q50:+.4f}, 68% CI = [{q16:+.4f}, {q84:+.4f}]",
          ""]

# Best-fit point
ibest = int(np.argmax(lnp))
lines += ["## Best-fit (max log-posterior)",
          f"  ln_p_max = {lnp.max():.2f}  (chi2 = {-2*lnp.max():.2f})", ""]
for i, lab in enumerate(labels):
    lines.append(f"  {lab:10s}  {chain[ibest, i]:+.5f}")

(OUT / "chain_summary.md").write_text("\n".join(lines), encoding="utf-8")

# Corner plot
fig = corner.corner(chain, labels=labels, truths=truths,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_fmt=".4f")
fig.savefig(OUT / "chain_corner.png", dpi=120)
plt.close(fig)

print("Saved: chain_summary.md, chain_corner.png")
print("\n".join(lines))
