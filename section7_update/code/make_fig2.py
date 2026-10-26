#!/usr/bin/env python3
"""Figure 2 of Section 7.1: null distributions of q(m0) and q(m_hat), and calibration of the
corresponding p-values. q(m0): new background-only toys at m0 = 3; q(m_hat): the 350-point-grid
toys of run_section71.py (run71_350*.npz)."""
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import bs_trials_check as b

mpl.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "mathtext.fontset": "cm",
})
N_FIXED, M0 = 100_000, 3.0

# q(m0): fit at a single, pre-specified mass
rng = np.random.default_rng(2026)
mean = b.N_EVENTS * b.bkg_and_derivs(np.array([b.LAM_TRUE]))[0][0] * b.DX
S0 = b.sig_shapes(np.array([M0]))
q0 = []
for _ in range(N_FIXED // 1000):
    n = rng.poisson(mean, size=(1000, mean.size)).astype(float)
    q0.append(b.q_scan(n, S0)[0][:, 0])
q0 = np.concatenate(q0)
# q(m_hat): maximum over the 350-point scan
qhat = np.concatenate([np.load(f)["qmax"] for f in sorted(glob.glob("run71_350*.npz"))])
print(f"{q0.size} fixed-mass toys (fraction with q=0: {np.mean(q0 == 0):.3f}), {qhat.size} scanned toys")

p0 = 0.5 * stats.chi2.sf(q0, 1)        # valid p-value at a pre-specified mass (p = 1/2 when q = 0)
ploc = 0.5 * stats.chi2.sf(qhat, 1)    # local p-value of the scan maximum

def style_legend(leg):
    leg.get_frame().set_edgecolor("0.6"); leg.get_frame().set_linewidth(0.8)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

# left: null densities
a = ax[0]
bins = np.linspace(0, 16, 65)
a.hist(q0, bins=bins, density=True, histtype="stepfilled", color="royalblue", alpha=0.3,
       edgecolor="royalblue", lw=1.2, label=r"Fixed mass, $q(m_0)$")
a.hist(qhat, bins=bins, density=True, histtype="step", color="black", lw=1.4,
       label=r"Scanned mass, $q(\hat m)$")
qq = np.linspace(0.01, 16, 500)
a.plot(qq, 0.5 * stats.chi2.pdf(qq, 1), "-", color="crimson", lw=1.6, label=r"$\frac{1}{2}\chi^2_1$")
a.set_yscale("log"); a.set_xlim(0, 16); a.set_ylim(1e-5, 5)
a.set_xlabel(r"Test statistic $q$"); a.set_ylabel("Probability density")
style_legend(a.legend(frameon=True, loc="upper right"))

# right: calibration, P(p <= alpha) vs alpha
a = ax[1]
alpha = np.logspace(-5, 0, 400)
cdf = lambda p: np.searchsorted(np.sort(p), alpha, side="right") / p.size
a.plot(alpha, cdf(p0), "-", color="royalblue", lw=1.8, label=r"Fixed mass, $p$-value of $q(m_0)$")
a.plot(alpha, cdf(ploc), "-", color="black", lw=1.8, label=r"Scanned mass, $p_\mathrm{local}$")
a.plot(alpha, alpha, "--", color="0.45", lw=1.2, label=r"Calibrated, $P(p\leq\alpha)=\alpha$")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-5, 1); a.set_ylim(1e-5, 1.5)
a.set_xlabel(r"$\alpha$"); a.set_ylabel(r"$P(p\leq\alpha\,|\,H_0)$")
style_legend(a.legend(frameon=True, loc="lower right"))

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/null_dist_pvalues.{ext}", dpi=200)
# the two panels as separate files, with the names used in the tex
from matplotlib.transforms import Bbox
fig.canvas.draw()
for a_, name in zip(ax, ("q_null_dist", "pval_dist")):
    bb = a_.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"figures/{name}.pdf", bbox_inches=Bbox.from_extents(bb.x0 - 0.05, bb.y0 - 0.05, bb.x1 + 0.05, bb.y1 + 0.05))
for a_ in (0.05, 0.01, 1e-3):
    print(f"alpha={a_}: P(p0<=a)={np.mean(p0 <= a_):.4f}  P(p_local<=a)={np.mean(ploc <= a_):.4f}")
