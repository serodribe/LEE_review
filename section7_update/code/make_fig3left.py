#!/usr/bin/env python3
"""Fig. 3 (left) of Section 7.1: one background-only pseudo-dataset, its fits, and the scan q(m)
with the upcrossings of c0 = 0.5. Fixed seed; picks the first toy with a typical maximum
(3.5 < q_hat < 5) in the low-count region 6 < m_hat < 9.3, where the excess is visible on the log scale, so that the figure is reproducible.
Produces two versions of the lower panel: q(m) and sqrt(q(m))."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

import bs_trials_check as b

mpl.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "mathtext.fontset": "cm",
})
C0, N0_GV = 0.5, 4.62                      # <N(c0)> measured on the 350-point grid (Table 1 run)
m_grid = np.linspace(b.XMIN, b.XMAX, 350)
S = b.sig_shapes(m_grid)
mean = b.N_EVENTS * b.bkg_and_derivs(np.array([b.LAM_TRUE]))[0][0] * b.DX

rng = np.random.default_rng(7)
for itoy in range(1000):
    n = rng.poisson(mean).astype(float)
    q = b.q_scan(n[None], S)[0][0]
    j = np.argmax(q)
    if 3.5 < q[j] < 5 and 6.0 < m_grid[j] < 9.3:
        break
m_hat, q_hat = m_grid[j], q[j]
up = np.flatnonzero((q[:-1] < C0) & (q[1:] >= C0))
m_up = m_grid[up] + (C0 - q[up]) / (q[up + 1] - q[up]) * (m_grid[1] - m_grid[0])   # linear interpolation
p_loc = 0.5 * stats.chi2.sf(q_hat, 1)
p_gv = min(1.0, p_loc + N0_GV * np.exp(-(q_hat - C0) / 2))
print(f"toy {itoy}: m_hat = {m_hat:.2f}, q_hat = {q_hat:.2f} (Z_local = {np.sqrt(q_hat):.2f}), "
      f"{len(up)} upcrossings of c0, p_local = {p_loc:.3g}, p_global(GV) = {p_gv:.2f} "
      f"(Z_global = {stats.norm.isf(p_gv):.2f})")

# fits for the upper panel
nll0, lam0 = b.fit_null(n[None])
c = n.sum() * b.DX
def nll1(p):
    lam, mu = p
    bb = b.bkg_and_derivs(np.array([lam]))[0][0]
    nu = c * ((1 - mu) * bb + mu * S[j])
    return np.sum(nu - n * np.log(nu))
lam1, mu1 = minimize(nll1, [lam0[0], 0.001], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10}).x
xs = np.linspace(b.XMIN, b.XMAX, 600)
bkg_pdf = lambda x, lam: lam * np.exp(-lam * x) / (1 - np.exp(-lam * b.XMAX))
sig = stats.norm.pdf(xs, m_hat, b.SIGMA_SIG) / (stats.norm.cdf(b.XMAX, m_hat, b.SIGMA_SIG) - stats.norm.cdf(b.XMIN, m_hat, b.SIGMA_SIG))
y_b = c * bkg_pdf(xs, lam0[0])
y_sb = c * ((1 - mu1) * bkg_pdf(xs, lam1) + mu1 * sig)

for version in ("q", "sqrtq"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True,   # same size as Fig. 3 right
                                 gridspec_kw={"height_ratios": [2, 1.5], "hspace": 0.06})
    a1.errorbar(b.X, n, yerr=np.sqrt(n), fmt="o", ms=3.2, color="black", elinewidth=0.8,
                capsize=0, label="Pseudo-data (background only)", zorder=5)
    a1.plot(xs, y_sb, "-", color="crimson", lw=1.6, label=rf"Signal + background fit, $\hat m={m_hat:.2f}$")
    a1.set_yscale("log"); a1.set_ylabel("Events per bin")
    a1.set_ylim(0.5 * n.min(), 2.2 * n.max())
    leg = a1.legend(frameon=True, loc="upper right"); leg.get_frame().set_edgecolor("0.6")

    yv = q if version == "q" else np.sqrt(q)
    lvl = C0 if version == "q" else np.sqrt(C0)
    a2.fill_between(m_grid, lvl, yv, where=yv >= lvl, color="royalblue", alpha=0.18, lw=0)
    a2.plot(m_grid, yv, "-", color="black", lw=1.4)
    a2.axhline(lvl, ls="--", color="crimson", lw=1.5)
    a2.plot(m_up, np.full(m_up.size, lvl), "o", color="royalblue", mec="black", mew=0.6, ms=7, zorder=5,
            label=r"Upcrossings of $c_0$")
    flat = np.flatnonzero(yv < lvl)                     # put the label where the curve is below c0
    seg = np.split(flat, np.flatnonzero(np.diff(flat) > 1) + 1)
    longest = max(seg, key=len)
    a2.text(m_grid[longest].mean(), lvl + 0.04 * yv.max(),
            r"$c_0=0.5$" if version == "q" else r"$\sqrt{c_0}$", color="crimson",
            va="bottom", ha="center", fontsize=12)
    a2.set_ylim(0, 1.25 * yv.max())
    a2.set_ylabel(r"$q(m)$" if version == "q" else r"$\sqrt{q(m)}$")
    a2.set_xlabel(r"$m$"); a2.set_xlim(b.XMIN, b.XMAX)
    leg = a2.legend(frameon=True, loc="upper left"); leg.get_frame().set_edgecolor("0.6")
    fig.align_ylabels([a1, a2])
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.11, hspace=0.06)
    name = "mu_scan_upcrossings" if version == "q" else "mu_scan_upcrossings_sqrtq"
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200)
