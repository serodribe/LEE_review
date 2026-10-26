"""Remake Fig. 3 (right) and Fig. 4 of Section 7.1 from the logs of section7_checks.py,
in the style of the LEE_review notebook."""
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "mathtext.fontset": "cm",
})
num = r"([-\d.]+)"

def davies_tf(sigmas, nuisances, c=10.0, lam=0.5):
    """Trials factor 1 + (1/2pi) int sqrt(-rho_11(m)) dm e^{-c/2} / p_local, with rho computed from the
    binned signal templates projected orthogonally to the background directions removed by the fit
    (normalisation: 'norm', slope: 'slope'), inner product weighted by 1/b (Poisson, at the null)."""
    import bs_trials_check as b
    bb, b1, _ = b.bkg_and_derivs(np.array([lam])); bb, b1 = bb[0], b1[0]
    ip = lambda u, v: np.sum(u * v / bb, -1)
    dirs = [d for k, d in (("norm", bb), ("slope", b1)) if k in nuisances]
    m = np.linspace(b.XMIN, b.XMAX, 4001)
    out = []
    for sg in sigmas:
        g = b.sig_shapes(m, sg)
        if dirs:
            B = np.stack(dirs)
            G = np.array([[ip(x, y) for y in B] for x in B])
            g = g - np.linalg.solve(G, np.stack([ip(g, x) for x in B])).T @ B
        w = g / np.sqrt(ip(g, g))[:, None]
        dw = np.gradient(w, m[1] - m[0], axis=0)
        n_up = np.trapezoid(np.sqrt(ip(dw, dw)), m) / (2 * np.pi) * np.exp(-c / 2)
        out.append(1 + n_up / (0.5 * stats.chi2.sf(c, 1)))
    return np.array(out)


def style_legend(leg):
    leg.get_frame().set_edgecolor("0.6"); leg.get_frame().set_linewidth(0.8)

# ---------------- Fig. 3 right: trials factor vs number of mass points ----------------
rows = [l.split() for l in open("chk_grid.log") if re.match(r"\s*\d+\s+\d", l)]
d = np.array([[float(x) for x in r if x != "+-"] for r in rows])
N, gv05, gv05e, gv75, gv75e, emp, empe = d[:, 0], d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 7]

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(N, N, "--", color="0.35", lw=1.6, marker="s", ms=4, label="Bonferroni ($=N$)")
ax.errorbar(N, gv05, yerr=gv05e, fmt="o-", ms=4.5, color="crimson", ecolor="crimson",
            elinewidth=0.9, capsize=2, lw=1.6, label=r"Gross–Vitells ($c_0=0.5$)")
ax.errorbar(N, gv75, yerr=gv75e, fmt="^-", ms=4.5, color="royalblue", ecolor="royalblue",
            elinewidth=0.9, capsize=2, lw=1.6, label=r"Gross–Vitells ($c_0=7.5$)")
ax.errorbar(N, emp, yerr=empe, fmt="D", ms=4, mfc="white", color="k", ecolor="k",
            elinewidth=0.9, capsize=2, label="Pseudo-experiments (discrete scan)")
ax.set_xlabel(r"Number of scanned mass points, $N$")
ax.set_ylabel(r"Trials factor, $p_{\mathrm{global}}/p_{\mathrm{local}}$")
ax.set_yscale("log")
style_legend(ax.legend(frameon=True, loc="lower right", handlelength=1.8, borderpad=0.6))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/trial_factor_Gv_Bon.{ext}", dpi=200)

# ---------------- Fig. 4: trials factor vs template width ----------------
rows = [l.split() for l in open("chk_width.log") if re.match(r"\s*\d+\.\d+\s+\d", l)]
d = np.array([[float(x) for x in r if x != "+-"] for r in rows])
sig, tf, tfe = d[:, 0], d[:, 2], d[:, 3]
c = 10.0
kappa = np.exp(-c / 2) / (2 * np.sqrt(2) * np.pi * stats.norm.sf(np.sqrt(c)))
s = np.linspace(0.08, 5.1, 400)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.axhline(1.0, ls="--", color="gray", lw=1.2)
ax.plot(s, 1 + kappa * 10 / s, "--", color="crimson", lw=1.4, label="Narrow-template approximation")
s_fine = np.linspace(0.08, 5.1, 120)
ax.plot(s_fine, davies_tf(s_fine, ("norm", "slope")), "-", color="black", lw=1.3,
        label="Davies, background slope fitted")
ax.plot(s_fine, davies_tf(s_fine, ("norm",)), "-", color="royalblue", lw=1.3,
        label="Davies, background slope fixed")
ax.errorbar(sig, tf, yerr=tfe, fmt="o", ms=4, color="black", ecolor="black",
            elinewidth=0.8, capsize=0, label=r"GV toys, background slope fitted")
rows = [l.split() for l in open("chk_width_fixnu.log") if re.match(r"\s*\d+\.\d+\s+\d", l)]
d2 = np.array([[float(x) for x in r if x != "+-"] for r in rows])
ax.errorbar(d2[:, 0], d2[:, 2], yerr=d2[:, 3], fmt="o", ms=4.5, mfc="white", color="royalblue",
            ecolor="royalblue", elinewidth=0.8, capsize=0, label=r"GV toys, background slope fixed")
ax.set_xlabel(r"Signal template width, $\sigma$")
ax.set_ylabel(r"Trials factor, $p_{\mathrm{global}}/p_{\mathrm{local}}$")
ax.set_yscale("log")
style_legend(ax.legend(frameon=True, loc="upper right", handlelength=1.8, borderpad=0.6, fontsize=10))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/tfactor_vs_signalwidth.{ext}", dpi=200)
print("done")
