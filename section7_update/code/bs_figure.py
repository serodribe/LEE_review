#!/usr/bin/env python3
"""Table 1 numbers and the Bayer-Seljak comparison figure for Section 7.1,
from run71_100.npz and run71_350.npz (produced by run_section71.py)."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "mathtext.fontset": "cm",
})
L, C, C0 = 10.0, 10.0, 0.5
p_loc = lambda q: 0.5 * stats.chi2.sf(q, 1)                              # exact one-sided
p_asym = lambda q: np.exp(-q / 2) / np.sqrt(2 * np.pi * q)               # B&S asymptotic form, t = 1


def load(n_grid):
    """Concatenate all available independent replicas run71_<N>.npz, run71_<N>_r1.npz, ..."""
    import glob
    files = sorted(glob.glob(f"run71_{n_grid}.npz") + glob.glob(f"run71_{n_grid}_r*.npz"))
    parts = [dict(np.load(f)) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "levels"}
    d["levels"] = parts[0]["levels"]
    print(f"grid {n_grid}: replicas {files}")
    return d


def summary(n_grid):
    d = load(n_grid)
    qmax, up, lev = d["qmax"], d["up"], d["levels"]
    n = qmax.size
    k10, k0 = np.flatnonzero(np.isclose(lev, C))[0], np.flatnonzero(np.isclose(lev, C0))[0]
    r = {}
    k = np.count_nonzero(qmax >= C)
    r["emp"] = (k / n / p_loc(C), np.sqrt(k * (1 - k / n)) / n / p_loc(C))
    r["dav"] = (1 + up[:, k10].mean() / p_loc(C), up[:, k10].std() / np.sqrt(n) / p_loc(C))
    f = np.exp(-(C - C0) / 2) / p_loc(C)
    r["gv"] = (1 + up[:, k0].mean() * f, up[:, k0].std() / np.sqrt(n) * f)
    r["N0"] = up[:, k0].mean()
    # Bayer-Seljak at q = c from peaks near c: N_B scales as sqrt(q), so rescale each toy to q = c
    ok = np.isfinite(d["sig_m"]) & (np.abs(d["q_pk"] - C) < 2)
    kappa = d["sig_m"][ok] * np.sqrt(d["q_pk"][ok])
    NB = L * np.sqrt(C) / (np.sqrt(2 * np.pi) * kappa)
    tf = (1 - np.exp(-NB * p_asym(C))) / p_loc(C)
    r["bs"] = (tf.mean(), tf.std() / np.sqrt(tf.size), tf.std(), ok.sum())
    r["sqrtlam_curv"] = 1 / kappa.mean()
    r["sqrtlam_up"] = 2 * np.pi * r["N0"] * np.exp(C0 / 2) / L
    return d, r


if __name__ == "__main__":
    res = {}
    for g in (100, 350):
        d, r = summary(g)
        res[g] = (d, r)
        print(f"--- {g}-point grid, {d['qmax'].size} toys ---")
        for key, name in (("dav", "Davies"), ("gv", "GV c0=0.5"), ("emp", "empirical")):
            print(f"  {name:10s} {r[key][0]:6.2f} +- {r[key][1]:.2f}")
        print(f"  B&S        {r['bs'][0]:6.2f} +- {r['bs'][1]:.2f}   (spread between datasets {r['bs'][2]:.1f},"
              f" {100 * r['bs'][2] / r['bs'][0]:.0f}%, from {r['bs'][3]} toys)")
        print(f"  <N(c0)> = {r['N0']:.3f};  sqrt(lambda): curvature {r['sqrtlam_curv']:.3f}, upcrossings {r['sqrtlam_up']:.3f}")

    # ---------------- figure (350-point grid) ----------------
    d, r = res[350]
    qmax, up, lev = d["qmax"], d["up"], d["levels"]
    n = qmax.size
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: trials factor vs local significance
    a = ax[0]
    cc = np.linspace(4.0, 17.6, 300)
    a.plot(np.sqrt(cc), 1 + r["N0"] * np.exp(-(cc - C0) / 2) / p_loc(cc), "-", color="crimson", lw=1.6,
           label=r"Gross–Vitells ($c_0=0.5$)", zorder=3)
    kk = np.array([np.count_nonzero(qmax >= c) for c in lev])
    sel = (kk >= 30) & (lev >= 4.0)
    a.errorbar(np.sqrt(lev[sel]), kk[sel] / n / p_loc(lev[sel]),
               yerr=np.sqrt(kk[sel] * (1 - kk[sel] / n)) / n / p_loc(lev[sel]),
               fmt="D", ms=4, mfc="white", color="k", elinewidth=0.9, capsize=2,
               label="Pseudo-experiments", zorder=4)
    ok = np.isfinite(d["sig_m"]) & (d["q_pk"] > 0.5)
    qi, si = d["q_pk"][ok], d["sig_m"][ok]
    NBi = L / (np.sqrt(2 * np.pi) * si)
    tfi = (1 - np.exp(-NBi * p_asym(qi))) / p_loc(qi)
    zi = np.sqrt(qi)
    edges = np.arange(1.9, 4.31, 0.2)
    zc, med, lo, hi = [], [], [], []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        m = (zi >= z0) & (zi < z1)
        if m.sum() >= 30:
            zc.append(0.5 * (z0 + z1)); med.append(np.median(tfi[m]))
            lo.append(np.percentile(tfi[m], 16)); hi.append(np.percentile(tfi[m], 84))
    a.fill_between(zc, lo, hi, color="royalblue", alpha=0.25, lw=0,
                   label="Bayer–Seljak, 68% of datasets")
    a.plot(zc, med, "-", color="royalblue", lw=1.6, label="Bayer–Seljak, median")
    a.set_xlabel(r"Local significance $Z_\mathrm{local}=\sqrt{q(\hat m)}$")
    a.set_ylabel(r"Trials factor, $p_{\mathrm{global}}/p_{\mathrm{local}}$")
    a.set_xlim(1.9, 4.2); a.set_ylim(0, 80)
    leg = a.legend(frameon=True, loc="upper left")
    leg.get_frame().set_edgecolor("0.6")

    # right: per-toy global p-value, B&S and GV vs pseudo-experiments
    a = ax[1]
    order = np.sort(qmax)
    p_emp = 1 - np.searchsorted(order, qmax, side="left") / n
    p_bs = 1 - np.exp(-np.exp(-d["qS_mps"] / 2))
    a.scatter(p_emp, p_bs, s=1.5, alpha=0.25, color="royalblue", rasterized=True,
              label="Bayer–Seljak (one point per dataset)")
    qs = np.sort(qmax)[::-1]
    pe = np.arange(1, n + 1) / n
    p_gv = np.minimum(1, p_loc(qs) + r["N0"] * np.exp(-(qs - C0) / 2))
    a.plot(pe, p_gv, "-", color="crimson", lw=1.6, label=r"Gross–Vitells ($c_0=0.5$)")
    a.plot([1e-6, 2], [1e-6, 2], "k--", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(1e-5, 1.3); a.set_ylim(1e-5, 1.3)
    a.set_xlabel(r"Global $p$-value from pseudo-experiments")
    a.set_ylabel(r"Estimated global $p$-value")
    leg = a.legend(frameon=True, loc="upper left", markerscale=6)
    leg.get_frame().set_edgecolor("0.6")
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/bs_comparison.{ext}", dpi=200)

    # numbers quoted in the text
    print("\nB&S per-dataset spread of the trials factor (std/mean) by Z bin:")
    for z0 in (2.0, 2.5, 3.0, 3.5, 4.0):
        m = (zi >= z0 - 0.1) & (zi < z0 + 0.1)
        if m.sum() > 20:
            print(f"  Z={z0}: {np.std(tfi[m]) / np.mean(tfi[m]):.2f}  (1/(sqrt2 Z) = {1 / (np.sqrt(2) * z0):.2f}, {m.sum()} toys)")
    print("Calibration of p_BS (MPS):  alpha, P(p_BS <= alpha)")
    for al in (0.3, 0.1, 0.05, 0.02, 0.01):
        print(f"  {al:5.2f}  {np.mean(p_bs <= al):.4f}")
    print(f"edge fraction of MAP peaks: {np.mean((d['m_hat'] < 0.06) | (d['m_hat'] > 9.94)):.3f}")
