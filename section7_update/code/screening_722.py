#!/usr/bin/env python3
"""Section 7.2.2: a scaled-down general search with 50 event classes, a new particle contributing to 10 of them.

Design (fixed in advance, seed DESIGN_SEED):
  * class i has its own background slope nu_i ~ U[0.2, 0.6] and detector resolution sigma_i ~ U[0.15, 0.40];
    it is fitted with a Gaussian template of width sigma_i (as in Section 7.2.1);
  * a narrow resonance (GAMMA = 0.02) at the common mass M_TRUE = 3 is injected in N_SIG = 10 classes chosen at
    random; its observed shape in class i is a Gaussian of width sqrt(GAMMA^2 + sigma_i^2);
  * signal strengths: expected (Asimov) local significances evenly spaced between 2.5 and 4.5 sigma;
  * the mass scan of each class is corrected with GV (c0 = 0.5), <N(c0)> calibrated per class with N_CAL toys;
  * Bonferroni, Sidak and BH at alpha = 0.05 across the 50 classes.
Steps (independent random streams):
  calib    : <N(c0)> per class                                           (seed SEED + 1)
  single   : one pseudo-experiment, for the figure                        (seed SEED)
  ensemble : N_REP pseudo-experiments -> FWER, FDR, identification rates   (seed SEED + 2)
  plot     : redraw figures from the saved results
Usage: python screening_722.py [calib] [single] [ensemble] [plot]   (default: all)
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import bs_trials_check as b
from multichannel_721 import decisions as decisions_generic

DESIGN_SEED, SEED = 722, 20261026
N_CH, N_SIG = 50, 10
M_TRUE, GAMMA = 3.0, 0.02
C0, ALPHA = 0.5, 0.05
N_CAL, N_REP, BATCH, NPROC = 2000, 1000, 100, 2
m_grid = np.linspace(b.XMIN, b.XMAX, 350)

_d = np.random.default_rng(DESIGN_SEED)
NU = np.round(_d.uniform(0.2, 0.6, N_CH), 3)
SIG = np.round(_d.uniform(0.15, 0.40, N_CH), 3)
SIG_IDX = np.sort(_d.choice(N_CH, N_SIG, replace=False))
Z_A = np.zeros(N_CH)
Z_A[SIG_IDX] = _d.permutation(np.linspace(2.5, 4.5, N_SIG))


def gauss_trunc(x, m, s):
    return stats.norm.pdf(x, m, s) / (stats.norm.cdf(b.XMAX, m, s) - stats.norm.cdf(b.XMIN, m, s))


def expected(i, mu):
    bb = b.bkg_and_derivs(np.array([NU[i]]))[0][0]
    return b.N_EVENTS * b.DX * ((1 - mu) * bb + mu * gauss_trunc(b.X, M_TRUE, np.hypot(GAMMA, SIG[i])))


def mu_for_asimov(i):
    if Z_A[i] <= 0:
        return 0.0
    S1 = b.sig_shapes(np.array([M_TRUE]), SIG[i])
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if b.q_scan(expected(i, mid)[None], S1)[0][0, 0] < Z_A[i] ** 2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


MU = np.array([mu_for_asimov(i) for i in range(N_CH)])


def qmax_and_up(i, n):
    q = b.q_scan(n, b.sig_shapes(m_grid, SIG[i]), n_iter=12)[0]
    return q.max(axis=1), b.upcrossings(q, C0)


def calib_class(i):
    rng = np.random.default_rng([SEED + 1, i])
    mean0 = expected(i, 0.0)
    up = np.concatenate([qmax_and_up(i, rng.poisson(mean0, size=(BATCH, mean0.size)).astype(float))[1]
                         for _ in range(N_CAL // BATCH)])
    return float(up.mean()), float(up.std() / np.sqrt(up.size))


def ensemble_class(i):
    rng = np.random.default_rng([SEED + 2, i])
    mean = expected(i, MU[i])
    return np.concatenate([qmax_and_up(i, rng.poisson(mean, size=(BATCH, mean.size)).astype(float))[0]
                           for _ in range(N_REP // BATCH)])


def p_gv(q, N0):
    return np.minimum(1.0, 0.5 * stats.chi2.sf(q, 1) + N0 * np.exp(-(q - C0) / 2))


def calib():
    with Pool(NPROC) as pool:
        res = pool.map(calib_class, range(N_CH))
    json.dump(res, open("screening_722_calib.json", "w"))
    N0 = np.array([r[0] for r in res])
    print(f"calib: <N(c0)> from {N0.min():.2f} to {N0.max():.2f}", flush=True)


def single():
    N0 = np.array([r[0] for r in json.load(open("screening_722_calib.json"))])
    rng = np.random.default_rng(SEED)
    q = np.array([qmax_and_up(i, rng.poisson(expected(i, MU[i]))[None].astype(float))[0][0] for i in range(N_CH)])
    p = p_gv(q, N0)
    dec = decisions_generic(p[None])
    sel = {k: [int(i) + 1 for i in np.flatnonzero(v[0])] for k, v in dec.items()}
    print("single: selected", sel)
    json.dump(dict(q=q.tolist(), p=p.tolist(), selected=sel), open("screening_722_single.json", "w"), indent=1)


def ensemble():
    N0 = np.array([r[0] for r in json.load(open("screening_722_calib.json"))])
    with Pool(NPROC) as pool:
        qhat = np.stack(pool.map(ensemble_class, range(N_CH)), axis=1)      # (N_REP, N_CH)
    p = p_gv(qhat, N0[None])
    is_sig = Z_A > 0
    out = {}
    for name, rej in decisions_generic(p).items():
        V, R = rej[:, ~is_sig].sum(1), rej.sum(1)
        S = rej[:, is_sig].sum(1)
        fdp = np.where(R > 0, V / np.maximum(R, 1), 0.0)
        out[name] = dict(fwer=float(np.mean(V > 0)), fdr=float(fdp.mean()), mean_false=float(V.mean()),
                         mean_true=float(S.mean()), ident={int(i) + 1: float(rej[:, i].mean()) for i in SIG_IDX})
        print(f"  {name:10s} FWER={out[name]['fwer']:.3f} FDR={out[name]['fdr']:.3f} "
              f"<false>={V.mean():.2f} <true>={S.mean():.2f}", flush=True)
    np.savez_compressed("screening_722_ensemble.npz", qhat=qhat, p=p)
    json.dump(dict(n_rep=N_REP, results=out), open("screening_722_ensemble.json", "w"), indent=1)


STYLE = {"None": ("black", ":", "o", "No correction"), "Bonferroni": ("royalblue", "-", "s", "Bonferroni"),
         "Sidak": ("#e6a800", "--", "D", "Šidák"), "BH": ("forestgreen", "-", "^", "BH")}


def plot():
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 1.0, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
        "ytick.minor.visible": True, "mathtext.fontset": "cm"})
    is_sig = Z_A > 0
    # (a) one pseudo-experiment: sorted p-values and thresholds
    sg = json.load(open("screening_722_single.json"))
    p = np.array(sg["p"])
    order = np.argsort(p)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    k = np.arange(1, N_CH + 1)
    ax.axhline(ALPHA, color="black", ls=":", lw=1.4, label=r"No correction, $\alpha$")
    ax.axhline(ALPHA / N_CH, color="royalblue", ls="-", lw=1.8, label=r"Bonferroni, $\alpha/N_\mathrm{ch}$")
    ax.axhline(1 - (1 - ALPHA) ** (1 / N_CH), color="#e6a800", ls="--", lw=1.8,
               label=r"Šidák, $1-(1-\alpha)^{1/N_\mathrm{ch}}$")
    ax.plot(k, k * ALPHA / N_CH, "-", color="forestgreen", lw=1.8, label=r"BH, $k\alpha/N_\mathrm{ch}$")
    for r, i in enumerate(order):
        s = is_sig[i]
        ax.plot(r + 1, p[i], "*" if s else "o", ms=13 if s else 6, color="crimson" if s else "0.6",
                mec="black", mew=0.6, zorder=5)
    ax.plot([], [], "*", ms=11, color="crimson", mec="black", mew=0.6, ls="none", label="Injected signal")
    ax.plot([], [], "o", ms=6, color="0.6", mec="black", mew=0.6, ls="none", label="Background only")
    ax.set_yscale("log"); ax.set_xlim(0, N_CH + 1); ax.set_ylim(max(p.min() / 5, 1e-8), 3)
    ax.set_xlabel(r"Rank $k$"); ax.set_ylabel(r"GV-corrected $p$-value")
    leg = ax.legend(frameon=True, loc="lower right", fontsize=10); leg.get_frame().set_edgecolor("0.6")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/screening_thresholds.{ext}", dpi=200)
    plt.close(fig)
    # (b) identification rate vs expected significance, over the pseudo-experiments
    ens = json.load(open("screening_722_ensemble.json"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    z = Z_A[SIG_IDX]
    o = np.argsort(z)
    for name, (c, ls, mk, lab) in STYLE.items():
        ide = np.array([ens["results"][name]["ident"][str(int(i) + 1)] for i in SIG_IDX])
        ax.plot(z[o], ide[o], ls=ls, marker=mk, color=c, lw=1.6, ms=5, label=lab,
                mfc="white" if name == "Sidak" else c)
    ax.set_xlabel(r"Expected local significance of the class [$\sigma$]")
    ax.set_ylabel("Identification rate")
    ax.set_ylim(0, 1.02)
    leg = ax.legend(frameon=True, loc="lower right"); leg.get_frame().set_edgecolor("0.6")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/screening_power.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["calib", "single", "ensemble", "plot"]
    print("signal classes:", (SIG_IDX + 1).tolist(), " Z_A:", Z_A[SIG_IDX].round(2).tolist())
    print("mu:", MU[SIG_IDX].round(5).tolist())
    for s in steps:
        globals()[s]()
