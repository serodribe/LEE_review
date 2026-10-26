#!/usr/bin/env python3
"""Section 7.2.1: a resonance decaying into six channels with different detector resolutions.

Design (fixed in advance):
  * each channel i has its own background slope nu_i and detector resolution sigma_i (>= bin width);
  * the resonance has mass M_TRUE and a small intrinsic width GAMMA << sigma_i; its observed shape in
    channel i is a Gaussian of width sqrt(GAMMA^2 + sigma_i^2) (intrinsic width smeared by the resolution);
  * each channel is fitted with a Gaussian template of width sigma_i (the resolution of that channel);
  * the signal is injected in channels 1, 3, 6 with strengths giving an expected (Asimov) local significance
    Z_A = 4.5, 3.5, 4.5 at the true mass; channels 2, 4, 5 are background only;
  * the mass scan of each channel is corrected with GV (c0 = 0.5); <N(c0)> is calibrated once per channel
    with N_CAL background-only toys generated with that channel's slope and fitted with its template;
  * the channel p-values are corrected with Bonferroni, Sidak and BH at alpha = 0.05.

Steps (independent random streams, so that changing one step does not change the others):
  calib     : <N(c0)> per channel                                   (seed SEED + 1)
  single    : one pseudo-experiment, shown in Table 2 and the figure  (seed SEED)
  ensemble  : N_REP pseudo-experiments -> FWER, FDR, power per procedure (seed SEED + 2)
Usage: python multichannel_721.py [calib] [single] [ensemble] [plot]   (default: calib single ensemble)
"""
import json
import sys
from math import comb

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import bs_trials_check as b

SEED = 20261025
M_TRUE, GAMMA = 3.0, 0.02
NU = [0.50, 0.30, 0.40, 0.20, 0.60, 0.35]
SIG = [0.15, 0.25, 0.20, 0.40, 0.30, 0.35]
Z_A = [4.5, 0.0, 3.5, 0.0, 0.0, 4.5]
C0, ALPHA = 0.5, 0.05
N_CAL, N_REP, BATCH = 5000, 5000, 100
m_grid = np.linspace(b.XMIN, b.XMAX, 350)
NCH = len(NU)


def gauss_trunc(x, m, s):
    norm = stats.norm.cdf(b.XMAX, m, s) - stats.norm.cdf(b.XMIN, m, s)
    return stats.norm.pdf(x, m, s) / norm


def expected(nu, mu, sig_eff):
    bb = b.bkg_and_derivs(np.array([nu]))[0][0]
    return b.N_EVENTS * b.DX * ((1 - mu) * bb + mu * gauss_trunc(b.X, M_TRUE, sig_eff))


def mu_for_asimov(nu, s, z):
    """Signal fraction giving an Asimov local significance z at the true mass (bisection)."""
    if z <= 0:
        return 0.0
    seff, S1 = np.hypot(GAMMA, s), b.sig_shapes(np.array([M_TRUE]), s)
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if b.q_scan(expected(nu, mid, seff)[None], S1)[0][0, 0] < z**2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


MU = [mu_for_asimov(nu, s, z) for nu, s, z in zip(NU, SIG, Z_A)]
SHAPES = [b.sig_shapes(m_grid, s) for s in SIG]


def scan(n, i):
    """q(m) for a batch of datasets n (B, K) of channel i."""
    return b.q_scan(n, SHAPES[i], n_iter=12)[0]


def p_gv(q_hat, N0):
    return np.minimum(1.0, 0.5 * stats.chi2.sf(q_hat, 1) + N0 * np.exp(-(q_hat - C0) / 2))


def decisions(p):
    """p: (R, NCH) channel p-values -> dict of boolean rejection arrays (R, NCH)."""
    R, n = p.shape
    out = {"None": p <= ALPHA, "Bonferroni": p <= ALPHA / n, "Sidak": p <= 1 - (1 - ALPHA) ** (1 / n)}
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    ok = ps <= ALPHA * np.arange(1, n + 1) / n
    k = np.where(ok.any(axis=1), n - np.argmax(ok[:, ::-1], axis=1), 0)      # largest rank passing
    rej_sorted = np.arange(n)[None, :] < k[:, None]
    bh = np.zeros_like(rej_sorted)
    np.put_along_axis(bh, order, rej_sorted, axis=1)
    out["BH"] = bh
    return out


def calib():
    rng = np.random.default_rng(SEED + 1)
    N0 = []
    for i in range(NCH):
        mean0 = expected(NU[i], 0.0, SIG[i])
        up = [b.upcrossings(scan(rng.poisson(mean0, size=(BATCH, mean0.size)).astype(float), i), C0)
              for _ in range(N_CAL // BATCH)]
        up = np.concatenate(up)
        N0.append((float(up.mean()), float(up.std() / np.sqrt(up.size))))
        print(f"  calib ch{i + 1}: <N(c0)> = {N0[-1][0]:.3f} +- {N0[-1][1]:.3f}", flush=True)
    json.dump(N0, open("multichannel_721_calib.json", "w"))


def single():
    N0 = [x[0] for x in json.load(open("multichannel_721_calib.json"))]
    rng = np.random.default_rng(SEED)
    res, curves = [], []
    for i in range(NCH):
        n = rng.poisson(expected(NU[i], MU[i], np.hypot(GAMMA, SIG[i]))).astype(float)
        q = scan(n[None], i)[0]
        j = int(np.argmax(q))
        pl = 0.5 * stats.chi2.sf(q[j], 1)
        pg = float(p_gv(q[j], N0[i]))
        res.append(dict(ch=i + 1, n_ev=int(n.sum()), nu=NU[i], sigma=SIG[i], mu=MU[i], z_asimov=Z_A[i],
                        N0=N0[i], m_hat=float(m_grid[j]), q_hat=float(q[j]), p_local=pl, p_gv=pg, trials=pg / pl))
        curves.append(q)
        print(f"ch{i + 1}: nu={NU[i]} sigma={SIG[i]} mu={MU[i]:.5f} (Z_A={Z_A[i]}) N_ev={int(n.sum())} "
              f"m_hat={m_grid[j]:.2f} q_hat={q[j]:.2f} p_local={pl:.2e} p_GV={pg:.3g} TF={pg / pl:.0f}")
    p = np.array([r["p_gv"] for r in res])
    dec = decisions(p[None])
    sel = {k: [i + 1 for i in range(NCH) if v[0, i]] for k, v in dec.items()}
    print("selected:", sel, " ranks:", [int(o) + 1 for o in np.argsort(p)])
    qc = np.sum(curves, axis=0)
    jc = int(np.argmax(qc))
    p_mix = sum(comb(NCH, k) * 2.0**-NCH * stats.chi2.sf(qc[jc], k) for k in range(1, NCH + 1))
    best = max(res, key=lambda r: r["q_hat"])
    combined = dict(m_hat=float(m_grid[jc]), q=float(qc[jc]), p_local_mixture=float(p_mix),
                    z_local=float(stats.norm.isf(p_mix)), z_naive=float(np.sqrt(qc[jc])),
                    best_ch=best["ch"], z_best_single=float(np.sqrt(best["q_hat"])))
    print("combined:", combined)
    json.dump(dict(seed=SEED, channels=res, selected=sel, combined=combined),
              open("multichannel_721_single.json", "w"), indent=1)

    fmt_p = lambda x: "1" if x >= 0.9995 else (f"{x:.2f}" if x >= 0.01 else
                      "$%.1f\\times10^{%d}$" % (x / 10**np.floor(np.log10(x)), np.floor(np.log10(x))))
    print("\n% Table 2 rows")
    for r in res:
        mu = "0" if r["mu"] == 0 else f"{r['mu']:.4f}"
        print(f"    {r['ch']} & $10^5$ & {r['nu']:.2f} & {r['sigma']:.2f} & {mu} & {r['m_hat']:.2f} & "
              f"{r['q_hat']:.1f} & {fmt_p(r['p_gv'])} \\\\")
    figure(res)


def ensemble():
    N0 = [x[0] for x in json.load(open("multichannel_721_calib.json"))]
    rng = np.random.default_rng(SEED + 2)
    qhat = np.zeros((N_REP, NCH))
    for i in range(NCH):
        mean = expected(NU[i], MU[i], np.hypot(GAMMA, SIG[i]))
        for k in range(N_REP // BATCH):
            qhat[k * BATCH:(k + 1) * BATCH, i] = scan(rng.poisson(mean, size=(BATCH, mean.size)).astype(float), i).max(axis=1)
        print(f"  ensemble ch{i + 1} done", flush=True)
    p = p_gv(qhat, np.array(N0)[None, :])
    is_sig = np.array(Z_A) > 0
    out = {}
    for name, rej in decisions(p).items():
        V = rej[:, ~is_sig].sum(axis=1)
        Rn = rej.sum(axis=1)
        fdp = np.where(Rn > 0, V / np.maximum(Rn, 1), 0.0)
        out[name] = dict(fwer=float(np.mean(V > 0)), fwer_err=float(np.sqrt(np.mean(V > 0) * (1 - np.mean(V > 0)) / N_REP)),
                         fdr=float(fdp.mean()), fdr_err=float(fdp.std() / np.sqrt(N_REP)),
                         power={f"ch{i + 1}": float(rej[:, i].mean()) for i in np.flatnonzero(is_sig)})
        print(f"  {name:10s} FWER={out[name]['fwer']:.4f}+-{out[name]['fwer_err']:.4f}  "
              f"FDR={out[name]['fdr']:.4f}+-{out[name]['fdr_err']:.4f}  power={out[name]['power']}")
    np.savez_compressed("multichannel_721_ensemble.npz", qhat=qhat, p=p)
    json.dump(dict(n_rep=N_REP, seed=SEED + 2, results=out), open("multichannel_721_ensemble.json", "w"), indent=1)


def figure(res):
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 1.0, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.minor.visible": False,
        "ytick.minor.visible": True, "mathtext.fontset": "cm"})
    p = np.array([r["p_gv"] for r in res])
    order = np.argsort(p)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(ALPHA, color="black", ls=":", lw=1.4, label=r"No correction, $\alpha$")
    ax.axhline(ALPHA / NCH, color="royalblue", ls="-", lw=1.8, label=r"Bonferroni, $\alpha/N_\mathrm{ch}$")
    ax.axhline(1 - (1 - ALPHA) ** (1 / NCH), color="#e6a800", ls="--", lw=1.8,
               label=r"Šidák, $1-(1-\alpha)^{1/N_\mathrm{ch}}$")
    rr = np.linspace(0.6, NCH + 0.4, 100)
    ax.plot(rr, rr * ALPHA / NCH, "-", color="forestgreen", lw=1.8, label=r"BH, $k\alpha/N_\mathrm{ch}$")
    for k, i in enumerate(order):
        sig = res[i]["z_asimov"] > 0
        ax.plot(k + 1, p[i], "*" if sig else "o", ms=16 if sig else 9, color="crimson" if sig else "0.6",
                mec="black", mew=0.7, zorder=5)
        ax.annotate(f"ch. {res[i]['ch']}", (k + 1, p[i]), textcoords="offset points", xytext=(10, -3), fontsize=11)
    ax.plot([], [], "*", ms=12, color="crimson", mec="black", mew=0.7, ls="none", label="Injected signal")
    ax.plot([], [], "o", ms=8, color="0.6", mec="black", mew=0.7, ls="none", label="Background only")
    ax.set_yscale("log"); ax.set_xlim(0.5, NCH + 0.9); ax.set_ylim(min(p.min() / 5, 1e-4), 3)
    ax.set_xticks(np.arange(1, NCH + 1)); ax.set_xlabel(r"Rank $k$")
    ax.set_ylabel(r"GV-corrected $p$-value")
    leg = ax.legend(frameon=True, loc="center left", bbox_to_anchor=(1.01, 0.5))
    leg.get_frame().set_edgecolor("0.6")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/multichannel_thresholds.{ext}", dpi=200)


def plot():
    """Redraw the figure from multichannel_721_single.json (no refitting)."""
    figure(json.load(open("multichannel_721_single.json"))["channels"])


if __name__ == "__main__":
    steps = sys.argv[1:] or ["calib", "single", "ensemble"]
    print("signal fractions:", [f"{m:.5f}" for m in MU])
    for s in steps:
        globals()[s]()
