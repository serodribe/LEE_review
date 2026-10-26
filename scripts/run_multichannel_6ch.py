"""
Section 2.1 of the notebook: a resonance decaying into six channels with different detector resolutions.

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
  calib     : <N(c0)> per channel                                        (seed [SEED + 1, channel])
  single    : one pseudo-experiment (Table 2) and the combined fit        (seed SEED)
  ensemble  : N_REP pseudo-experiments -> q_hat and p_GV per channel, FWER, FDR, power   (seed [SEED + 2, channel])
Outputs in output/data/multichannel_6ch/: calib.json, single.json, ensemble.npz, ensemble.json

Usage (from the repository root):
    python scripts/run_multichannel_6ch.py [calib] [single] [ensemble] [--n-jobs N]   (default: all steps)
"""
import argparse
import json
import os
import time
from math import comb

import numpy as np
from scipy import stats

import LEEutils as U

OUT_DIR = os.path.join(U.DATA_DIR, "multichannel_6ch")
SEED = 20261025
M_TRUE, GAMMA = 3.0, 0.02
NU = [0.50, 0.30, 0.40, 0.20, 0.60, 0.35]
SIG = [0.15, 0.25, 0.20, 0.40, 0.30, 0.35]
Z_A = [4.5, 0.0, 3.5, 0.0, 0.0, 4.5]
C0, ALPHA = 0.5, 0.05
N_CAL, N_REP, BATCH = 5000, 5000, 1
N_GRID = 350
NCH = len(NU)

m_grid = np.linspace(U.XMIN, U.XMAX, N_GRID)
MU = [U.mu_for_asimov(nu, s, z, M_TRUE, GAMMA) for nu, s, z in zip(NU, SIG, Z_A)]
SHAPES = [U.sig_shapes(m_grid, s) for s in SIG]


def scan(n, i):
    """q(m) for a batch of datasets n (B, K) of channel i."""
    return U.q_scan(n, SHAPES[i], n_iter=12)[0]


def calib_worker(rng, n, i):
    mean0 = U.expected_channel(NU[i], 0.0, M_TRUE, SIG[i])
    return U.upcrossings(scan(rng.poisson(mean0, size=(n, mean0.size)).astype(float), i), C0)


def ensemble_worker(rng, n, i):
    mean = U.expected_channel(NU[i], MU[i], M_TRUE, np.hypot(GAMMA, SIG[i]))
    return scan(rng.poisson(mean, size=(n, mean.size)).astype(float), i).max(axis=1)


def calib(n_jobs):
    N0 = []
    for i in range(NCH):
        up = np.concatenate(U.run_batches(calib_worker, N_CAL, BATCH, [SEED + 1, i], i, n_jobs=n_jobs))
        N0.append((float(up.mean()), float(up.std() / np.sqrt(up.size))))
        print(f"  calib ch{i + 1}: <N(c0)> = {N0[-1][0]:.3f} +- {N0[-1][1]:.3f}", flush=True)
    json.dump(dict(n_cal=N_CAL, seed=SEED + 1, batch=BATCH, N0=N0), open(os.path.join(OUT_DIR, "calib.json"), "w"), indent=1)


def single():
    N0 = [x[0] for x in json.load(open(os.path.join(OUT_DIR, "calib.json")))["N0"]]
    rng = np.random.default_rng(SEED)
    res, curves = [], []
    for i in range(NCH):
        n = rng.poisson(U.expected_channel(NU[i], MU[i], M_TRUE, np.hypot(GAMMA, SIG[i]))).astype(float)
        q = scan(n[None], i)[0]
        j = int(np.argmax(q))
        pl = U.p_local(q[j])
        pg = float(U.p_gv(q[j], N0[i], C0))
        res.append(dict(ch=i + 1, n_ev=int(n.sum()), nu=NU[i], sigma=SIG[i], mu=MU[i], z_asimov=Z_A[i],
                        N0=N0[i], m_hat=float(m_grid[j]), q_hat=float(q[j]), p_local=pl, p_gv=pg, trials=pg / pl))
        curves.append(q)
        print(f"ch{i + 1}: nu={NU[i]} sigma={SIG[i]} mu={MU[i]:.5f} (Z_A={Z_A[i]}) N_ev={int(n.sum())} "
              f"m_hat={m_grid[j]:.2f} q_hat={q[j]:.2f} p_local={pl:.2e} p_GV={pg:.3g} TF={pg / pl:.0f}")
    p = np.array([r["p_gv"] for r in res])
    dec = U.decisions(p[None], ALPHA)
    sel = {k: [i + 1 for i in range(NCH) if v[0, i]] for k, v in dec.items()}
    print("selected:", sel, " ranks:", [int(o) + 1 for o in np.argsort(p)])
    # combined fit: independent channels, each with its own signal fraction and background, common mass
    qc = np.sum(curves, axis=0)
    jc = int(np.argmax(qc))
    p_mix = sum(comb(NCH, k) * 2.0**-NCH * stats.chi2.sf(qc[jc], k) for k in range(1, NCH + 1))
    best = max(res, key=lambda r: r["q_hat"])
    combined = dict(m_hat=float(m_grid[jc]), q=float(qc[jc]), p_local_mixture=float(p_mix),
                    z_local=float(stats.norm.isf(p_mix)), z_naive=float(np.sqrt(qc[jc])),
                    best_ch=best["ch"], z_best_single=float(np.sqrt(best["q_hat"])))
    print("combined:", combined)
    json.dump(dict(seed=SEED, channels=res, selected=sel, combined=combined),
              open(os.path.join(OUT_DIR, "single.json"), "w"), indent=1)


def ensemble(n_jobs):
    N0 = [x[0] for x in json.load(open(os.path.join(OUT_DIR, "calib.json")))["N0"]]
    qhat = np.zeros((N_REP, NCH))
    for i in range(NCH):
        qhat[:, i] = np.concatenate(U.run_batches(ensemble_worker, N_REP, BATCH, [SEED + 2, i], i, n_jobs=n_jobs))
        print(f"  ensemble ch{i + 1} done", flush=True)
    p = U.p_gv(qhat, np.array(N0)[None, :], C0)
    is_sig = np.array(Z_A) > 0
    out = {}
    for name, rej in U.decisions(p, ALPHA).items():
        V = rej[:, ~is_sig].sum(axis=1)
        Rn = rej.sum(axis=1)
        fdp = np.where(Rn > 0, V / np.maximum(Rn, 1), 0.0)
        out[name] = dict(fwer=float(np.mean(V > 0)), fwer_err=float(np.sqrt(np.mean(V > 0) * (1 - np.mean(V > 0)) / N_REP)),
                         fdr=float(fdp.mean()), fdr_err=float(fdp.std() / np.sqrt(N_REP)),
                         power={f"ch{i + 1}": float(rej[:, i].mean()) for i in np.flatnonzero(is_sig)})
        print(f"  {name:10s} FWER={out[name]['fwer']:.4f}+-{out[name]['fwer_err']:.4f}  "
              f"FDR={out[name]['fdr']:.4f}+-{out[name]['fdr_err']:.4f}  power={out[name]['power']}")
    np.savez_compressed(os.path.join(OUT_DIR, "ensemble.npz"), qhat=qhat, p=p, N0=np.array(N0),
                        sig_channels=np.flatnonzero(is_sig), nu=NU, sigma=SIG, z_asimov=Z_A, mu=MU)
    json.dump(dict(n_rep=N_REP, seed=SEED + 2, batch=BATCH, results=out),
              open(os.path.join(OUT_DIR, "ensemble.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="calib, single, ensemble (default: all)")
    ap.add_argument("--n-jobs", type=int, default=U.N_JOBS)
    a = ap.parse_args()
    steps = a.steps or ["calib", "single", "ensemble"]
    if not set(steps) <= {"calib", "single", "ensemble"}:
        ap.error(f"unknown step in {steps}")
    os.makedirs(OUT_DIR, exist_ok=True)
    print("signal fractions:", [f"{m:.5f}" for m in MU], flush=True)
    for step in steps:
        t0 = time.time()
        if step == "single":
            single()
        else:
            globals()[step](a.n_jobs)
        print(f"[{step}: {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
