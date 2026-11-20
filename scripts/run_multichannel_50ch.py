"""
Section 2.2 of the notebook: a scaled-down general search with 50 event classes, a new particle contributing to 10 of them.

Design (fixed in advance, seed DESIGN_SEED):
  * class i has its own background slope nu_i ~ U[0.2, 0.6] and detector resolution sigma_i ~ U[0.15, 0.40];
    it is fitted with a Gaussian template of width sigma_i (as in Section 2.1);
  * a narrow resonance (GAMMA = 0.02) at the common mass M_TRUE = 3 is injected in N_SIG = 10 classes chosen at
    random; its observed shape in class i is a Gaussian of width sqrt(GAMMA^2 + sigma_i^2);
  * signal strengths: expected (Asimov) local significances evenly spaced between 2.5 and 4.5 sigma;
  * the mass scan of each class is corrected with GV (c0 = 0.5), <N(c0)> calibrated per class with N_CAL toys;
  * Bonferroni, Sidak and BH at alpha = 0.05 across the 50 classes.

Steps (independent random streams):
  calib    : <N(c0)> per class                                                        (seed [SEED + 1, class])
  single   : one pseudo-experiment -> q_hat, p_GV and the classes selected by each procedure   (seed SEED)
  ensemble : N_REP pseudo-experiments -> q_hat and p_GV per class, FWER, FDR, identification rates
                                                                                       (seed [SEED + 2, class])
Outputs in output/data/screening_50ch/: calib.json, single.json, ensemble.npz, ensemble.json

Usage (from the repository root):
    python scripts/run_multichannel_50ch.py [calib] [single] [ensemble] [--n-jobs N]   (default: all steps)
"""
import argparse
import json
import os
import time

import numpy as np

import LEEutils as U

OUT_DIR = os.path.join(U.DATA_DIR, "screening_50ch")
DESIGN_SEED, SEED = 722, 20261026
N_CH, N_SIG = 50, 10
M_TRUE, GAMMA = 3.0, 0.02
C0, ALPHA = 0.5, 0.05
N_CAL, N_REP, BATCH = 2000, 3000, 1
N_GRID = 350

m_grid = np.linspace(U.XMIN, U.XMAX, N_GRID)
_d = np.random.default_rng(DESIGN_SEED)
NU = np.round(_d.uniform(0.2, 0.6, N_CH), 3)
SIG = np.round(_d.uniform(0.15, 0.40, N_CH), 3)
SIG_IDX = np.sort(_d.choice(N_CH, N_SIG, replace=False))
Z_A = np.zeros(N_CH)
Z_A[SIG_IDX] = _d.permutation(np.linspace(2.5, 4.5, N_SIG))
MU = np.array([U.mu_for_asimov(NU[i], SIG[i], Z_A[i], M_TRUE, GAMMA) for i in range(N_CH)])


def qmax_and_up(n, S):
    q = U.q_scan(n, S, n_iter=12)[0]
    return q.max(axis=1), U.upcrossings(q, C0)


def calib_worker(rng, n, i, S):
    mean0 = U.expected_channel(NU[i], 0.0, M_TRUE, np.hypot(GAMMA, SIG[i]))
    return qmax_and_up(rng.poisson(mean0, size=(n, mean0.size)).astype(float), S)[1]


def ensemble_worker(rng, n, i, S):
    mean = U.expected_channel(NU[i], MU[i], M_TRUE, np.hypot(GAMMA, SIG[i]))
    return qmax_and_up(rng.poisson(mean, size=(n, mean.size)).astype(float), S)[0]


def calib(n_jobs):
    res = []
    for i in range(N_CH):
        up = np.concatenate(U.run_batches(calib_worker, N_CAL, BATCH, [SEED + 1, i], i, U.sig_shapes(m_grid, SIG[i]),
                                          n_jobs=n_jobs))
        res.append((float(up.mean()), float(up.std() / np.sqrt(up.size))))
    json.dump(dict(n_cal=N_CAL, seed=SEED + 1, batch=BATCH, N0=res), open(os.path.join(OUT_DIR, "calib.json"), "w"), indent=1)
    N0 = np.array([r[0] for r in res])
    print(f"calib: <N(c0)> from {N0.min():.2f} to {N0.max():.2f}", flush=True)


def single(n_jobs):
    N0 = np.array([r[0] for r in json.load(open(os.path.join(OUT_DIR, "calib.json")))["N0"]])
    rng = np.random.default_rng(SEED)
    q = np.array([qmax_and_up(rng.poisson(U.expected_channel(NU[i], MU[i], M_TRUE, np.hypot(GAMMA, SIG[i])))[None].astype(float),
                              U.sig_shapes(m_grid, SIG[i]))[0][0] for i in range(N_CH)])
    p = U.p_gv(q, N0, C0)
    sel = {k: [int(i) + 1 for i in np.flatnonzero(v[0])] for k, v in U.decisions(p[None], ALPHA).items()}
    print("single: selected", sel, flush=True)
    json.dump(dict(seed=SEED, q=q.tolist(), p=p.tolist(), selected=sel), open(os.path.join(OUT_DIR, "single.json"), "w"), indent=1)


def ensemble(n_jobs):
    N0 = np.array([r[0] for r in json.load(open(os.path.join(OUT_DIR, "calib.json")))["N0"]])
    qhat = np.zeros((N_REP, N_CH))
    for i in range(N_CH):
        qhat[:, i] = np.concatenate(U.run_batches(ensemble_worker, N_REP, BATCH, [SEED + 2, i], i,
                                                  U.sig_shapes(m_grid, SIG[i]), n_jobs=n_jobs))
    p = U.p_gv(qhat, N0[None], C0)
    is_sig = Z_A > 0
    out = {}
    for name, rej in U.decisions(p, ALPHA).items():
        V, R = rej[:, ~is_sig].sum(1), rej.sum(1)
        S = rej[:, is_sig].sum(1)
        fdp = np.where(R > 0, V / np.maximum(R, 1), 0.0)
        out[name] = dict(fwer=float(np.mean(V > 0)), fwer_err=float(np.sqrt(np.mean(V > 0) * (1 - np.mean(V > 0)) / N_REP)),
                         fdr=float(fdp.mean()), fdr_err=float(fdp.std() / np.sqrt(N_REP)), mean_false=float(V.mean()),
                         mean_true=float(S.mean()), ident={int(i) + 1: float(rej[:, i].mean()) for i in SIG_IDX})
        print(f"  {name:10s} FWER={out[name]['fwer']:.3f} FDR={out[name]['fdr']:.3f} "
              f"<false>={V.mean():.2f} <true>={S.mean():.2f}", flush=True)
    np.savez_compressed(os.path.join(OUT_DIR, "ensemble.npz"), qhat=qhat, p=p, N0=N0, sig_channels=SIG_IDX,
                        nu=NU, sigma=SIG, z_asimov=Z_A, mu=MU)
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
    print("signal classes:", (SIG_IDX + 1).tolist(), " Z_A:", Z_A[SIG_IDX].round(2).tolist())
    print("mu:", MU[SIG_IDX].round(5).tolist(), flush=True)
    for step in steps:
        t0 = time.time()
        globals()[step](a.n_jobs)
        print(f"[{step}: {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
