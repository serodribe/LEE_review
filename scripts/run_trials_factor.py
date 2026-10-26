"""
Background-only toys of the single-channel bump hunt for Sections 1.1, 1.2 and 1.4 of the notebook:
null distributions of q (Fig. q_null_dist / pval_dist), trials factor at c = 10 (Table 1) and the
Bayer-Seljak estimate (Fig. bs_comparison).

  * fixed mass: q(m0) at m0 = 3 for N_FIXED toys                                  -> q_fixed_m0.npz
  * scanned mass, for each grid of N points (GRIDS) and N_TOYS toys, per toy:      -> trials_factor_<N>.npz
      qmax, m_hat     max_m q(m) on the grid and its position
      up[:, k]        number of upcrossings of level LEVELS[k]
      q_pk, sig_m     Bayer-Seljak inputs at the highest peak: q at the vertex and width sigma_m = sqrt(2/-q''),
                      from a local refit at m_hat + (-H, 0, H), so that they do not depend on the grid spacing
      qS_mps          (grids with >= 300 points) q_S of the peak maximising q_S over all local maxima ('MPS')

Usage (from the repository root):
    python scripts/run_trials_factor.py [--n-jobs N]
"""
import argparse
import os
import time

import numpy as np

import LEEutils as U

OUT_DIR = os.path.join(U.DATA_DIR, "single_channel")
GRIDS = (100, 350)
N_TOYS = 300000
N_FIXED, M0 = 100000, 3.0
LEVELS = np.round(np.arange(0.5, 16.01, 0.5), 2)
H = 0.02
SEED_SCAN, SEED_FIXED = 71, 2026
BATCH, BATCH_FIXED = 1, 1000
MEAN = U.expected_counts(U.LAM, 1.0, 0.0, U.BINS, U.N_EVENTS)


def fixed_worker(rng, n, S0):
    counts = rng.poisson(MEAN, size=(n, MEAN.size)).astype(float)
    q, _, _, g = U.q_scan(counts, S0)
    return q[:, 0], g


def scan_worker(rng, n, m_grid, S, mps):
    counts = rng.poisson(MEAN, size=(n, MEAN.size)).astype(float)
    q, _, _, g = U.q_scan(counts, S)
    j = np.argmax(q, axis=1)
    out = dict(qmax=q[np.arange(n), j], m_hat=m_grid[j],
               up=np.stack([U.upcrossings(q, c) for c in LEVELS], axis=1).astype(np.int16))
    # local refit around the grid maximum
    mm = m_grid[j][:, None] + np.array([-H, 0.0, H])[None, :]
    Sl = U.sig_shapes(mm.ravel()).reshape(n, 3, -1)
    ql, _, _, g2 = U.q_scan(counts, Sl, n_iter=12)
    d2 = (ql[:, 2] - 2 * ql[:, 1] + ql[:, 0]) / H**2
    d1 = (ql[:, 2] - ql[:, 0]) / (2 * H)
    ok = d2 < 0
    out["q_pk"] = np.where(ok, ql[:, 1] - d1**2 / (2 * np.where(ok, d2, -1)), np.nan)
    out["sig_m"] = np.where(ok, np.sqrt(2 / np.where(ok, -d2, 1)), np.nan)
    if mps:
        out["qS_mps"] = U.bayer_seljak(q, m_grid[1] - m_grid[0])["qS"]
    out["grad"] = max(g, g2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=U.N_JOBS)
    ap.add_argument("--n-toys", type=int, default=N_TOYS)
    ap.add_argument("--n-fixed", type=int, default=N_FIXED)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Fixed mass
    t0 = time.time()
    res = U.run_batches(fixed_worker, a.n_fixed, BATCH_FIXED, SEED_FIXED, U.sig_shapes(np.array([M0])), n_jobs=a.n_jobs)
    q0, worst = np.concatenate([r[0] for r in res]), max(r[1] for r in res)
    np.savez_compressed(os.path.join(OUT_DIR, "q_fixed_m0.npz"), q0=q0, m0=M0, seed=SEED_FIXED, batch=BATCH_FIXED,
                        worst_grad=worst)
    print(f"fixed mass: {q0.size} toys in {time.time() - t0:.0f}s, worst |grad| = {worst:.1e}", flush=True)

    # Scanned mass
    for n_grid in GRIDS:
        t0 = time.time()
        m_grid = np.linspace(U.XMIN, U.XMAX, n_grid)
        res = U.run_batches(scan_worker, a.n_toys, BATCH, [SEED_SCAN, n_grid], m_grid, U.sig_shapes(m_grid),
                            n_grid >= 300, n_jobs=a.n_jobs)
        d = {k: np.concatenate([r[k] for r in res]) for k in res[0] if k != "grad"}
        worst = max(r["grad"] for r in res)
        np.savez_compressed(os.path.join(OUT_DIR, f"trials_factor_{n_grid}.npz"), **d, levels=LEVELS, m_grid=m_grid,
                            h=H, seed=[SEED_SCAN, n_grid], batch=BATCH, worst_grad=worst)
        print(f"grid {n_grid}: {d['qmax'].size} toys in {time.time() - t0:.0f}s, worst |grad| = {worst:.1e}", flush=True)


if __name__ == "__main__":
    main()
