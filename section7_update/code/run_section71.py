#!/usr/bin/env python3
"""Single consistent run for Section 7.1 (Table 1 and the Bayer-Seljak figure).

For each grid (100 and 350 equally spaced mass points) and 1e5 background-only toys, stores per toy:
  qmax                  max_m q(m) on the grid
  up[:, k]              number of upcrossings of level LEVELS[k]
  q_pk, sig_m           Bayer-Seljak inputs at the highest peak: q at the vertex and width
                        sigma_m = sqrt(2/-q''), from a local refit at m_hat + (-h, 0, h), h = 0.02,
                        so that they do not depend on the grid spacing
  qS_mps                (350 grid only) q_S of the peak maximising q_S over all local maxima ('MPS')
Output: run71_<N>.npz
"""
import sys
import time

import numpy as np

import bs_trials_check as b

LEVELS = np.round(np.arange(0.5, 16.01, 0.5), 2)
H = 0.02


def run(n_grid, ntoys, seed, batch=100, tag=""):
    rng = np.random.default_rng(seed)
    m_grid = np.linspace(b.XMIN, b.XMAX, n_grid)
    S = b.sig_shapes(m_grid)
    mean = b.N_EVENTS * b.bkg_and_derivs(np.array([b.LAM_TRUE]))[0][0] * b.DX
    out = {k: [] for k in ("qmax", "up", "q_pk", "sig_m", "m_hat", "qS_mps")}
    t0, worst = time.time(), 0.0
    for start in range(0, ntoys, batch):
        nb = min(batch, ntoys - start)
        n = rng.poisson(mean, size=(nb, mean.size)).astype(float)
        q, _, g = b.q_scan(n, S)
        worst = max(worst, g)
        j = np.argmax(q, axis=1)
        out["qmax"].append(q[np.arange(nb), j])
        out["m_hat"].append(m_grid[j])
        out["up"].append(np.stack([b.upcrossings(q, c) for c in LEVELS], axis=1).astype(np.int16))
        # local refit around the grid maximum
        mm = m_grid[j][:, None] + np.array([-H, 0.0, H])[None, :]
        Sl = b.sig_shapes(mm.ravel()).reshape(nb, 3, -1)
        ql, _, g2 = b.q_scan(n, Sl, n_iter=12)
        worst = max(worst, g2)
        d2 = (ql[:, 2] - 2 * ql[:, 1] + ql[:, 0]) / H**2
        d1 = (ql[:, 2] - ql[:, 0]) / (2 * H)
        ok = d2 < 0
        out["q_pk"].append(np.where(ok, ql[:, 1] - d1**2 / (2 * np.where(ok, d2, -1)), np.nan))
        out["sig_m"].append(np.where(ok, np.sqrt(2 / np.where(ok, -d2, 1)), np.nan))
        if n_grid >= 300:
            out["qS_mps"].append(b.bayer_seljak(q, m_grid[1] - m_grid[0])["qS"])
        if (start // batch) % 50 == 0:
            print(f"  grid {n_grid}: {start + nb}/{ntoys}, {time.time() - t0:.0f}s", flush=True)
    res = {k: np.concatenate(v) for k, v in out.items() if v}
    res["levels"] = LEVELS
    np.savez_compressed(f"run71_{n_grid}{tag}.npz", **res)
    print(f"grid {n_grid} done in {time.time() - t0:.0f}s, worst |grad| = {worst:.1e}", flush=True)


if __name__ == "__main__":
    ntoys = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    rep = int(sys.argv[3]) if len(sys.argv) > 3 else 0          # replica index: independent seed
    run(int(sys.argv[1]), ntoys, seed=71 + int(sys.argv[1]) + 1000 * rep, tag=f"_r{rep}" if rep else "")
