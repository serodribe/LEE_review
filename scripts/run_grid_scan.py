"""
Section 1.3 of the notebook (Fig. trial_factor_Gv_Bon): trials factor at c = 10 as a function of the number N
of equally spaced mass points, with the GV procedure (c0 = 0.5 and 7.5) and from the fraction of toys with
max_i q(m_i) >= c (discrete scan). The same background-only toys are analysed on every grid.

Output: grid_scan.npz with, for each toy and grid, the upcrossings of 0.5 and 7.5 and max_i q(m_i).

Usage (from the repository root):
    python scripts/run_grid_scan.py [--n-jobs N]
"""
import argparse
import os
import time

import numpy as np

import LEEutils as U

OUT_DIR = os.path.join(U.DATA_DIR, "single_channel")
N_POINTS = np.arange(11, 200, 10)
C0_LIST = (0.5, 7.5)
N_TOYS = 10000
SEED, BATCH = 2, 1
MEAN = U.expected_counts(U.LAM, 1.0, 0.0, U.BINS, U.N_EVENTS)


def worker(rng, n, shapes):
    counts = rng.poisson(MEAN, size=(n, MEAN.size)).astype(float)
    up = np.zeros((n, len(shapes), len(C0_LIST)), dtype=np.int16)
    qmax = np.zeros((n, len(shapes)))
    for k, S in enumerate(shapes):
        q = U.q_scan(counts, S)[0]
        for l, c0 in enumerate(C0_LIST):
            up[:, k, l] = U.upcrossings(q, c0)
        qmax[:, k] = q.max(axis=1)
    return up, qmax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=U.N_JOBS)
    ap.add_argument("--n-toys", type=int, default=N_TOYS)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "grid_scan.npz"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    t0 = time.time()
    shapes = [U.sig_shapes(np.linspace(U.XMIN, U.XMAX, N)) for N in N_POINTS]
    res = U.run_batches(worker, a.n_toys, BATCH, SEED, shapes, n_jobs=a.n_jobs)
    up = np.concatenate([r[0] for r in res])        # (n_toys, len(N_POINTS), len(C0_LIST))
    qmax = np.concatenate([r[1] for r in res])      # (n_toys, len(N_POINTS))
    np.savez_compressed(a.out, up=up, qmax=qmax, n_points=N_POINTS, c0=C0_LIST, seed=SEED, batch=BATCH)
    print(f"{qmax.shape[0]} toys, {len(N_POINTS)} grids in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
