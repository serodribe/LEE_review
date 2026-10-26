"""
Section 1.5 of the notebook (Fig. tfactor_vs_signalwidth): GV trials factor at c = 10 as a function of the width
of the signal template used in the fit, with the background slope fitted or fixed to its true value.
The same background-only toys are analysed with every width.

Output: width_scan.npz with, for each toy and width, the upcrossings of c0 = 0.5 (slope fitted and fixed).

Usage (from the repository root):
    python scripts/run_width_scan.py [--n-jobs N]
"""
import argparse
import os
import time

import numpy as np

import LEEutils as U

OUT_DIR = os.path.join(U.DATA_DIR, "single_channel")
SIGMAS = np.unique(np.append(np.linspace(0.1, 5, 20), U.SIGMA_SIG))  # includes the fixed width of 1.2 (sigma = 0.2) as a cross-check
N_GRID = 350
C0 = 0.5
N_TOYS = 1000
SEED, BATCH = 1, 1
MEAN = U.expected_counts(U.LAM, 1.0, 0.0, U.BINS, U.N_EVENTS)


def worker(rng, n, shapes):
    counts = rng.poisson(MEAN, size=(n, MEAN.size)).astype(float)
    up_fit = np.zeros((n, len(shapes)), dtype=np.int16)
    up_fix = np.zeros((n, len(shapes)), dtype=np.int16)
    refits, worst = [], 0.0
    for k, S in enumerate(shapes):
        q, _, _, g = U.q_scan(counts, S)
        if g > 1e-4:                                 # rare non-converged fit: redo with more iterations
            q, _, _, g2 = U.q_scan(counts, S, n_iter=40)
            refits.append((k, g, g2))
            g = g2
        worst = max(worst, g)
        up_fit[:, k] = U.upcrossings(q, C0)
        up_fix[:, k] = U.upcrossings(U.q_fixed_nu(counts, S), C0)
    return up_fit, up_fix, refits, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=U.N_JOBS)
    ap.add_argument("--n-toys", type=int, default=N_TOYS)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    m_grid = np.linspace(U.XMIN, U.XMAX, N_GRID)
    shapes = [U.sig_shapes(m_grid, s) for s in SIGMAS]
    res = U.run_batches(worker, a.n_toys, BATCH, SEED, shapes, n_jobs=a.n_jobs)
    up_fit = np.concatenate([r[0] for r in res])    # (n_toys, len(SIGMAS))
    up_fix = np.concatenate([r[1] for r in res])
    for i, r in enumerate(res):
        for k, g, g2 in r[2]:
            print(f"  sigma={SIGMAS[k]:.2f}, batch {i}: |grad| {g:.1e} -> {g2:.1e} after 40 iterations")
    worst = max(r[3] for r in res)
    np.savez_compressed(os.path.join(OUT_DIR, "width_scan.npz"), up_fit=up_fit, up_fix=up_fix, sigmas=SIGMAS,
                        n_grid=N_GRID, c0=C0, seed=SEED, batch=BATCH, worst_grad=worst)
    print(f"{up_fit.shape[0]} toys, {len(SIGMAS)} widths in {time.time() - t0:.0f}s, worst |grad| = {worst:.1e}", flush=True)


if __name__ == "__main__":
    main()
