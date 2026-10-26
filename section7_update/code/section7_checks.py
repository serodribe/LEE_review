#!/usr/bin/env python3
"""
Numerical checks for Section 7.1 of the LEE review, using the fast vectorised
fitter of bs_trials_check.py (same model as the LEE_review notebook).

  width   : GV trials factor at c = 10 vs template width, compared with
            Eq. (tf_analytic): 1 + kappa(c) (U-L)/sigma        [fills the red text of 7.1.3]
  grid    : GV trials factor at c = 10 vs number of mass points, c0 = 0.5 and 7.5,
            plus the fully empirical trials factor of the discrete scan   [checks Fig. 3 right]
  table100: Davies / GV / empirical trials factors on the 100-point grid   [checks Table 1]

Usage: python section7_checks.py width|grid|table100 [--ntoys N]
"""
import argparse
import time

import numpy as np
from scipy import stats

import bs_trials_check as b

C, C0 = 10.0, 0.5
P_LOC = 0.5 * stats.chi2.sf(C, 1)
MEAN = b.N_EVENTS * b.bkg_and_derivs(np.array([b.LAM_TRUE]))[0][0] * b.DX


def toys(rng, n):
    return rng.poisson(MEAN, size=(n, MEAN.size)).astype(float)


def kappa(c):
    return np.exp(-c / 2) / (2 * np.sqrt(2) * np.pi * stats.norm.sf(np.sqrt(c)))


def q_fixed_nu(n, S, lam=b.LAM_TRUE, n_iter=10):
    """q(m) with the background slope fixed to its true value (only mu is fitted)."""
    c = (n.sum(axis=1) * b.DX)[:, None, None]
    bb = b.bkg_and_derivs(np.array([lam]))[0][0]
    nn, mu = n[:, None, :], np.zeros((n.shape[0], S.shape[0]))
    nll0 = b.nll(n, c[:, 0] * bb)
    for _ in range(n_iter):
        nu = c * ((1 - mu[..., None]) * bb + mu[..., None] * S[None])
        d = c * (S[None] - bb)
        mu -= np.sum((1 - nn / nu) * d, -1) / np.sum(nn / nu**2 * d**2, -1)
    nu = c * ((1 - mu[..., None]) * bb + mu[..., None] * S[None])
    q = 2 * (nll0[:, None] - b.nll(nn, nu))
    return np.where(mu > 0, np.maximum(q, 0), 0)


def width_fixnu(ntoys, seed=1):
    """Same toys and widths as width(), but with the background slope fixed in the fit."""
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(0.1, 5, 20)
    m_grid = np.linspace(b.XMIN, b.XMAX, 350)
    shapes = {s: b.sig_shapes(m_grid, s) for s in sigmas}
    up = {s: [] for s in sigmas}
    for start in range(0, ntoys, 50):
        n = toys(rng, min(50, ntoys - start))
        for s in sigmas:
            up[s].append(b.upcrossings(q_fixed_nu(n, shapes[s]), C0))
    print(" sigma   <N(c0)>     TF_GV(c=10)   [background slope fixed]")
    for s in sigmas:
        u = np.concatenate(up[s])
        tf = 1 + u.mean() * np.exp(-(C - C0) / 2) / P_LOC
        err = u.std() / np.sqrt(u.size) * np.exp(-(C - C0) / 2) / P_LOC
        print(f" {s:5.2f}  {u.mean():7.3f}   {tf:7.2f} +- {err:5.2f}")


def width(ntoys, seed=1):
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(0.1, 5, 20)
    m_grid = np.linspace(b.XMIN, b.XMAX, 350)
    shapes = {s: b.sig_shapes(m_grid, s) for s in sigmas}
    up = {s: [] for s in sigmas}
    worst = 0.0
    for start in range(0, ntoys, 50):
        n = toys(rng, min(50, ntoys - start))          # same toys for every width
        for s in sigmas:
            q, _, g = b.q_scan(n, shapes[s])
            if g > 1e-4:                                 # rare non-converged fit: redo with more iterations
                q, _, g2 = b.q_scan(n, shapes[s], n_iter=40)
                print(f"  sigma={s:.2f}, toys {start}-{start + len(n)}: |grad| {g:.1e} -> {g2:.1e} after 40 iterations")
                g = g2
            worst = max(worst, g)
            up[s].append(b.upcrossings(q, C0))
    print(f"worst |grad| = {worst:.1e}")
    print(" sigma   <N(c0)>     TF_GV(c=10)        Eq.(tf_analytic)   ratio")
    for s in sigmas:
        u = np.concatenate(up[s])
        tf = 1 + u.mean() * np.exp(-(C - C0) / 2) / P_LOC
        err = u.std() / np.sqrt(u.size) * np.exp(-(C - C0) / 2) / P_LOC
        pred = 1 + kappa(C) * (b.XMAX - b.XMIN) / s
        print(f" {s:5.2f}  {u.mean():7.3f}   {tf:7.2f} +- {err:5.2f}     {pred:7.2f}        {tf / pred:5.2f}")


def grid(ntoys, seed=2):
    rng = np.random.default_rng(seed)
    n_points = np.arange(11, 200, 10)
    grids = {N: np.linspace(b.XMIN, b.XMAX, N) for N in n_points}
    shapes = {N: b.sig_shapes(g) for N, g in grids.items()}
    res = {N: dict(u05=[], u75=[], qmax=[]) for N in n_points}
    for start in range(0, ntoys, 100):
        n = toys(rng, min(100, ntoys - start))          # same toys for every grid
        for N in n_points:
            q, _, _ = b.q_scan(n, shapes[N])
            res[N]["u05"].append(b.upcrossings(q, 0.5))
            res[N]["u75"].append(b.upcrossings(q, 7.5))
            res[N]["qmax"].append(q.max(axis=1))
    print("   N  spacing/sigma   GV c0=0.5        GV c0=7.5         empirical (discrete scan)")
    for N in n_points:
        r = {k: np.concatenate(v) for k, v in res[N].items()}
        out = []
        for key, c0 in (("u05", 0.5), ("u75", 7.5)):
            f = np.exp(-(C - c0) / 2) / P_LOC
            out.append((1 + r[key].mean() * f, r[key].std() / np.sqrt(ntoys) * f))
        k = np.count_nonzero(r["qmax"] >= C)
        emp, emp_err = k / ntoys / P_LOC, np.sqrt(k * (1 - k / ntoys)) / ntoys / P_LOC
        sp = (b.XMAX - b.XMIN) / (N - 1) / b.SIGMA_SIG
        print(f" {N:3d}    {sp:5.2f}      {out[0][0]:6.1f} +- {out[0][1]:4.1f}   "
              f"{out[1][0]:6.1f} +- {out[1][1]:4.1f}    {emp:6.1f} +- {emp_err:4.1f}")


def table100(ntoys, seed=3):
    rng = np.random.default_rng(seed)
    m_grid = np.linspace(b.XMIN, b.XMAX, 100)
    s = b.sig_shapes(m_grid)
    qmax, uc, uc0 = [], [], []
    for start in range(0, ntoys, 200):
        q, _, _ = b.q_scan(toys(rng, min(200, ntoys - start)), s)
        qmax.append(q.max(axis=1)); uc.append(b.upcrossings(q, C)); uc0.append(b.upcrossings(q, C0))
    qmax, uc, uc0 = map(np.concatenate, (qmax, uc, uc0))
    k = np.count_nonzero(qmax >= C)
    f = np.exp(-(C - C0) / 2) / P_LOC
    print(f"100-point grid, {ntoys} toys")
    print(f"  Davies     {1 + uc.mean() / P_LOC:6.2f} +- {uc.std() / np.sqrt(ntoys) / P_LOC:.2f}")
    print(f"  GV c0=0.5  {1 + uc0.mean() * f:6.2f} +- {uc0.std() / np.sqrt(ntoys) * f:.2f}   <N(c0)> = {uc0.mean():.3f}")
    print(f"  empirical  {k / ntoys / P_LOC:6.2f} +- {np.sqrt(k * (1 - k / ntoys)) / ntoys / P_LOC:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["width", "width_fixnu", "grid", "table100"])
    ap.add_argument("--ntoys", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    dict(width=width, width_fixnu=width_fixnu, grid=grid, table100=table100)[a.what](a.ntoys)
    print(f"[{time.time() - t0:.0f}s]")
