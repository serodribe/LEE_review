#!/usr/bin/env python3
"""
Numerical check of the Bayer & Seljak (2020) trials factor on the single-channel
bump hunt of Section 7.1 of the LEE review.

The model is identical to notebooks/experiments.ipynb of the LEE_review repository:
  * x in [0, 10], 69 bins (np.linspace(0, 10, 70)), N = 100k events per toy
  * background: truncated exponential with free rate lam (true value 0.5)
  * signal: truncated Gaussian of fixed width 0.2, location m scanned on 350 points
  * binned Poisson NLL with nu_i = N f(x_i) dx, N = sum_i n_i
  * q(m) = 2 [NLL_0 - NLL_1(m)], with mu >= 0 (q = 0 when mu_hat < 0), Eq. (19)

The fits are done with a vectorised Newton iteration (all toys in a batch and all
mass points at once), which is ~1000x faster than calling scipy per point.
Use --validate to compare it with the notebook's scipy/L-BFGS-B fit.

At the level c (default 10) the script compares four trials-factor estimates,
  1. fully empirical : P(max_m q >= c) / p_local
  2. Davies          : 1 + <N(c)> / p_local, upcrossings counted directly at c
  3. Gross-Vitells   : 1 + <N(c0)> exp(-(c-c0)/2) / p_local, with c0 = 0.5
  4. Bayer-Seljak    : N_B = (U-L) / (sqrt(2 pi) sigma_m), with sigma_m from the
                       curvature of q(m) at the peak, -q''/2 = 1/sigma_m^2,
and performs two further checks:
  5. the identity sigma_m = 1 / (Z sqrt(lambda_2)), i.e. that N_B equals the
     leading Davies term (lambda_2 = -rho_11 measured from upcrossings at c0)
  6. the toy-by-toy calibration of the B&S global p-value
     p_BS = 1 - exp(-exp(-q_S/2)), q_S = q - 2 ln N_B + ln(2 pi q).

Usage:
  python bs_trials_check.py --ntoys 10000 --level 10
  python bs_trials_check.py --validate
"""
import argparse
import time

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# Model (same as the review notebook)
# ----------------------------------------------------------------------------
XMIN, XMAX = 0.0, 10.0
SIGMA_SIG = 0.2
LAM_TRUE = 0.5
N_EVENTS = 100_000
EDGES = np.linspace(XMIN, XMAX, 70)
X = 0.5 * (EDGES[1:] + EDGES[:-1])          # bin centres, K = 69
DX = EDGES[1] - EDGES[0]
RANGE = XMAX - XMIN                          # prior volume of the location


def sig_shapes(m_grid, sigma=SIGMA_SIG):
    """Truncated-Gaussian pdf at the bin centres, shape (M, K)."""
    m = m_grid[:, None]
    norm = stats.norm.cdf(XMAX, m, sigma) - stats.norm.cdf(XMIN, m, sigma)
    return stats.norm.pdf(X[None, :], m, sigma) / np.maximum(norm, 1e-12)


def bkg_and_derivs(lam):
    """Truncated-exponential pdf at the bin centres and its first two lam-derivatives.

    lam has any shape S; returns arrays of shape S + (K,).
    b = lam exp(-lam x) / D,  D = 1 - exp(-lam (XMAX - XMIN))   (XMIN = 0)
    """
    lam = lam[..., None]
    E = np.exp(-lam * XMAX)
    D = 1.0 - E
    b = lam * np.exp(-lam * X) / D
    g1 = 1.0 / lam - X - XMAX * E / D                     # d ln b / d lam
    g2 = -1.0 / lam**2 + XMAX**2 * E / D**2               # d g1 / d lam
    return b, b * g1, b * (g1**2 + g2)


def nll(n, nu):
    return np.sum(nu - n * np.log(nu), axis=-1)


# ----------------------------------------------------------------------------
# Vectorised profile-likelihood fits
# ----------------------------------------------------------------------------
def fit_null(n, n_iter=12):
    """Background-only fit, Newton in lam. n: (B, K). Returns (nll0, lam0)."""
    c = n.sum(axis=1, keepdims=True) * DX
    lam = np.full(n.shape[0], LAM_TRUE)
    for _ in range(n_iter):
        b, b1, b2 = bkg_and_derivs(lam)
        nu, d1, d2 = c * b, c * b1, c * b2
        r, w = 1.0 - n / nu, n / nu**2
        g = np.sum(r * d1, axis=1)
        h = np.sum(r * d2 + w * d1**2, axis=1)
        lam = lam - g / h
    b, _, _ = bkg_and_derivs(lam)
    return nll(n, c * b), lam


def q_scan(n, s, n_iter=8):
    """q(m) for a batch of toys. n: (B, K) counts, s: (M, K) signal shapes.

    Newton iteration in (lam, mu), unconstrained in mu; points with mu_hat < 0
    are set to q = 0, which is what the bounded fit (mu >= 0) of Eq. (19) gives.
    Returns q (B, M), mu_hat (B, M), max |gradient| (convergence diagnostic).
    """
    nll0, lam0 = fit_null(n)
    B, M = n.shape[0], s.shape[-2]
    c = (n.sum(axis=1) * DX)[:, None, None]                # (B, 1, 1)
    nn = n[:, None, :]                                      # (B, 1, K)
    ss = s[None, :, :] if s.ndim == 2 else s                # (1, M, K) or per-toy (B, M, K)
    lam = np.repeat(lam0[:, None], M, axis=1)               # (B, M)
    mu = np.zeros((B, M))
    for _ in range(n_iter):
        b, b1, b2 = bkg_and_derivs(lam)
        mm = mu[..., None]
        nu = c * ((1.0 - mm) * b + mm * ss)
        d_l = c * (1.0 - mm) * b1
        d_m = c * (ss - b)
        d_ll = c * (1.0 - mm) * b2
        d_lm = -c * b1
        r, w = 1.0 - nn / nu, nn / nu**2
        g_l = np.sum(r * d_l, axis=-1)
        g_m = np.sum(r * d_m, axis=-1)
        h_ll = np.sum(r * d_ll + w * d_l**2, axis=-1)
        h_lm = np.sum(r * d_lm + w * d_l * d_m, axis=-1)
        h_mm = np.sum(w * d_m**2, axis=-1)
        det = h_ll * h_mm - h_lm**2
        lam = lam - (h_mm * g_l - h_lm * g_m) / det
        mu = mu - (h_ll * g_m - h_lm * g_l) / det
    b, _, _ = bkg_and_derivs(lam)
    mm = mu[..., None]
    nu = c * ((1.0 - mm) * b + mm * ss)
    q = 2.0 * (nll0[:, None] - nll(nn, nu))
    q = np.where(mu > 0, np.maximum(q, 0.0), 0.0)
    gmax = np.max(np.abs(np.stack([g_l, g_m])))
    return q, mu, gmax


# ----------------------------------------------------------------------------
# Per-toy analysis of the q(m) curve
# ----------------------------------------------------------------------------
def upcrossings(q, level):
    return np.sum((q[:, :-1] < level) & (q[:, 1:] >= level), axis=1)


def curvature(q, j, h):
    """q'' at index j from a 5-point least-squares parabola (Savitzky-Golay).
    The window is shifted inwards at the edges of the grid."""
    M = q.shape[-1]
    j0 = np.clip(j, 2, M - 3)
    idx = j0[:, None] + np.arange(-2, 3)[None, :]
    qq = np.take_along_axis(q, idx, axis=1)
    return (2 * qq[:, 0] - qq[:, 1] - 2 * qq[:, 2] - qq[:, 3] + 2 * qq[:, 4]) / (7 * h**2)


def bayer_seljak(q, h):
    """For each toy, scan all local maxima of q(m), compute sigma_m, N_B and q_S
    (one-sided, t = 1), and keep the peak with the largest q_S (the 'MPS' peak).
    Also returns sigma_m at the highest peak (the 'MAP' peak)."""
    B, M = q.shape
    qs_best = np.full(B, -np.inf)
    out = dict(q=np.zeros(B), sig_m=np.full(B, np.nan), NB=np.full(B, np.nan), edge=np.zeros(B, bool))
    is_max = np.zeros_like(q, dtype=bool)
    is_max[:, 1:-1] = (q[:, 1:-1] > q[:, :-2]) & (q[:, 1:-1] >= q[:, 2:])
    is_max[:, 0] = q[:, 0] > q[:, 1]
    is_max[:, -1] = q[:, -1] > q[:, -2]
    is_max &= q > 1e-6
    rows, cols = np.nonzero(is_max)
    qpk = q[rows, cols]
    qpp = curvature(q[rows], cols, h)
    good = qpp < 0
    sig_m = np.where(good, np.sqrt(2.0 / np.where(good, -qpp, 1.0)), np.nan)
    NB = RANGE / (np.sqrt(2 * np.pi) * sig_m)
    qS = np.where(good, qpk - 2 * np.log(NB) + np.log(2 * np.pi * qpk), -np.inf)
    for r_, c_, qp_, sm_, nb_, qs_ in zip(rows, cols, qpk, sig_m, NB, qS):
        if qs_ > qs_best[r_]:
            qs_best[r_] = qs_
            out["q"][r_], out["sig_m"][r_], out["NB"][r_] = qp_, sm_, nb_
            out["edge"][r_] = (c_ < 2) or (c_ > M - 3)
    out["qS"] = qs_best
    # sigma_m at the global maximum of q (MAP peak)
    jmax = np.argmax(q, axis=1)
    qpp_map = curvature(q, jmax, h)
    out["sig_m_map"] = np.where(qpp_map < 0, np.sqrt(2.0 / np.abs(qpp_map)), np.nan)
    out["edge_map"] = (jmax < 2) | (jmax > M - 3)
    return out


# ----------------------------------------------------------------------------
# Validation against the notebook implementation
# ----------------------------------------------------------------------------
def validate(n_toys=3, seed=7):
    from scipy.optimize import minimize, minimize_scalar

    def expected(lam, m, mu, n_tot):
        b = lam * np.exp(-lam * X) / (1 - np.exp(-lam * XMAX))
        norm = stats.norm.cdf(XMAX, m, SIGMA_SIG) - stats.norm.cdf(XMIN, m, SIGMA_SIG)
        sgl = stats.norm.pdf(X, m, SIGMA_SIG) / max(norm, 1e-12)
        return n_tot * ((1 - mu) * b + mu * sgl) * DX

    def nll_np(cnt, e):
        e = np.clip(e, 1e-9, None)
        return np.sum(e - cnt * np.log(e))

    rng = np.random.default_rng(seed)
    m_grid = np.linspace(XMIN, XMAX, 60)
    s = sig_shapes(m_grid)
    mean = N_EVENTS * bkg_and_derivs(np.array([LAM_TRUE]))[0][0] * DX
    for t in range(n_toys):
        n = rng.poisson(mean)
        q_vec, _, gmax = q_scan(n[None, :].astype(float), s)
        n_tot = n.sum()
        r0 = minimize_scalar(lambda l: nll_np(n, expected(l, 1.0, 0.0, n_tot)), bounds=(0.01, 5),
                             method="bounded", options={"xatol": 1e-12})
        q_ref = []
        for m in m_grid:
            # tight tolerances: with scipy's defaults (as in the notebook) q is only
            # accurate to a few 1e-2, which is the residual difference you would see
            r1 = minimize(lambda p: nll_np(n, expected(p[0], m, p[1], n_tot)), x0=[r0.x, 0.0],
                          bounds=[(0.01, 5.0), (0.0, 1.0)], method="L-BFGS-B",
                          options={"ftol": 1e-15, "gtol": 1e-10})
            q_ref.append(2 * (r0.fun - r1.fun))
        q_ref = np.maximum(np.array(q_ref), 0)
        d = np.abs(q_vec[0] - q_ref)
        print(f"toy {t}: max q = {q_ref.max():.3f} (scipy) vs {q_vec.max():.3f} (Newton); "
              f"max |dq| = {d.max():.2e}; Newton max|grad| = {gmax:.1e}")


# ----------------------------------------------------------------------------
# Main study
# ----------------------------------------------------------------------------
def run(n_toys, n_grid, level, c0, batch, seed, plot):
    rng = np.random.default_rng(seed)
    m_grid = np.linspace(XMIN, XMAX, n_grid)
    h = m_grid[1] - m_grid[0]
    s = sig_shapes(m_grid)
    mean = N_EVENTS * bkg_and_derivs(np.array([LAM_TRUE]))[0][0] * DX

    qmax, up_c, up_c0 = [], [], []
    bs = {k: [] for k in ("q", "qS", "sig_m", "NB", "edge", "sig_m_map", "edge_map")}
    t0, worst_grad = time.time(), 0.0
    for start in range(0, n_toys, batch):
        nb = min(batch, n_toys - start)
        n = rng.poisson(mean, size=(nb, mean.size)).astype(float)
        q, _, gmax = q_scan(n, s)
        worst_grad = max(worst_grad, gmax)
        qmax.append(q.max(axis=1))
        up_c.append(upcrossings(q, level))
        up_c0.append(upcrossings(q, c0))
        o = bayer_seljak(q, h)
        for k in bs:
            bs[k].append(o[k])
        if (start // batch) % 10 == 0:
            print(f"  {start + nb}/{n_toys} toys, {time.time() - t0:.0f}s", flush=True)
    qmax = np.concatenate(qmax)
    up_c, up_c0 = np.concatenate(up_c), np.concatenate(up_c0)
    bs = {k: np.concatenate(v) for k, v in bs.items()}
    print(f"done in {time.time() - t0:.0f}s; worst Newton |gradient| = {worst_grad:.1e}\n")

    Zc = np.sqrt(level)
    p_loc = 0.5 * stats.chi2.sf(level, 1)                  # exact one-sided local p-value
    p_loc_asym = np.exp(-level / 2) / np.sqrt(2 * np.pi * level)  # B&S asymptotic form (t = 1)

    # 1-3: empirical, Davies (direct upcrossings at c), GV extrapolation from c0
    k = np.count_nonzero(qmax >= level)
    tf_emp = k / n_toys / p_loc
    tf_emp_err = np.sqrt(k * (1 - k / n_toys)) / n_toys / p_loc
    tf_dav = 1 + up_c.mean() / p_loc
    tf_dav_err = up_c.std() / np.sqrt(n_toys) / p_loc
    Nc0 = up_c0.mean()
    tf_gv = 1 + Nc0 * np.exp(-(level - c0) / 2) / p_loc
    tf_gv_err = up_c0.std() / np.sqrt(n_toys) * np.exp(-(level - c0) / 2) / p_loc

    # 4: Bayer-Seljak N_B at q = c. sigma_m * sqrt(q) ~ const = 1/sqrt(lambda_2) for high
    # peaks; estimate it from interior MAP peaks with q in [c-2, c+2] and rescale to q = c.
    sel = (np.abs(qmax - level) < 2) & ~bs["edge_map"] & np.isfinite(bs["sig_m_map"])
    kappa = bs["sig_m_map"][sel] * np.sqrt(qmax[sel])
    kap, kap_err = kappa.mean(), kappa.std() / np.sqrt(sel.sum())
    NB_c = RANGE * Zc / (np.sqrt(2 * np.pi) * kap)
    NB_c_err = NB_c * kap_err / kap
    p_bs = 1 - np.exp(-NB_c * p_loc_asym)                  # B&S global p-value at q = c
    tf_bs = p_bs / p_loc
    tf_bs_err = tf_bs * NB_c_err / NB_c

    # 5: identity sigma_m = 1/(Z sqrt(lambda_2)).  Rice/Davies: <N(c0)> = L sqrt(l2)/(2 pi) e^{-c0/2}
    sqrt_l2_upx = 2 * np.pi * Nc0 * np.exp(c0 / 2) / RANGE
    sqrt_l2_upx_err = 2 * np.pi * up_c0.std() / np.sqrt(n_toys) * np.exp(c0 / 2) / RANGE
    sig_eff = np.sqrt(SIGMA_SIG**2 + DX**2 / 12)            # template width incl. binning
    sqrt_l2_th = 1 / (np.sqrt(2) * sig_eff)                  # flat-background approximation

    # 6: toy-by-toy calibration of p_BS against the empirical global p-value
    p_bs_toy = 1 - np.exp(-np.exp(-bs["qS"] / 2))
    order = np.sort(qmax)
    p_emp_toy = 1 - np.searchsorted(order, qmax, side="left") / n_toys

    print(f"=== Trials factor at c = {level:g} (Z_local = {Zc:.2f}), {n_toys} toys, {n_grid}-point grid ===")
    print(f"  p_local (exact, 1/2 chi2_1)            = {p_loc:.3e}")
    print(f"  1. fully empirical                     = {tf_emp:6.2f} +- {tf_emp_err:.2f}   ({k} toys above c)")
    print(f"  2. Davies, upcrossings at c            = {tf_dav:6.2f} +- {tf_dav_err:.2f}")
    print(f"  3. Gross-Vitells, c0 = {c0:g}             = {tf_gv:6.2f} +- {tf_gv_err:.2f}")
    print(f"  4. Bayer-Seljak, 1-exp(-N_B p_asym)/p  = {tf_bs:6.2f} +- {tf_bs_err:.2f}")
    print(f"     N_B at q = c                        = {NB_c:6.2f} +- {NB_c_err:.2f}"
          f"   (from {sel.sum()} peaks with |q-c|<2)")
    print()
    print("=== Identity sigma_m = 1/(Z sqrt(lambda_2)) ===")
    print(f"  sqrt(lambda_2) from peak curvature, 1/<sigma_m Z> = {1 / kap:.3f} +- {kap_err / kap**2:.3f}")
    print(f"  sqrt(lambda_2) from upcrossings at c0 (Rice)      = {sqrt_l2_upx:.3f} +- {sqrt_l2_upx_err:.3f}")
    print(f"  sqrt(lambda_2) flat-background approximation     = {sqrt_l2_th:.3f}")
    print()
    print("=== Calibration of the B&S global p-value (MPS peak, t = 1) ===")
    print(f"  selected peak at grid edge in {bs['edge'].mean() * 100:.1f}% of toys")
    print("   alpha    P(p_BS <= alpha)    P(p_emp <= alpha)")
    for a in (0.5, 0.3, 0.1, 0.05, 0.02, 0.01):
        print(f"   {a:5.2f}      {np.mean(p_bs_toy <= a):.4f}             {np.mean(p_emp_toy <= a):.4f}")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
        qq = np.linspace(-5, 22, 400)
        xs = np.sort(qmax)
        ax[0].step(xs, 1 - np.arange(n_toys) / n_toys, where="post", color="C0",
                   label=r"$\hat q_L=\max_m q(m)$ (toys)")
        xs = np.sort(bs["qS"][np.isfinite(bs["qS"])])
        ax[0].step(xs, 1 - np.arange(xs.size) / n_toys, where="post", color="C3",
                   label=r"$\hat q_S$ (toys)")
        ax[0].plot(qq, 1 - np.exp(-np.exp(-qq / 2)), "k--", label=r"$1-\exp(-e^{-q/2})$")
        ax[0].plot(qq[qq > 0], 0.5 * stats.chi2.sf(qq[qq > 0], 1), ":", color="0.5",
                   label=r"$\frac{1}{2}P(\chi^2_1>q)$ (no LEE)")
        ax[0].set_yscale("log"); ax[0].set_ylim(1 / n_toys, 1.5); ax[0].set_xlim(-5, 22)
        ax[0].set_xlabel("test statistic"); ax[0].set_ylabel("P(Q ≥ q | H0)")
        ax[0].legend(fontsize=9)
        ax[1].loglog(p_emp_toy, p_bs_toy, ".", ms=2, alpha=0.4)
        ax[1].plot([1 / n_toys, 1], [1 / n_toys, 1], "k--", lw=1)
        ax[1].set_xlabel("empirical global p-value (toys)")
        ax[1].set_ylabel("Bayer–Seljak global p-value")
        ax[1].set_xlim(1 / n_toys, 1.2); ax[1].set_ylim(1 / n_toys, 1.2)
        fig.tight_layout()
        fig.savefig(plot, dpi=150)
        print(f"\nplot saved to {plot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ntoys", type=int, default=10000)
    ap.add_argument("--grid", type=int, default=350, help="number of scanned mass points")
    ap.add_argument("--level", type=float, default=10.0, help="level c at which trials factors are compared")
    ap.add_argument("--c0", type=float, default=0.5, help="GV reference level")
    ap.add_argument("--batch", type=int, default=50, help="toys fitted simultaneously")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--plot", default="bs_check.png", help="output plot ('' to skip)")
    ap.add_argument("--validate", action="store_true", help="compare the fitter with the notebook's scipy fit")
    a = ap.parse_args()
    if a.validate:
        validate()
    else:
        run(a.ntoys, a.grid, a.level, a.c0, a.batch, a.seed, a.plot)
