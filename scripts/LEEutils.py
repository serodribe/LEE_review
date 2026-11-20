"""
Common settings and functions of the LEE experiments.

  * paths, number of parallel jobs and plot style
  * model of the single-channel bump hunt (truncated exponential + truncated Gaussian)
  * vectorised profile-likelihood fits: q(m) for a batch of toys and all mass points at once, with a
    Newton iteration in (lam, mu) (much faster than calling scipy per mass point)
  * Gross-Vitells, Davies and Bayer-Seljak helpers
  * multi-channel helpers (Asimov signal strength, GV p-value, Bonferroni / Sidak / BH decisions) and plot

The heavy computations are run by the run_*.py scripts in this folder, which save their output in
output/data/; the notebook only loads these files and makes the plots.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import stats

# ----------------------------------------------------------------------
# Paths and parallel jobs
# ----------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(ROOT_DIR, "output", "plots", "classic")
DATA_DIR = os.path.join(ROOT_DIR, "output", "data")
N_JOBS = max(1, os.cpu_count() - 2)

# ----------------------------------------------------------------------
# "Paper-like" style settings
# ----------------------------------------------------------------------
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "mathtext.fontset": "cm",
}

# Common axes margins (figure fractions) for figures placed side by side in the paper
FIG_MARGINS = dict(left=0.15, right=0.97, top=0.97, bottom=0.13)


def set_style():
    plt.style.use('default')
    mpl.rcParams.update(PLOT_STYLE)


def style_legend(leg):
    leg.get_frame().set_edgecolor('0.6')
    leg.get_frame().set_linewidth(0.8)
    return leg


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
XMIN, XMAX = 0.0, 10.0
SIGMA_SIG = 0.2  # fixed signal width
M_SIG = 3.0      # fixed signal mean
MU_SIG = 0.02    # fixed signal fraction
LAM = 0.5        # fixed background decay rate
N_EVENTS = 100000

BINS = np.linspace(XMIN, XMAX, 71)          # bin edges: 70 bins of width 1/7
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
DX = BINS[1] - BINS[0]
RANGE = XMAX - XMIN                         # scan range U - L


def bkg_pdf(x, lam):
    """Truncated exponential pdf on [XMIN, XMAX]."""
    norm = (np.exp(-lam * XMIN) - np.exp(-lam * XMAX))/lam
    return np.exp(-lam * x) / norm


def sig_pdf(x, m, sigma=SIGMA_SIG):
    """Truncated Gaussian pdf on [XMIN, XMAX]."""
    norm = stats.norm.cdf(XMAX, m, sigma) - stats.norm.cdf(XMIN, m, sigma)
    return stats.norm.pdf(x, m, sigma) / max(norm, 1e-12)


def expected_counts(lam, m, mu, edges, n_tot, width=SIGMA_SIG):
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    pdf_vals = (1 - mu) * bkg_pdf(centers, lam) + mu * sig_pdf(centers, m, sigma=width)
    return n_tot * pdf_vals * widths


def sig_shapes(m_grid, sigma=SIGMA_SIG):
    """Truncated-Gaussian pdf at the bin centres for every mass of m_grid, shape (M, K)."""
    m = np.asarray(m_grid)[:, None]
    norm = stats.norm.cdf(XMAX, m, sigma) - stats.norm.cdf(XMIN, m, sigma)
    return stats.norm.pdf(CENTERS[None, :], m, sigma) / np.maximum(norm, 1e-12)


def bkg_and_derivs(lam):
    """Truncated-exponential pdf at the bin centres and its first two lam-derivatives.

    lam has any shape S; returns arrays of shape S + (K,).
    b = lam exp(-lam x) / D,  D = 1 - exp(-lam (XMAX - XMIN))   (XMIN = 0)
    """
    lam = lam[..., None]
    E = np.exp(-lam * XMAX)
    D = 1.0 - E
    b = lam * np.exp(-lam * CENTERS) / D
    g1 = 1.0 / lam - CENTERS - XMAX * E / D               # d ln b / d lam
    g2 = -1.0 / lam**2 + XMAX**2 * E / D**2               # d g1 / d lam
    return b, b * g1, b * (g1**2 + g2)


def nll(n, nu):
    """Poisson negative log-likelihood (binned), dropping constant terms, summed over the last axis."""
    return np.sum(nu - n * np.log(nu), axis=-1)


# ----------------------------------------------------------------------
# Vectorised profile-likelihood fits
# ----------------------------------------------------------------------
def fit_null(n, n_iter=12):
    """Background-only fit, Newton in lam. n: (B, K). Returns (nll0, lam0)."""
    c = n.sum(axis=1, keepdims=True) * DX
    lam = np.full(n.shape[0], LAM)
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
    """q(m) for a batch of toys. n: (B, K) counts, s: (M, K) signal shapes (or per-toy shapes (B, M, K)).

    Newton iteration in (lam, mu), unconstrained in mu; points with mu_hat < 0 are set to q = 0,
    which is what the bounded fit (mu >= 0) gives. The total number of events is fixed to the observed one.
    Returns q (B, M), mu_hat (B, M), lam_hat (B, M) and max |gradient| (convergence diagnostic).
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
    with np.errstate(invalid="ignore"):   # nu <= 0 only for large negative mu_hat, where q is set to 0 below
        q = 2.0 * (nll0[:, None] - nll(nn, nu))
    q = np.where(mu > 0, np.maximum(q, 0.0), 0.0)
    gmax = np.max(np.abs(np.stack([g_l, g_m])))
    return q, mu, lam, gmax


def q_fixed_nu(n, S, lam=LAM, n_iter=10):
    """q(m) with the background slope fixed to its true value (only mu is fitted)."""
    c = (n.sum(axis=1) * DX)[:, None, None]
    bb = bkg_and_derivs(np.array([lam]))[0][0]
    nn, mu = n[:, None, :], np.zeros((n.shape[0], S.shape[0]))
    nll0 = nll(n, c[:, 0] * bb)
    for _ in range(n_iter):
        nu = c * ((1 - mu[..., None]) * bb + mu[..., None] * S[None])
        d = c * (S[None] - bb)
        mu -= np.sum((1 - nn / nu) * d, -1) / np.sum(nn / nu**2 * d**2, -1)
    nu = c * ((1 - mu[..., None]) * bb + mu[..., None] * S[None])
    q = 2 * (nll0[:, None] - nll(nn, nu))
    return np.where(mu > 0, np.maximum(q, 0), 0)


# ----------------------------------------------------------------------
# Per-toy analysis of the q(m) curve
# ----------------------------------------------------------------------
def upcrossings(q, level):
    """Number of upcrossings of level by each q(m) curve, q: (B, M)."""
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


# ----------------------------------------------------------------------
# p-values and trials factors
# ----------------------------------------------------------------------
def p_local(q):
    """One-sided local p-value, q ~ 1/2 chi2_0 + 1/2 chi2_1."""
    return 0.5 * stats.chi2.sf(q, 1)


def p_asym(q):
    """Bayer-Seljak asymptotic form of the local p-value (t = 1)."""
    return np.exp(-q / 2) / np.sqrt(2 * np.pi * q)


def p_gv(q, N0, c0=0.5):
    """GV-corrected global p-value for a scan maximum q, with <N(c0)> = N0."""
    return np.minimum(1.0, p_local(q) + N0 * np.exp(-(q - c0) / 2))


def kappa(c):
    """Prefactor of the narrow-template approximation, p_global/p_local ~ 1 + kappa(c) (U-L)/sigma."""
    return np.exp(-c / 2) / (2 * np.sqrt(2) * np.pi * stats.norm.sf(np.sqrt(c)))


def davies_tf(sigmas, nuisances, c=10.0, lam=LAM):
    """Trials factor 1 + (1/2pi) int sqrt(-rho_11(m)) dm e^{-c/2} / p_local, with rho computed from the
    binned signal templates projected orthogonally to the background directions removed by the fit
    (normalisation: 'norm', slope: 'slope'), inner product weighted by 1/b (Poisson, at the null)."""
    bb, b1, _ = bkg_and_derivs(np.array([lam])); bb, b1 = bb[0], b1[0]
    ip = lambda u, v: np.sum(u * v / bb, -1)
    dirs = [d for k, d in (("norm", bb), ("slope", b1)) if k in nuisances]
    m = np.linspace(XMIN, XMAX, 4001)
    out = []
    for sg in sigmas:
        g = sig_shapes(m, sg)
        if dirs:
            B = np.stack(dirs)
            G = np.array([[ip(x, y) for y in B] for x in B])
            g = g - np.linalg.solve(G, np.stack([ip(g, x) for x in B])).T @ B
        w = g / np.sqrt(ip(g, g))[:, None]
        dw = np.gradient(w, m[1] - m[0], axis=0)
        n_up = np.trapezoid(np.sqrt(ip(dw, dw)), m) / (2 * np.pi) * np.exp(-c / 2)
        out.append(1 + n_up / p_local(c))
    return np.array(out)


# ----------------------------------------------------------------------
# Multi-channel helpers
# ----------------------------------------------------------------------
def gauss_trunc(x, m, s):
    return stats.norm.pdf(x, m, s) / (stats.norm.cdf(XMAX, m, s) - stats.norm.cdf(XMIN, m, s))


def expected_channel(nu, mu, m, width):
    """Expected counts of a channel with background slope nu and a signal fraction mu at mass m, of width `width`."""
    bb = bkg_and_derivs(np.array([nu]))[0][0]
    return N_EVENTS * DX * ((1 - mu) * bb + mu * gauss_trunc(CENTERS, m, width))


def mu_for_asimov(nu, sigma, z, m, gamma):
    """Signal fraction giving an expected (Asimov) local significance z at the true mass m (bisection).
    The observed signal has width sqrt(gamma^2 + sigma^2); the template has width sigma."""
    if z <= 0:
        return 0.0
    seff, S1 = np.hypot(gamma, sigma), sig_shapes(np.array([m]), sigma)
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if q_scan(expected_channel(nu, mid, m, seff)[None], S1)[0][0, 0] < z**2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def decisions(p, alpha=0.05):
    """p: (R, NCH) channel p-values -> dict of boolean rejection arrays (R, NCH)."""
    R, n = p.shape
    out = {"None": p <= alpha, "Bonferroni": p <= alpha / n, "Sidak": p <= 1 - (1 - alpha) ** (1 / n)}
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    ok = ps <= alpha * np.arange(1, n + 1) / n
    k = np.where(ok.any(axis=1), n - np.argmax(ok[:, ::-1], axis=1), 0)      # largest rank passing
    rej_sorted = np.arange(n)[None, :] < k[:, None]
    bh = np.zeros_like(rej_sorted)
    np.put_along_axis(bh, order, rej_sorted, axis=1)
    out["BH"] = bh
    return out


# ----------------------------------------------------------------------
# Parallel toys
# ----------------------------------------------------------------------
def _run_chunk(worker, seeds, sizes, args):
    return [worker(np.random.default_rng(s), n, *args) for s, n in zip(seeds, sizes)]


def run_batches(worker, n_toys, batch, seed, *args, n_jobs=N_JOBS):
    """Run worker(rng, n, *args) on batches of at most `batch` toys, in parallel.

    Batch k uses np.random.default_rng(SeedSequence(seed).spawn(n_batches)[k]), so the results depend
    only on (seed, n_toys, batch) and not on the number of jobs. The batches are sent to the workers in
    chunks, so that the arguments are not copied for every batch. Returns the list of worker outputs, in order.
    """
    sizes = [min(batch, n_toys - start) for start in range(0, n_toys, batch)]
    seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    n_chunks = min(len(sizes), 8 * n_jobs)
    bounds = np.linspace(0, len(sizes), n_chunks + 1).astype(int)
    chunks = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_chunk)(worker, seeds[i0:i1], sizes[i0:i1], args) for i0, i1 in zip(bounds[:-1], bounds[1:]))
    return [out for chunk in chunks for out in chunk]


# ------------------------------------------------------------
# Plot: global p-values vs rank with all multiple-testing thresholds
# ------------------------------------------------------------
def plot_multiple_testing(p_global_toys, sig_channels, fname, alpha=0.05, ylabel='Median GV-corrected $p$-value'):
    """
    Median GV-corrected p-value per channel over the toys (error bars: central 68% interval, 16th-84th percentiles),
    ranked by the median, with no-correction, Bonferroni, Šidák and BH thresholds. With a single toy, its p-values.
    """
    p_global_toys = np.clip(np.atleast_2d(p_global_toys), None, 1.0)  # (N_TOYS, N_CHANNELS)
    N_CHANNELS = p_global_toys.shape[1]
    bkg_mask = np.ones(N_CHANNELS, dtype=bool)
    bkg_mask[sig_channels] = False
    p_global_chan = np.median(p_global_toys, axis=0)
    p_lo_chan, p_hi_chan = np.percentile(p_global_toys, [16, 84], axis=0)
    s_sig, s_bkg, ms_bh = (350, 120, 5) if N_CHANNELS <= 15 else (150, 40, 3)  # smaller markers for many channels

    # Rank channels by median global p-value
    p_global_argsorted = np.argsort(p_global_chan)
    p_global_sorted = p_global_chan[p_global_argsorted]
    p_lo_sorted, p_hi_sorted = p_lo_chan[p_global_argsorted], p_hi_chan[p_global_argsorted]
    ranks = np.arange(1, N_CHANNELS + 1)
    is_sig_sorted = ~bkg_mask[p_global_argsorted]

    # Thresholds on the global p-value
    alpha_bonferroni = alpha / N_CHANNELS
    alpha_sidak = 1 - (1 - alpha) ** (1 / N_CHANNELS)
    bh_thresholds = ranks * alpha / N_CHANNELS  # FDR level of 5%

    # ------------------------------------------------------------------------------------------
    # Plot: all corrections in one figure
    # ------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(alpha, color='black', ls='--', lw=1.3, label=r'No correction ($\alpha$)')
    ax.axhline(alpha_bonferroni, color='#0072B2', ls='-', lw=3.5, label=r'Bonferroni ($\alpha/N_{\rm ch}$)')
    ax.axhline(alpha_sidak, color='#E69F00', ls=':', lw=2.5, label=r'Šidák ($1-(1-\alpha)^{1/N_{\rm ch}}$)')
    ax.plot(ranks, bh_thresholds, '-o', color='#009E73', lw=2, ms=ms_bh, label=r'BH ($k\alpha/N_{\rm ch}$)')

    for mask, color in [(is_sig_sorted, 'crimson'), (~is_sig_sorted, '0.4')]:
        ax.errorbar(ranks[mask], p_global_sorted[mask], yerr=[p_global_sorted[mask] - p_lo_sorted[mask], p_hi_sorted[mask] - p_global_sorted[mask]],
                    fmt='none', ecolor=color, elinewidth=1.2, capsize=3, zorder=4)
    ax.scatter(ranks[is_sig_sorted], p_global_sorted[is_sig_sorted], s=s_sig, marker='*', color='crimson', edgecolor='black', linewidth=0.8, zorder=5, label='Injected signal')
    ax.scatter(ranks[~is_sig_sorted], p_global_sorted[~is_sig_sorted], s=s_bkg, marker='o', color='0.6', edgecolor='0.3', linewidth=0.8, zorder=5, label='Background only')

    ax.set_yscale('log')
    ax.set_xlim(0.5, N_CHANNELS + 0.5)
    ax.set_ylim(min(p_lo_sorted.min(), alpha_bonferroni) / 5, 3)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlabel('Rank $k$')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2.2, labelspacing=1.1)
    fig.tight_layout()

    # Channel labels (only readable for a small number of channels): each label takes the first candidate
    # position that stays inside the axes and does not overlap any line, marker or previously placed label
    if N_CHANNELS <= 15:
        renderer = fig.canvas.get_renderer()
        line_paths = [line.get_transform().transform_path(line.get_path()) for line in ax.lines]
        line_paths += [ax.transData.transform_path(mpl.path.Path([[k, lo], [k, hi]])) for k, lo, hi in zip(ranks, p_lo_sorted, p_hi_sorted)]  # error bars
        pts_disp = ax.transData.transform(np.column_stack([ranks, p_global_sorted]))
        marker_r = np.where(is_sig_sorted, np.sqrt(s_sig), np.sqrt(s_bkg)) / 2 * fig.dpi / 72  # marker radius in pixels
        marker_boxes = [mpl.transforms.Bbox.from_extents(x - r, y - r, x + r, y + r) for (x, y), r in zip(pts_disp, marker_r)]
        # (dx, dy) in points, ha, va: preferred positions first
        candidates_sig = [(10, 6, 'left', 'bottom'), (10, -6, 'left', 'top'), (-10, 6, 'right', 'bottom'), (-10, -6, 'right', 'top'),
                          (0, 14, 'center', 'bottom'), (0, -14, 'center', 'top')]
        candidates_bkg = [(0, -12, 'center', 'top'), (0, 12, 'center', 'bottom'), (10, -6, 'left', 'top'), (10, 6, 'left', 'bottom'),
                          (-10, -6, 'right', 'top'), (-10, 6, 'right', 'bottom')]
        candidates_far = [(dx * f, dy * f, ha, va) for f in (2, 3) for dx, dy, ha, va in candidates_sig]
        placed = []
        for i, (k, ch, p, is_sig) in enumerate(zip(ranks, p_global_argsorted, p_global_sorted, is_sig_sorted)):
            candidates = (candidates_sig if is_sig else candidates_bkg) + candidates_far
            for dx, dy, ha, va in candidates:
                ann = ax.annotate(f'ch. {ch + 1}', (k, p), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=13)
                bbox = ann.get_window_extent(renderer).padded(2)
                free = (ax.bbox.contains(bbox.x0, bbox.y0) and ax.bbox.contains(bbox.x1, bbox.y1)
                        and not any(path.intersects_bbox(bbox, filled=False) for path in line_paths)
                        and not any(bbox.overlaps(b) for j, b in enumerate(marker_boxes) if j != i)
                        and not any(bbox.overlaps(b) for b in placed))
                if free:
                    break
                ann.remove()
            else:  # no free position: fall back to the preferred one
                dx, dy, ha, va = candidates[0]
                ann = ax.annotate(f'ch. {ch + 1}', (k, p), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=13)
            placed.append(ann.get_window_extent(renderer))

    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.show()
