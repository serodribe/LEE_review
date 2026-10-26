"""
50-channel multiple-testing study (notebook section "Unknown physics process"), run outside Jupyter.

Each channel's toys and GV calibration are saved to OUT_DIR/channel_XX.npz as soon as they finish, and channels
already on disk are skipped, so the run can be stopped and restarted at any time. Once every channel is done,
the global p-values are combined into OUT_DIR/res_50ch.npz, which the notebook loads for the plot.

Usage (from anywhere; survives SSH/VS Code disconnects):
    nohup python scripts/run_multichannel_50ch.py > output/data/multichannel_50ch/run.log 2>&1 &
    tail -f output/data/multichannel_50ch/run.log
"""
import argparse
import os
import time

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, "output", "data", "multichannel_50ch")

# ----------------------------------------------------------------------
# Model (same as the notebook)
# ----------------------------------------------------------------------
XMIN, XMAX = 0.0, 10.0
SIGMA_SIG = 0.2  # fixed signal width
bins = np.linspace(XMIN, XMAX, 70)
N_events = 100000

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

def Z_from_mu(mu, n_tot, lam, m, width=SIGMA_SIG):
        """ Compute the significance Z for a given signal fraction mu. """
        x_low = m - 3 * width
        x_high = m + 3 * width
        N_sig = mu * n_tot * np.trapezoid(sig_pdf(np.linspace(x_low, x_high, 1000), m, width), np.linspace(x_low, x_high, 1000))
        N_bkg = (1 - mu) * n_tot * np.trapezoid(bkg_pdf(np.linspace(x_low, x_high, 1000), lam), np.linspace(x_low, x_high, 1000))
        if N_bkg <= 0:
            return 0.0
        return np.sqrt(N_sig**2 / N_bkg)

def mu_from_Z(z, n_tot, lam, m, width = SIGMA_SIG):
    """ Compute the signal fraction that corresponds to a given significance Z, for a given background and signal model. """
    result = minimize_scalar(lambda mu: (Z_from_mu(mu, n_tot, lam, m, width) - z)**2, bounds=(0, 1), method='bounded')
    return result.x if result.success else None

# ----------------------------------------------------------------------
# Likelihood fits (same as the notebook)
# ----------------------------------------------------------------------
def nll(counts, expected):
    """Poisson negative log-likelihood (binned), dropping constant terms."""
    expected = np.clip(expected, 1e-9, None)
    return np.sum(expected - counts * np.log(expected))

def fit_bkg_only(counts, edges, n_tot):
    """Null hypothesis: pure exponential. Fit lambda, return (nll_min, lambda_hat)."""
    def obj(lam):
        return nll(counts, expected_counts(lam, 1.0, 0.0, edges, n_tot))
    res = minimize_scalar(obj, bounds=(0.01, 5.0), method="bounded")
    return res.fun, res.x

def fit_sig_plus_bkg(counts, edges, n_tot, m, lam_init, sigma=SIGMA_SIG):
    """Alternative hypothesis at fixed bump position m: profile signal fraction."""
    def obj(params):
        lam, mu = params
        return nll(counts, expected_counts(lam, m, mu, edges, n_tot, width=sigma))
    res = minimize(obj, x0=[lam_init, 0.0],bounds=[(0.01, 5.0), (0., 1.0)],method="L-BFGS-B",)
    lam_hat, mu_hat = res.x
    return res.fun, lam_hat, mu_hat

def count_upcrossings(q_values, threshold):
    q_values = np.asarray(q_values)
    return np.sum(
        (q_values[:-1] < threshold) &
        (q_values[1:] >= threshold)
    )

# ------------------------------------------------------------
# Toy workers (same as the notebook)
# ------------------------------------------------------------
def scan_q(counts, m_grid, edges, sigma):
    """q(m) over the grid. The background-only fit does not depend on m, so it is done once."""
    n_tot = counts.sum()
    nll_null, lam_hat_null = fit_bkg_only(counts, edges, n_tot)
    q_values = np.empty(len(m_grid))
    mu_hats = np.empty(len(m_grid))
    for j, m in enumerate(m_grid):
        nll_alt, _, mu_hats[j] = fit_sig_plus_bkg(counts, edges, n_tot, m, lam_hat_null, sigma=sigma)
        q_values[j] = 2.0 * (nll_null - nll_alt)
    return q_values, mu_hats

def simulate_channel_toy(seed, lam, sigma_scan, m_grid, edges, n_events, mu_inj=0.0, m_inj=0.0, sigma_inj=0.0):
    """One pseudo-experiment of one channel (signal injected if mu_inj > 0). Returns (q_max, m_hat, mu_hat)."""
    rng = np.random.default_rng(seed)
    n_sig = rng.poisson(n_events * mu_inj) if mu_inj > 0 else 0
    counts = rng.poisson(expected_counts(lam, 1.0, 0.0, edges, n_events - n_sig))
    if n_sig > 0:
        counts = counts + np.histogram(rng.normal(m_inj, sigma_inj, size=n_sig), bins=edges)[0]
    q_values, mu_hats = scan_q(counts, m_grid, edges, sigma_scan)
    i_max = np.argmax(q_values)
    return max(q_values[i_max], 0.0), m_grid[i_max], mu_hats[i_max]

def simulate_null_upcrossings(seed, lam, sigma_scan, m_grid, edges, n_events, c0):
    """Background-only toy for the GV calibration: number of upcrossings of q(m) above c0."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(expected_counts(lam, 1.0, 0.0, edges, n_events))
    q_values, _ = scan_q(counts, m_grid, edges, sigma_scan)
    return count_upcrossings(q_values, c0)

# ------------------------------------------------------------
# Checkpointed version of run_multichannel_toys (same seeds, so same results)
# ------------------------------------------------------------
def run_channel(channel, cfg, toy_seeds, gv_seeds, n_jobs):
    """Toys and GV calibration of one channel."""
    n_toys, n_gv_toys = cfg['n_toys'], cfg['n_gv_toys']
    with Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto") as parallel:
        toy_results = parallel(
            delayed(simulate_channel_toy)(
                toy_seeds[channel * n_toys + i_toy],
                cfg['lam_ch'][channel],
                cfg['sigma_ch'][channel],
                cfg['m_scan'],
                bins,
                N_events,
                cfg['mu_inj'][channel],
                cfg['m_inj'],
                cfg['sigma_inj'][channel]
            )
            for i_toy in range(n_toys)
        )
        Nc_low = parallel(
            delayed(simulate_null_upcrossings)(
                gv_seeds[channel * n_gv_toys + i_toy],
                cfg['lam_ch'][channel],
                cfg['sigma_ch'][channel],
                cfg['m_scan'],
                bins,
                N_events,
                cfg['c0']
            )
            for i_toy in range(n_gv_toys)
        )
    toy_results = np.array(toy_results)  # (n_toys, 3)
    return dict(q_max=toy_results[:, 0], m_hat=toy_results[:, 1], mu_hat=toy_results[:, 2], Nc_low=np.array(Nc_low))

def combine_channels(cfg, ch_results):
    """Local and global p-values, as in run_multichannel_toys. Arrays have shape (n_toys, n_channels)."""
    q_max = np.column_stack([r['q_max'] for r in ch_results])
    m_hat = np.column_stack([r['m_hat'] for r in ch_results])
    mu_hat = np.column_stack([r['mu_hat'] for r in ch_results])
    Nc_low = np.stack([r['Nc_low'] for r in ch_results])  # (n_channels, n_gv_toys)
    Nc0 = Nc_low.mean(axis=1)
    Nc0_err = Nc_low.std(axis=1) / np.sqrt(cfg['n_gv_toys'])

    # q ~ 1/2 chi2_0 + 1/2 chi2_1 (mu >= 0): P(q >= c) = 1/2 chi2_1.sf(c) for c > 0 and 1 for c = 0
    p_local = np.where(q_max > 0, 0.5 * stats.chi2.sf(q_max, df=1), 1.0)
    gv_extrapol = np.exp(-(q_max - cfg['c0']) / 2)
    p_global = np.clip(p_local + Nc0 * gv_extrapol, None, 1.0)
    p_global_err = Nc0_err * gv_extrapol

    is_sig = cfg['mu_inj'] > 0
    for kind, mask in [('bkg', ~is_sig), ('signal', is_sig)]:
        print(f"{kind} channels ({mask.sum()}): <N(C0)> in [{Nc0[mask].min():.2f}, {Nc0[mask].max():.2f}], "
              f"median q_max={np.median(q_max[:, mask]):.2f}, median p_global={np.median(p_global[:, mask]):.3e}, "
              f"frac(q_max < C0)={np.mean(q_max[:, mask] < cfg['c0']):.3f}")

    return dict(q_max=q_max, m_hat=m_hat, mu_hat=mu_hat, p_local=p_local, p_global=p_global,
                p_global_err=p_global_err, Nc0=Nc0, Nc0_err=Nc0_err)

def make_config(n_toys, n_gv_toys):
    """Channel configuration of the notebook's 50-channel cell (same seed, same draws in the same order)."""
    N_CHANNELS_50 = 50  # number of channels
    N_SIG_50 = 10  # number of channels with signal
    M_INJ_50 = 2.5  # fixed signal position (same for all channels)
    LAM_RANGE = np.linspace(0.1, 0.5, 11)  # different background decay rate
    CONFIG_SEED_50 = 1357
    rng_config = np.random.default_rng(CONFIG_SEED_50)
    SIGMA_CH_50 = rng_config.uniform(0.2, 0.5, size=N_CHANNELS_50)  # signal template width used in the scan of each channel
    LAM_CH_50 = rng_config.choice(LAM_RANGE, size=N_CHANNELS_50)
    sig_channels_50ch = rng_config.choice(N_CHANNELS_50, size=N_SIG_50, replace=False)  # channels with signal
    Z_SIG_50 = rng_config.uniform(2, 4, size=N_SIG_50)
    MU_INJ_50 = np.zeros(N_CHANNELS_50)
    MU_INJ_50[sig_channels_50ch] = [mu_from_Z(z, N_events, lam, M_INJ_50, width=sigma) for z, lam, sigma in zip(Z_SIG_50, LAM_CH_50[sig_channels_50ch], SIGMA_CH_50[sig_channels_50ch])]
    return dict(n_channels=N_CHANNELS_50, n_toys=n_toys, n_gv_toys=n_gv_toys, seed=CONFIG_SEED_50,
                lam_ch=LAM_CH_50, sigma_ch=SIGMA_CH_50, mu_inj=MU_INJ_50, m_inj=M_INJ_50, sigma_inj=SIGMA_CH_50,
                sig_channels=sig_channels_50ch, z_sig=Z_SIG_50, m_scan=np.linspace(XMIN, XMAX, 200), c0=0.5)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-toys", type=int, default=5000)
    parser.add_argument("--n-gv-toys", type=int, default=800)
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = make_config(args.n_toys, args.n_gv_toys)
    n_channels = cfg['n_channels']
    print(f"Signal channels: {cfg['sig_channels']}, Z_SIG={cfg['z_sig']}, mu={cfg['mu_inj'][cfg['sig_channels']]}", flush=True)

    # Same seed streams as run_multichannel_toys: seed of (channel, toy) is element channel * n_toys + toy
    toy_seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence([cfg['seed'], 13579]).spawn(n_channels * cfg['n_toys'])]
    gv_seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence([cfg['seed'], 97531]).spawn(n_channels * cfg['n_gv_toys'])]

    ch_results = []
    t_start = time.time()
    for channel in range(n_channels):
        fname = os.path.join(args.out_dir, f"channel_{channel:02d}.npz")
        if os.path.exists(fname):
            r = dict(np.load(fname))
            if int(r['n_toys']) == cfg['n_toys'] and int(r['n_gv_toys']) == cfg['n_gv_toys']:
                ch_results.append(r)
                print(f"Channel {channel + 1}/{n_channels}: loaded from {fname}", flush=True)
                continue
            print(f"Channel {channel + 1}/{n_channels}: {fname} has different n_toys/n_gv_toys, recomputing", flush=True)

        t0 = time.time()
        r = run_channel(channel, cfg, toy_seeds, gv_seeds, args.n_jobs)
        np.savez(fname + ".tmp.npz", n_toys=cfg['n_toys'], n_gv_toys=cfg['n_gv_toys'], **r)
        os.replace(fname + ".tmp.npz", fname)  # atomic: a killed run never leaves a half-written checkpoint
        ch_results.append(r)
        print(f"Channel {channel + 1}/{n_channels}: done in {(time.time() - t0) / 60:.1f} min "
              f"(total {(time.time() - t_start) / 60:.1f} min)", flush=True)

    res = combine_channels(cfg, ch_results)
    out_file = os.path.join(args.out_dir, "res_50ch.npz")
    np.savez(out_file, **res, **{k: v for k, v in cfg.items()})
    print(f"Saved {out_file}", flush=True)

if __name__ == "__main__":
    main()
