Scripts used for the updated Section 7.1 results (same toy model as notebooks/experiments.ipynb).
  bs_trials_check.py   vectorised fitter (q(m) scan), Bayer-Seljak helpers; --validate compares with the notebook fit
  run_section71.py     Table 1 and Bayer-Seljak figure: python run_section71.py <grid> 100000 <replica>  (replicas 0,1,2)
  bs_figure.py         Table 1 numbers + figures/bs_comparison.pdf from run71_*.npz
  section7_checks.py   grid scan (Fig. 3 right), width scan (Fig. 4), width scan with fixed slope
  make_fig3left.py     figures/gg/mu_scan_upcrossings.pdf (Fig. 3 left; _sqrtq variant)
  make_fig2.py         figures/q_null_dist.pdf and figures/pval_dist.pdf (Fig. 2)
  make_figures.py      figures/trial_factor_Gv_Bon.pdf and figures/tfactor_vs_signalwidth.pdf from the *.log files
Run from this folder. The run71_*.npz files (per-toy results) are not included (large); rerun to regenerate.
  multichannel_721.py  Section 7.2.1: per-channel GV calibration, one pseudo-experiment (Table 2, figure),
                       5000 pseudo-experiments (Table 3); outputs multichannel_721_*.json, log
  screening_722.py     Section 7.2.2: 50 classes, 10 with signal; per-class GV calibration (2000 toys), one
                       pseudo-experiment and 1000 pseudo-experiments; figures screening_*.pdf, Table 4
