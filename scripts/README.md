# Experiment scripts

Every script inserts the repository root into `sys.path` at import, so run each
one **from the repository root**:

```bash
python scripts/00_sanity_checks.py
```

The numbered scripts form a pipeline. Script `00` gates everything and must print
`OK` on every check before anything else is trusted. Scripts marked **needs data**
expect a cleaned option-chain CSV at `data/spx_<date>.csv` (see
[`../data/README.md`](../data/README.md)); you can point most of them at the
shipped synthetic sample instead. Scripts marked **standalone** run with no
external data.

Many scripts accept CLI flags for quick smoke tests, for example `--de-maxiter`,
`--M-paths`, `--max-maturities`, `--max-strikes`, `--no-ab`, `--skip-calib`.

---

## Utility

| Script | Data | What it does |
|--------|------|--------------|
| `make_sample_data.py` | standalone | Generates `data/sample_synthetic_spx.csv`, a synthetic option chain priced from an n-factor lifted Heston model, so the real-data scripts run out of the box. |

## 1. Sanity and core reproducible pipeline (00-05)

The backbone experiments of the thesis.

| Script | Data | What it does |
|--------|------|--------------|
| `00_sanity_checks.py` | standalone | Gating checks: lifted Heston characteristic function, put-call parity, no-arbitrage, aBergomi forward-variance and MC error, kernel L2 fit, COS vs analytic Black-Scholes. All must print `OK`. |
| `01_lh_convergence.py` | standalone | Lifted Heston IV-surface convergence in n against an n=500 reference for H in {0.05, 0.10, 0.20, 0.30}. Writes metrics JSON and a convergence plot. |
| `02_kernel_study.py` | standalone | Isolates kernel approximation error, comparing the geometric and L2-fit kernels in L2 and sup norm, to test whether the LH-geo vs aB-L2 gap is a kernel artefact. |
| `03_synthetic_comparison.py` | standalone | Generates a synthetic market surface from n=500 lifted Heston, then calibrates LH-geo and aB-L2 (n=20) to it to quantify calibration bias against a known ground truth. |
| `04_spx_comparison.py` | needs data | Main empirical result: in-sample calibration and out-of-sample evaluation of LH-geo, aB-L2, aB-geo on one cleaned SPX date (`--data CSV`) or a synthetic surface (`--synthetic`). |
| `05_make_thesis_figures.py` | standalone | Reads committed JSON/CSV from experiments 01-04 and regenerates publication figures with a uniform style. No new computation. |

## 2. Lambda sweep and temporal out-of-sample (06, 07, 07b)

Extends the comparison to the IV-RMSE versus ATM-skew tradeoff and to cross-day
parameter stability.

| Script | Data | What it does |
|--------|------|--------------|
| `06_lambda_sweep.py` | needs data | Sweeps four models over five skew-weight values `lambda` on one SPX surface, flushing a reloadable JSON after each (model, lambda) pair. |
| `07_temporal_oos.py` | needs data | Calibrates all four single-factor models on a train date, refits only `xi0` on a test date with (H, rho, nu/eta) fixed, and reports in-sample vs temporal out-of-sample IV-RMSE and skew-RMSE. |
| `07b_temporal_oos_two_factor.py` | needs data | Same temporal out-of-sample protocol for the two-factor lifted Heston (Chapter 8, independent blocks), COS pricing only. |
| `plot_07_temporal_oos.py` | needs data | Loads a `07` result, reprices all models on both surfaces, and draws IV-slice, ATM-skew, and bar figures (`--no-ab` skips slow aBergomi repricing). |
| `plot_07b_combined.py` | needs data | Combines `07` single-factor and `07b` two-factor saved params into IV-slice, ATM-skew, and bar figures for `lambda` in {0, inf}. |
| `build_unified_tables.py` | standalone | Transcribes `07` and `07b` JSON into unified five-model LaTeX tables with round-trip cell verification. No new computation. |

## 3. Two-factor and symmetric-split extension (08-13)

New work beyond the core thesis: the two-factor sanity checks plus the
positivity-clean symmetric-split remedy and its diagnostics.

| Script | Data | What it does |
|--------|------|--------------|
| `08_two_factor_sanity_checks.py` | standalone | Four checks for the two-factor lifted Heston: recover single-block LH, recover double Heston, COS vs MC agreement, CF normalisation. |
| `09_symmetric_sanity_checks.py` | needs data | Sanity checks for the symmetric-split model; checks 1-4 are self-contained, check 5 (raw positivity probe) reads a steep SPX `xi0`. |
| `10_symmetric_fair_run.py` | needs data | Clip-free fair run: 3 date pairs x {lambda 0, inf} x 5 seeds, tables, figures, and a REPORT.md, compared against committed single-factor results. |
| `11_two_factor_value_diagnostics.py` | needs data | Tests whether the additive split is genuinely two-factor and whether block 2 adds skew-shape freedom a single block cannot. Writes DIAGNOSTICS.md. |
| `12_extension_term_structure_split.py` | needs data | Evaluates a maturity-dependent split `w(t)` and a two-Hurst slow block, documents a correlated-blocks extension. Writes EXTENSIONS.md. |
| `13_symmetric_skew_rmse_plots.py` | needs data | Reproduces the ATM-skew term-structure and RMSE-bar figures with the symmetric-split model added beside the baselines. |
| `_symmetric_common.py` | (imported) | Shared harness for surface loading, loss construction, and the skew grid. Not run directly. |
| `_calib_symmetric.py` | (worker) | Per-config calibration worker for the 7-parameter symmetric-split model. Dispatched by `10`. |
| `_calib_symmetric_ts.py` | (worker) | Calibration worker for the 9-parameter term-structure-split model. Dispatched by `12`. |

## 4. Validation appendix: Abi Jaber (2019) reference date (cboe_2018_*)

Reproduces the thesis appendix that validates the implementation on the paper's own
reference date, 2018-06-20. Run the adapter first to convert the CBOE end-of-day
CSV into a surface the loader ingests, then the checks and calibrations.

| Script | Data | What it does |
|--------|------|--------------|
| `cboe_2018_adapter.py` | needs data | Step 1: converts the CBOE `UnderlyingOptionsEODCalcs` 2018-06-20 CSV into a loader-format surface (ACT/365, put-call-parity forward). |
| `cboe_2018_iv_check.py` | needs data | Inverts each option mid on the parity forward and cross-checks against CBOE-reported IVs. |
| `cboe_2018_atm_skew.py` | needs data | Empirical 98/102 ATM skew with a log-log power-law fit, compared to Abi Jaber (2019) Figure 1. |
| `cboe_2018_lh_calibration.py` | needs data | Calibrates LH-geo at lambda=0 (r_n=2.5) and compares the model skew to the empirical dots and the paper power law. |
| `cboe_2018_lh_calibration_paper_rn.py` | needs data | Control: same calibration with the paper geometric grid r_n, to isolate whether short-end overshoot is grid-driven. |
| `cboe_2018_short_end_diagnostic.py` | needs data | Investigates sub-month skew flattening: quote quality, monotonicity, liquid-only refits, and a CBOE-IV cross-check. |
| `cboe_2018_five_models_laminf.py` | needs data | Calibrates one of the five models at lambda=inf on 2018-06-20; designed to be launched in parallel per model. |
| `cboe_2018_atm_skew_two_panel_plot.py` | standalone | Redraws the appendix skew figure as a linear plus log-log two-panel layout from committed outputs. |
| `cboe_2018_lh_rn_comparison.py` | standalone | Side-by-side of the two LH-geo calibrations (r_n=2.5 vs paper r_n) from committed results. |
| `cboe_2018_five_models_plot.py` | standalone | Aggregates the five per-model lambda=inf outputs into a combined overlay of the calibrated skew curves. |

## 5. Data acquisition

Produce the cleaned `data/spx_<date>.csv` files. Both need external inputs and are
not self-contained.

| Script | Data | What it does |
|--------|------|--------------|
| `download_spx_polygon.py` | (network) | Fetches paginated SPX/SPXW option snapshots from the Polygon.io API (key required) and writes a cleaned CSV. |
| `extract_spx_eod_quotes.py` | (flatfile) | Filters a large Polygon quotes flatfile to SPX/SPXW, keeps the last quote at or before a target time, and writes a cleaned CSV. |

## Optional diagnostics

One-off numerical investigations, kept for reproducibility but not part of the main
pipeline. Most probe whether the two-factor deep-out-of-the-money "wobble" is a COS
/ Black-Scholes / Riccati numerical artefact or a real feature of the output.

`_diag_ncos.py`, `_diag_nsteps.py`, `_diag_price_vs_iv.py`, `_diag_svi_ab.py`,
`_diag_svi_ab_laminf.py`, `_diag_symmetric_positivity.py`, `_sanity_svi.py`.
