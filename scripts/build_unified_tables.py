#!/usr/bin/env python3
"""Build the two unified five-model results tables (lambda=0 and lambda=inf)
for Step 5, by pure transcription from the existing temporal-OOS JSON results.

NO new computation: every parameter / metric / runtime is read straight from
results/07_temporal_oos/ and results/07b_temporal_oos_two_factor/.

Emits LaTeX to _unified_tables.tex and then re-parses that file and diffs
every cell back against the JSON (round-trip verification).

Bound (star) rule: a parameter is flagged with $^{\\star}$ when its value,
formatted to the column's display precision, coincides with a calibration
bracket endpoint -- reproducing the actual flagging of the existing
tab:paper-faithful-calibrated-params table. Bounds are read from
src/calibration/optimizer.py (LH_BOUNDS/AB_BOUNDS) and
scripts/07b_temporal_oos_two_factor.py (TF_BOUNDS).
"""
from __future__ import annotations
import json, re, sys

DATES = ["2024-08-05", "2024-12-04", "2025-04-03"]
PAIR = {"2024-08-05": "2024-08-05_2024-08-06",
        "2024-12-04": "2024-12-04_2024-12-05",
        "2025-04-03": "2025-04-03_2025-04-04"}
SF = ["LH-geo", "LH-L2", "aB-L2", "aB-geo"]
LAMS = [("lam0.0", "0"), ("laminf", "inf")]

# Display labels (LaTeX) for the Model column
DISP = {"LH-geo": "LH-geo", "LH-L2": "LH-$L^{2}$",
        "aB-L2": "aB-$L^{2}$", "aB-geo": "aB-geo", "TF-LH": "TF-LH"}

# Calibration bounds (verbatim from the source).
LH_B = {"nu": (0.05, 3.0), "rho": (-0.99, 0.0), "H": (0.02, 0.49)}
AB_B = {"eta": (0.5, 5.0), "rho": (-0.99, 0.0), "H": (0.02, 0.49)}
TF_B = {"nu1": (0.05, 3.0), "rho1": (-0.99, -0.01), "H1": (0.02, 0.49),
        "lam2": (0.10, 10.0), "theta2": (0.001, 0.04),
        "nu2": (0.01, 2.0), "rho2": (-0.99, -0.01)}

STAR = "$^{\\star}$"


def sf_json(pair, suf):
    p = f"results/07_temporal_oos/{pair}_42_{suf}/temporal_oos_results.json"
    return json.load(open(p))["results"]


def tf_json(pair, suf):
    p = f"results/07b_temporal_oos_two_factor/{pair}_42_{suf}/temporal_oos_results.json"
    return json.load(open(p))["results"]["TF-LH"]


def cell(value, dp, bounds=None):
    """Format value to dp decimals; append star if it displays as a bound."""
    s = f"{value:.{dp}f}"
    starred = False
    if bounds is not None:
        starred = any(f"{b:.{dp}f}" == s for b in bounds)
    return s + (STAR if starred else ""), starred


def metric(value):
    return f"{value:.4f}"


# Track the star audit
audit = []  # (lam_label, date, model, param, displayed, bound_hit)


def param_cells(model, p, lam_label, date):
    """Return the three main-grid parameter cells (H, rho, nu/eta) for a model."""
    if model in ("LH-geo", "LH-L2"):
        H, rho, nu = p["H"], p["rho"], p["nu"]
        cH, sH = cell(H, 3, LH_B["H"]); crho, srho = cell(rho, 3, LH_B["rho"]); cnu, snu = cell(nu, 3, LH_B["nu"])
        bd = LH_B
    elif model in ("aB-L2", "aB-geo"):
        H, rho, nu = p["H"], p["rho"], p["eta"]
        cH, sH = cell(H, 3, AB_B["H"]); crho, srho = cell(rho, 3, AB_B["rho"]); cnu, snu = cell(nu, 3, AB_B["eta"])
        bd = AB_B
    else:  # TF-LH main columns map to block-1 params
        H, rho, nu = p["H1"], p["rho1"], p["nu1"]
        cH, sH = cell(H, 3, TF_B["H1"]); crho, srho = cell(rho, 3, TF_B["rho1"]); cnu, snu = cell(nu, 3, TF_B["nu1"])
        bd = TF_B
    if sH: audit.append((lam_label, date, model, "H", cH, bd.get("H", bd.get("H1"))))
    if srho: audit.append((lam_label, date, model, "rho", crho, bd.get("rho", bd.get("rho1"))))
    if snu: audit.append((lam_label, date, model, "nu/eta", cnu, bd.get("nu", bd.get("eta", bd.get("nu1")))))
    return cH, crho, cnu


def tf_extra(p, lam_label, date):
    clam, slam = cell(p["lam2"], 2, TF_B["lam2"])
    cth, sth = cell(p["theta2"], 4, TF_B["theta2"])
    cnu2, snu2 = cell(p["nu2"], 3, TF_B["nu2"])
    crho2, srho2 = cell(p["rho2"], 3, TF_B["rho2"])
    if slam: audit.append((lam_label, date, "TF-LH", "lam2", clam, TF_B["lam2"]))
    if sth: audit.append((lam_label, date, "TF-LH", "theta2", cth, TF_B["theta2"]))
    if snu2: audit.append((lam_label, date, "TF-LH", "nu2", cnu2, TF_B["nu2"]))
    if srho2: audit.append((lam_label, date, "TF-LH", "rho2", crho2, TF_B["rho2"]))
    return clam, cth, cnu2, crho2


def build_table(suf, lam_disp, label):
    lam_label = lam_disp
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l l | c c c | c c | c c | r}")
    lines.append("\\toprule")
    lines.append("\\multicolumn{2}{c|}{} & \\multicolumn{3}{c|}{Calibrated parameters} & "
                 "\\multicolumn{2}{c|}{In-sample} & \\multicolumn{2}{c|}{Temporal OOS} & \\\\")
    lines.append("Date & Model & $H/H_1$ & $\\rho/\\rho_1$ & $\\nu/\\eta/\\nu_1$ & "
                 "iv & sk & iv & sk & $t$\\,(s) \\\\")
    lines.append("\\midrule")
    for di, date in enumerate(DATES):
        pair = PAIR[date]
        sfr = sf_json(pair, suf)
        tfr = tf_json(pair, suf)
        # single-factor rows
        for mi, m in enumerate(SF):
            r = sfr[m]
            cH, crho, cnu = param_cells(m, r["params"], lam_label, date)
            datecol = date if mi == 0 else ""
            row = (f"{datecol} & {DISP[m]} & {cH} & {crho} & {cnu} & "
                   f"{metric(r['in_sample']['iv_rmse'])} & {metric(r['in_sample']['skew_rmse'])} & "
                   f"{metric(r['temporal_oos']['iv_rmse'])} & {metric(r['temporal_oos']['skew_rmse'])} & "
                   f"{r['elapsed_s']:.1f} \\\\")
            lines.append(row)
        # TF-LH main row
        cH, crho, cnu = param_cells("TF-LH", tfr["params"], lam_label, date)
        lines.append(f" & {DISP['TF-LH']} & {cH} & {crho} & {cnu} & "
                     f"{metric(tfr['in_sample']['iv_rmse'])} & {metric(tfr['in_sample']['skew_rmse'])} & "
                     f"{metric(tfr['temporal_oos']['iv_rmse'])} & {metric(tfr['temporal_oos']['skew_rmse'])} & "
                     f"{tfr['elapsed_s']:.1f} \\\\")
        # TF-LH block-2 sub-row
        clam, cth, cnu2, crho2 = tf_extra(tfr["params"], lam_label, date)
        lines.append(" & & \\multicolumn{8}{l}{\\footnotesize TF-LH block~2: "
                     f"$\\lambda_2$ = {clam}, $\\theta_2$ = {cth}, $\\nu_2$ = {cnu2}, $\\rho_2$ = {crho2}}} \\\\")
        lines.append("\\bottomrule" if di == len(DATES) - 1 else "\\midrule")
    lines.append("\\end{tabular}")
    lam_tex = "0" if lam_disp == "0" else "\\infty"
    lines.append("\\caption{Unified five-model comparison under the $\\lambda = " + lam_tex +
                 "$ calibration objective: calibrated parameters, in-sample (IS) and "
                 "temporal out-of-sample (tOOS) fit metrics, and per-calibration wall-clock "
                 "runtime, on the three train/test pairs (train date shown; tOOS is the next "
                 "trading day with parameters frozen). \\texttt{iv} = vega-weighted IV-RMSE; "
                 "\\texttt{sk} = unweighted RMSE of the 98/102 ATM skew. The three single-factor "
                 "parameter columns hold $(H,\\rho,\\nu)$ for LH, $(H,\\rho,\\eta)$ for aBergomi, "
                 "and $(H_1,\\rho_1,\\nu_1)$ for TF-LH, whose four block-2 parameters "
                 "$(\\lambda_2,\\theta_2,\\nu_2,\\rho_2)$ are given in the sub-row beneath each "
                 "TF-LH row ($V^{(2)}_0 = \\theta_2$). A $\\star$ marks a parameter at its "
                 "calibration bound (the displayed value coincides with the bracket endpoint). "
                 "All numbers transcribed from the temporal-OOS result JSONs; no recomputation.}")
    lines.append("\\label{" + label + "}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    t0 = build_table("lam0.0", "0", "tab:unified-lam0")
    tinf = build_table("laminf", "inf", "tab:unified-laminf")
    out = t0 + "\n\n" + tinf + "\n"
    with open("_unified_tables.tex", "w", encoding="utf-8", newline="\r\n") as f:
        f.write(out)
    print("WROTE _unified_tables.tex")
    print("\n===== STAR AUDIT (every starred cell) =====")
    for a in audit:
        print(f"  lambda={a[0]:3} {a[1]} {a[2]:7} {a[3]:7} displayed={a[4]:14} bound={a[5]}")
    verify()


def clean_cell(c):
    return c.replace(STAR, "").replace("$", "").replace("\\\\", "").strip()


def parse_main_rows(block):
    """Return list of (model_disp, [8 cleaned cell strings]) for each 10-field row."""
    rows = []
    for ln in block.splitlines():
        if "multicolumn" in ln:
            continue
        parts = ln.split("&")
        if len(parts) != 10:
            continue
        model = parts[1].strip()
        cells = [clean_cell(x) for x in parts[2:]]
        rows.append((model, cells))
    return rows


def parse_blk2_rows(block):
    """Return list of [lam2, theta2, nu2, rho2] cleaned strings for each block-2 sub-row."""
    out = []
    for ln in block.splitlines():
        if "block~2" not in ln:
            continue
        nums = re.findall(r"=\s*(-?[0-9.]+)", ln.replace(STAR, ""))
        out.append(nums)
    return out


def verify():
    """Round-trip: parse numbers back out of _unified_tables.tex and diff vs JSON."""
    txt = open("_unified_tables.tex", encoding="utf-8").read()
    blocks = {}
    for lab, suf in (("tab:unified-lam0", "lam0.0"), ("tab:unified-laminf", "laminf")):
        m = re.search(r"\\begin\{table\}.*?\\label\{" + re.escape(lab) + r"\}", txt, re.S)
        blocks[suf] = m.group(0)

    total = 0
    fails = []
    tf_laminf_metric_cells = 0

    for suf, lamname in (("lam0.0", "0"), ("laminf", "inf")):
        parsed_main = parse_main_rows(blocks[suf])
        parsed_blk2 = parse_blk2_rows(blocks[suf])
        # Build the full expected set from JSON (one entry per (date, model)).
        for date in DATES:
            pair = PAIR[date]
            sfr = sf_json(pair, suf)
            tfr = tf_json(pair, suf)
            for m in SF + ["TF-LH"]:
                if m == "TF-LH":
                    P, R = tfr["params"], tfr
                    exp = [f"{P['H1']:.3f}", f"{P['rho1']:.3f}", f"{P['nu1']:.3f}"]
                elif m in ("LH-geo", "LH-L2"):
                    P, R = sfr[m]["params"], sfr[m]
                    exp = [f"{P['H']:.3f}", f"{P['rho']:.3f}", f"{P['nu']:.3f}"]
                else:
                    P, R = sfr[m]["params"], sfr[m]
                    exp = [f"{P['H']:.3f}", f"{P['rho']:.3f}", f"{P['eta']:.3f}"]
                exp += [f"{R['in_sample']['iv_rmse']:.4f}", f"{R['in_sample']['skew_rmse']:.4f}",
                        f"{R['temporal_oos']['iv_rmse']:.4f}", f"{R['temporal_oos']['skew_rmse']:.4f}",
                        f"{R['elapsed_s']:.1f}"]
                # find a parsed row for this model whose 8 cells equal exp (uniquely ID's the date block)
                cand = [c for (mod, c) in parsed_main if mod == DISP[m] and c == exp]
                total += len(exp)
                if not cand:
                    # locate the closest parsed row for this model to report the diff
                    same_model = [c for (mod, c) in parsed_main if mod == DISP[m]]
                    fails.append((lamname, date, m, "NO EXACT ROW MATCH", exp, same_model))
                if suf == "laminf" and m == "TF-LH":
                    tf_laminf_metric_cells += 4  # iv_IS, sk_IS, iv_tOOS, sk_tOOS
            # block-2 sub-row
            tfp = tfr["params"]
            sub_exp = [f"{tfp['lam2']:.2f}", f"{tfp['theta2']:.4f}", f"{tfp['nu2']:.3f}", f"{tfp['rho2']:.3f}"]
            total += len(sub_exp)
            if sub_exp not in parsed_blk2:
                fails.append((lamname, date, "TF-LH-blk2", "NO EXACT MATCH", sub_exp, parsed_blk2))

    print("\n===== ROUND-TRIP VERIFICATION =====")
    print(f"  total cells checked: {total}")
    print(f"  TF-LH lambda=inf single-source metric cells checked: {tf_laminf_metric_cells} (expect 12)")
    if fails:
        print(f"  RESULT: FAIL ({len(fails)} mismatches)")
        for f in fails:
            print("   MISMATCH", f)
    else:
        print("  RESULT: PASS -- every JSON-derived cell is present in the generated LaTeX at display precision")


if __name__ == "__main__":
    main()
