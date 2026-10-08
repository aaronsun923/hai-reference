"""Amendment 5 (SPEC v1, LOCKED 2026-10-08, commit 3fa0858): I(h | m) against the raw classifier.

Sensitivity only. The locked results are produced by pipeline/02_full_run.py and are not touched.
This script reuses that module's functions. The one change is a flag on the F0 construction:
F0 is the only one-input pool in h1_engine ([log P_m], start weight 1.0). With RAW_F0 its weight is
fixed at 1 instead of fitted, so F0 = clipren(P_m), the raw classifier vector under the spec's
existing clip (CLIP = 1e-3, renormalized). The with-human pools are unchanged.

Order: (1) reproduction check: with the flag off, the H1c values for all 20 variants must reproduce
reports/full_tables/h1c_curve.csv and h1c_slopes.csv bit for bit, or the script stops; (2) the raw-F0
pass; (3) reports/amendment5_REPORT.md and reports/amendment5_results.json.

Path: the spec names code/amendment5_rawF0.py; this repository keeps its scripts in pipeline/.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_run", ROOT / "pipeline" / "02_full_run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SPEC_COMMIT = "3fa0858630f12d4edd4f085a6899b309ceeac343"
CHECK_COLS = ["I_h_ind", "I_h_pool"]          # the I(h | m) quantities of H1c

RAW_F0 = False
_fit = m.fit


def fit_with_flag(logs, y, w0, rule="log"):
    """F0 is the only one-input pool. With RAW_F0 its weight is fixed at 1 (no recalibration)."""
    if RAW_F0 and len(logs) == 1:
        return np.array([1.0]), True
    return _fit(logs, y, w0, rule)


m.fit = fit_with_flag


def prepare():
    """Same items, selection set, test set, folds, ratings and references as the full run."""
    h = pd.read_csv(ROOT / "data" / "h16_deid.csv").sort_values(["code", "task_order"])
    h = h[~h.duplicated(["code", "image_name", "noise_level"], keep="first")]
    items = h[["image_name", "noise_level", "image_category"]].drop_duplicates(["image_name", "noise_level"])
    items = items.sort_values(["image_name", "noise_level"]).reset_index(drop=True)
    key = pd.MultiIndex.from_frame(items[["image_name", "noise_level"]])
    truth = items.image_category.map(m.CLS).values
    P = {}
    for lv in m.LEVELS:
        d = pd.read_csv(m.RAW / "2ntrf" / f"hai_{lv}_model_preds_max_normalized.csv")
        d = d[(d.noise_type == "phase") & d.noise_level.isin(m.NOISE)]
        for a in m.ARCH:
            g = d[d.model_name == a].set_index(["image_name", "noise_level"]).reindex(key)
            assert g[m.C16].notna().all().all() and (g.category.map(m.CLS).values == truth).all()
            P[f"{a}_{lv}"] = m.clipren(g[m.C16].values.astype(float))
    variants = [f"{a}_{lv}" for a in m.ARCH for lv in m.LEVELS]
    images = np.array(sorted(items.image_name.unique()))
    sel = images[np.sort(np.random.default_rng(m.SEED_A).choice(1200, 300, replace=False))]
    test = np.setdiff1d(images, sel)
    fm = m.fold_map(test)
    sel_mask = items.image_name.isin(sel).values
    Q = {v: m.score(P[v][sel_mask], truth[sel_mask]).mean() for v in variants}
    ti = items[~sel_mask].copy()
    tidx = ti.index.values
    n_it = len(tidx)
    r = h[h.image_name.isin(test)].copy()
    r["item"] = pd.MultiIndex.from_frame(r[["image_name", "noise_level"]]).map(
        dict(zip(zip(ti.image_name, ti.noise_level), range(n_it))))
    r = r.assign(lab=r.participant_classification.map(m.CLS).values, true=r.image_category.map(m.CLS).values,
                 conf_i=r.confidence.map({c: i for i, c in enumerate(m.CONF)}).values,
                 noise_i=r.noise_level.map({n: i for i, n in enumerate(m.NOISE)}).values,
                 fold=r.image_name.map(fm).values).reset_index(drop=True)
    logs = {v: np.log(P[v][tidx]) for v in variants}

    def ref_same_level(v):
        a, lv = v.split("_")
        return np.log(m.clipren(m.softmax(sum(logs[f"{b}_{lv}"] for b in m.ARCH if b != a))))

    return dict(r=r, ri=r.item.values.astype(int), y=truth[tidx], fold_it=ti.image_name.map(fm).values,
                fold_r=r.fold.values, noise_it=ti.noise_level.values, noise_r=r.noise_level.values,
                img_it=ti.image_name.values, logs=logs, ref={v: ref_same_level(v) for v in variants},
                variants=variants, Q=Q)


def run_variants(d, hv, raw):
    global RAW_F0
    RAW_F0 = raw
    try:
        return {v: m.h1_engine(d["logs"][v], d["ref"][v], hv[0], hv[1], d["ri"], d["y"], d["fold_it"],
                               d["fold_r"], d["noise_it"], d["noise_r"], label=f"amend5:{'raw' if raw else 'recal'}:{v}")
                for v in d["variants"]}
    finally:
        RAW_F0 = False


def h1c(d, runs):
    """H1c curve (mean, image-cluster bootstrap CI) and slopes on Q, as in 02_full_run.main."""
    img_it = d["img_it"]
    curve = []
    for v in d["variants"]:
        b = m.cluster_boot(runs[v]["q"][CHECK_COLS].values, img_it)
        curve.append(dict(variant=v, Q=d["Q"][v], **{c: b[i]["mean"] for i, c in enumerate(CHECK_COLS)},
                          **{f"{c}_lo": b[i]["ci_lo"] for i, c in enumerate(CHECK_COLS)},
                          **{f"{c}_hi": b[i]["ci_hi"] for i, c in enumerate(CHECK_COLS)}))
    curve = pd.DataFrame(curve).set_index("variant")
    g_codes, g = np.unique(img_it, return_inverse=True)
    draws = np.random.default_rng(m.SEED_B).integers(0, len(g_codes), (m.N_BOOT, len(g_codes)))
    den = np.bincount(g, minlength=len(g_codes))[draws].sum(1)
    x = curve.Q.values
    slopes, boot = {}, {}
    for c in CHECK_COLS:
        bm = np.stack([np.bincount(g, runs[v]["q"][c].values, len(g_codes))[draws].sum(1) / den
                       for v in curve.index])
        _, _, _, flag = m.ols_flags(x, curve[c].values)
        b_all = np.array([m.slope(x, bm[:, i]) for i in range(m.N_BOOT)])
        uf = ~flag
        b_uf = np.array([m.slope(x[uf], bm[uf, i]) for i in range(m.N_BOOT)])
        slopes[c] = dict(slope_all=m.slope(x, curve[c].values), all_lo=np.percentile(b_all, 2.5),
                         all_hi=np.percentile(b_all, 97.5), slope_unflagged=m.slope(x[uf], curve[c].values[uf]),
                         unflagged_lo=np.percentile(b_uf, 2.5), unflagged_hi=np.percentile(b_uf, 97.5),
                         n_flagged=int(flag.sum()))
        boot[c] = (bm, b_all)
    return curve, pd.DataFrame(slopes).T, boot


def reproduction_check(curve, slopes):
    locked = pd.read_csv(m.TAB / "h1c_curve.csv", float_precision="round_trip").set_index("variant")
    locked_s = pd.read_csv(m.TAB / "h1c_slopes.csv", float_precision="round_trip").set_index("quantity")
    cols = [f"{c}{s}" for c in CHECK_COLS for s in ("", "_lo", "_hi")] + ["Q"]
    assert set(curve.index) == set(locked.index) and len(curve) == 20
    d_curve = (curve[cols] - locked.loc[curve.index, cols]).abs()
    scols = ["slope_all", "all_lo", "all_hi", "slope_unflagged", "unflagged_lo", "unflagged_hi", "n_flagged"]
    d_slope = (slopes[scols].astype(float) - locked_s.loc[CHECK_COLS, scols].astype(float)).abs()
    return d_curve, d_slope


def main():
    d = prepare()
    hv = m.h16_human_vectors(d["r"], "structured")
    runs_off = run_variants(d, hv, raw=False)
    curve, slopes, _ = h1c(d, runs_off)
    d_curve, d_slope = reproduction_check(curve, slopes)
    n_exact = int((d_curve.values == 0).sum() + (d_slope.values == 0).sum())
    n_vals = d_curve.size + d_slope.size
    print(f"Reproduction check (flag off) vs locked reports/full_tables/h1c_curve.csv and h1c_slopes.csv, "
          f"{len(curve)} variants, {CHECK_COLS}:")
    print(f"  values compared {n_vals}; bit-identical {n_exact}; max |difference| "
          f"curve {d_curve.values.max():.3g}, slopes {d_slope.values.max():.3g}")
    if n_exact != n_vals:
        print("  not bit-identical; largest differences:")
        print(d_curve.stack().sort_values(ascending=False).head(5).to_string())
        print(d_slope.stack().sort_values(ascending=False).head(5).to_string())
        sys.exit("Reproduction check FAILED: stopping before the raw-F0 run.")
    print("Reproduction check PASSED.")
    if "--check-only" in sys.argv:
        return

    # ---------------- raw-F0 pass
    runs_raw = run_variants(d, hv, raw=True)
    curve_raw, slopes_raw, _ = h1c(d, runs_raw)
    ok_all = all(r["ok"] for r in runs_off.values()) and all(r["ok"] for r in runs_raw.values())

    # paired difference I_raw - I_recal per variant: same image-cluster draws (cluster_boot, SEED_B)
    rows = []
    for v in d["variants"]:
        diff = runs_raw[v]["q"][CHECK_COLS].values - runs_off[v]["q"][CHECK_COLS].values
        b = m.cluster_boot(diff, d["img_it"])
        row = dict(variant=v, baseline_loss_Q=d["Q"][v])
        for i, c in enumerate(CHECK_COLS):
            rec, raw = curve.loc[v, c], curve_raw.loc[v, c]
            row.update({f"{c}_recal": rec, f"{c}_recal_lo": curve.loc[v, f"{c}_lo"], f"{c}_recal_hi": curve.loc[v, f"{c}_hi"],
                        f"{c}_raw": raw, f"{c}_raw_lo": curve_raw.loc[v, f"{c}_lo"], f"{c}_raw_hi": curve_raw.loc[v, f"{c}_hi"],
                        f"{c}_diff": b[i]["mean"], f"{c}_diff_lo": b[i]["ci_lo"], f"{c}_diff_hi": b[i]["ci_hi"]})
            assert abs(b[i]["mean"] - (raw - rec)) < 1e-12
            excl0 = b[i]["ci_lo"] > 0 or b[i]["ci_hi"] < 0
            row[f"{c}_rule_a"] = bool(excl0 and b[i]["mean"] > 0.2 * rec)
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("baseline_loss_Q").reset_index(drop=True)

    # rule (b): slope change vs half-width of the locked interval, all points and unflagged (co-primary)
    srows = []
    for c in CHECK_COLS:
        for kind, lo, hi in (("slope_all", "all_lo", "all_hi"), ("slope_unflagged", "unflagged_lo", "unflagged_hi")):
            half = (slopes.loc[c, hi] - slopes.loc[c, lo]) / 2
            change = slopes_raw.loc[c, kind] - slopes.loc[c, kind]
            srows.append(dict(quantity=c, slope=kind, recal=slopes.loc[c, kind], recal_lo=slopes.loc[c, lo],
                              recal_hi=slopes.loc[c, hi], raw=slopes_raw.loc[c, kind], raw_lo=slopes_raw.loc[c, lo],
                              raw_hi=slopes_raw.loc[c, hi], change=change, locked_half_width=half,
                              n_flagged_recal=int(slopes.loc[c, "n_flagged"]),
                              n_flagged_raw=int(slopes_raw.loc[c, "n_flagged"]),
                              rule_b=bool(abs(change) > half)))
    stab = pd.DataFrame(srows)

    a_hits = [(r.variant, c) for r in tab.itertuples() for c in CHECK_COLS if getattr(r, f"{c}_rule_a")]
    b_hits = [(r.quantity, r.slope) for r in stab.itertuples() if r.rule_b]
    material = bool(a_hits or b_hits)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    nonpos = [(r.variant, c) for r in tab.itertuples() for c in CHECK_COLS if getattr(r, f"{c}_recal") <= 0]

    out = dict(spec_commit=SPEC_COMMIT, repo_head_at_run=head, script="pipeline/amendment5_rawF0.py",
               path_deviation="spec names code/amendment5_rawF0.py; repository keeps scripts in pipeline/",
               raw_F0="w_m fixed at 1: F0 = clipren(P_m), CLIP = 1e-3 renormalized (spec's existing clip)",
               reproduction_check=dict(values_compared=n_vals, bit_identical=n_exact),
               optimizer_all_converged=bool(ok_all), n_boot=m.N_BOOT, boot_seed=m.SEED_B, fold_seed=m.SEED_A,
               variants=tab.to_dict("records"), slopes=stab.to_dict("records"),
               rule_a_hits=[list(x) for x in a_hits], rule_b_hits=[list(x) for x in b_hits],
               nonpositive_I_recal=[list(x) for x in nonpos], material=material)
    (m.REP / "amendment5_results.json").write_text(json.dumps(out, indent=1))
    write_report(out, tab, stab, a_hits, b_hits, material, nonpos)
    print(f"material: {material}; rule (a) hits {len(a_hits)}; rule (b) hits {len(b_hits)}")


def f3(x):
    return f"{x:+.3f}".replace("-", "−")


def ci(lo, hi):
    return f"[{f3(lo)}, {f3(hi)}]"


def write_report(out, tab, stab, a_hits, b_hits, material, nonpos):
    L = ["# Amendment 5: human increment against the raw classifier (sensitivity only)", "",
         f"SPEC v1 Amendment 5, LOCKED 2026-10-08, spec commit `{out['spec_commit']}`. "
         f"Repo HEAD at run: `{out['repo_head_at_run']}`.", "",
         "- **Status.** After results, sensitivity only. No locked estimate changes; every `I_recal` below is the "
         "locked H1c value.",
         "- **Deviation from the spec wording.** The spec names `code/amendment5_rawF0.py`. This repository keeps its "
         "scripts in `pipeline/`, so the script is `pipeline/amendment5_rawF0.py`.",
         "- **Raw F0.** F0's single weight is fixed at w_m = 1 (no fitted weight, no recalibration), so F0 is the "
         "classifier vector itself. To avoid log(0) it carries the spec's existing clip: probabilities clipped below "
         "at 1e-3 and renormalized (`clipren`, CLIP = 1e-3), the same treatment every classifier vector already has "
         "in the locked run. The with-human pools are unchanged: two fitted weights, the same 5 folds by image "
         "(seed 20260915), the same clipping.",
         f"- **Intervals.** Image-cluster bootstrap, {out['n_boot']} draws, seed {out['boot_seed']}, percentile 95%, "
         "the full run's draws. The difference `I_raw − I_recal` is bootstrapped per item on the same draws, so its "
         "interval is paired. Slope intervals use the same draws with flags fixed from the point estimates.",
         f"- **Reproduction check.** With the flag off, all {out['reproduction_check']['values_compared']} H1c values "
         "compared (I_h_ind and I_h_pool, mean and interval for 20 variants; Q; both slopes with intervals) are "
         "bit-identical to `reports/full_tables/h1c_curve.csv` and `h1c_slopes.csv`.",
         "- **Individual and pooled differences coincide.** Only F0 changes; the with-human pools do not. "
         "Per item, both increments therefore shift by the same F0 loss difference, and the I_raw − I_recal "
         "columns of the two tables are identical. Rule (a) can still differ between them, because its 20% "
         "threshold is taken on each version's own I_recal.",
         f"- Optimizer converged in every fit: {'yes' if out['optimizer_all_converged'] else 'NO'}.",
         "", "Units: nats per item. Positive increments mean the human lowers out-of-fold loss.", "",
         "## Locked reading (restated)", "",
         "- The difference is expected to be positive wherever the raw classifier is miscalibrated (recalibration "
         "takes credit the human would otherwise get).",
         "- The choice is material, and becomes a numbered item in the reporting standard, if (a) for at least one "
         "variant the paired interval on I_raw − I_recal excludes zero and the difference exceeds 20% of I_recal, "
         "or (b) the slope of I on baseline loss changes by more than the half-width of its locked interval. "
         "Otherwise the item moves to the discussion as a recommendation without a demonstration.", "",
         "## Verdict", ""]
    if material:
        why = []
        if a_hits:
            why.append(f"rule (a) holds for {len(a_hits)} variant-quantity pairs")
        if b_hits:
            why.append("rule (b) holds for " + ", ".join(f"{q} {s}" for q, s in b_hits))
        L.append("**Material.** The choice of no-human term is material under the locked rule: " + "; ".join(why)
                 + ". It becomes a numbered item in the reporting standard.")
    else:
        L.append("**Not material.** Neither rule (a) nor rule (b) holds. The item moves to the discussion as a "
                 "recommendation without a demonstration, and the paper says so.")
    if nonpos:
        L += ["", "Note: I_recal ≤ 0 for " + ", ".join(f"{v} {c}" for v, c in nonpos)
              + "; rule (a) is applied literally (difference > 0.2 × I_recal) there."]
    for c, name in (("I_h_ind", "individual"), ("I_h_pool", "pooled")):
        L += ["", f"## Per variant, {name} `I(h | m)`", "",
              "Ordered by baseline loss Q (selection-set log loss). Rule (a): paired interval excludes zero and "
              "difference > 0.2 × I_recal.", "",
              "| variant | baseline loss Q | I_recal [95% CI] | I_raw [95% CI] | I_raw − I_recal [paired 95% CI] | rule (a) |",
              "|---|---|---|---|---|---|"]
        for r in tab.itertuples():
            g = lambda k: getattr(r, f"{c}_{k}")
            L.append(f"| {r.variant} | {r.baseline_loss_Q:.3f} | {f3(g('recal'))} {ci(g('recal_lo'), g('recal_hi'))} | "
                     f"{f3(g('raw'))} {ci(g('raw_lo'), g('raw_hi'))} | {f3(g('diff'))} {ci(g('diff_lo'), g('diff_hi'))} | "
                     f"{'yes' if g('rule_a') else 'no'} |")
    L += ["", "## Slopes on baseline loss", "",
          "Rule (b): |slope_raw − slope_recal| > half-width of the locked interval. All points and unflagged points "
          "are co-primary, as in H1c.", "",
          "| quantity | slope | I_recal (locked) [95% CI] | I_raw [95% CI] | change | locked half-width | flagged recal / raw | rule (b) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in stab.itertuples():
        L.append(f"| {r.quantity} | {r.slope} | {f3(r.recal)} {ci(r.recal_lo, r.recal_hi)} | {f3(r.raw)} "
                 f"{ci(r.raw_lo, r.raw_hi)} | {f3(r.change)} | {r.locked_half_width:.3f} | "
                 f"{r.n_flagged_recal} / {r.n_flagged_raw} | {'yes' if r.rule_b else 'no'} |")
    L += ["", "Full values: `reports/amendment5_results.json`.", ""]
    (m.REP / "amendment5_REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
