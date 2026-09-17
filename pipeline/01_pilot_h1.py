"""Pilot (SPEC v1 §6, Amendment 2): H1a and H1b on 270 pilot images, plus §9 items 3, 4 and 9.

Inputs: data/h16_deid.csv (step 0) and the four normalized prediction files read from RAW.
Outputs: reports/pilot_results.json, reports/pilot_tables/*, reports/pilot_figures/*,
data/pilot_item_values.csv (item-level values; gitignored).

Implementation readings, all stated in the pilot report:
- Random draws use numpy.random.default_rng(seed) on the sorted image list: selection set
  rng(20260915).choice(1200, 300, replace=False); pilot rng(20260916).choice(900 test, 270,
  replace=False); folds rng(20260915).permutation(270 pilot), fold = position % 5; bootstrap
  rng(20260916).integers(0, 270, (2000, 270)).
- Clip at 0.001 then renormalize (Amendment 1.6), applied to every input vector and to every
  pooled or combined vector before it is scored or pooled again.
- Log-linear pool = normalized product (§4), weights 1 for the reference and for the pooled human.
- Confusion rows are cross-fitted with nesting: a test-fold rating uses rows fitted on the four
  training folds; a training-fold rating used to fit combiner weights uses rows fitted on the
  other three training folds. No image's own outcomes enter any vector it is scored or fitted with.
- Weights are fitted by minimizing mean clipped log loss (L-BFGS-B on the unclipped objective,
  then Nelder-Mead on the clipped objective). Item-level combiners (F0, pooled human, reference)
  are fitted on items; individual-human combiners on ratings.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path.home() / "Desktop" / "hai-reference-raw"
REP = ROOT / "reports"
TAB = REP / "pilot_tables"
FIG = REP / "pilot_figures"

C16 = ["airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
       "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"]
ARCH = ["alexnet", "densenet161", "googlenet", "resnet152", "vgg19"]
LEVELS = ["baseline", "epoch00", "epoch01", "epoch10"]
NOISE = [80, 95, 110, 125]
CONF = ["low", "medium", "high"]
CLIP = 1e-3
SEED_A, SEED_B = 20260915, 20260916
N_BOOT = 2000


def clipren(p):
    p = np.clip(p, CLIP, None)
    return p / p.sum(-1, keepdims=True)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def pool(logs, w):
    """Log-linear pool of log-vectors with weights w, clipped and renormalized."""
    return clipren(softmax(np.tensordot(np.asarray(w, float), np.stack(logs), axes=1)))


def losses(p, y):
    return -np.log(p[np.arange(len(y)), y])


def fit(logs, y, w0):
    X = np.stack(logs)

    def obj(w, clip):
        p = softmax(np.tensordot(w, X, axes=1))
        if clip:
            p = clipren(p)
        return losses(p, y).mean()

    r1 = minimize(obj, np.asarray(w0, float), args=(False,), method="L-BFGS-B")
    r2 = minimize(obj, r1.x, args=(True,), method="Nelder-Mead",
                  options=dict(xatol=1e-6, fatol=1e-10, maxiter=20000))
    return r2.x, bool(r2.success)


# ---------------------------------------------------------------- data
def load():
    h = pd.read_csv(ROOT / "data" / "h16_deid.csv")
    h = h.sort_values(["code", "task_order"])
    dup = h.duplicated(["code", "image_name", "noise_level"], keep="first")
    dropped = h[dup][["image_name", "noise_level"]]
    h = h[~dup]

    items = h[["image_name", "noise_level"]].drop_duplicates().sort_values(["image_name", "noise_level"])
    items = items.reset_index(drop=True)
    assert len(items) == 4800
    key = pd.MultiIndex.from_frame(items)

    P = {}
    for lv in LEVELS:
        d = pd.read_csv(RAW / "2ntrf" / f"hai_{lv}_model_preds_max_normalized.csv")
        d = d[(d.noise_type == "phase") & d.noise_level.isin(NOISE)]
        for a in ARCH:
            g = d[d.model_name == a].set_index(["image_name", "noise_level"]).reindex(key)
            assert g[C16].notna().all().all()
            assert (g.category.values == h.drop_duplicates(["image_name", "noise_level"])
                    .set_index(["image_name", "noise_level"]).reindex(key).image_category.values).all()
            P[f"{a}_{lv}"] = clipren(g[C16].values.astype(float))
    cls = {c: i for i, c in enumerate(C16)}
    truth = (h.drop_duplicates(["image_name", "noise_level"]).set_index(["image_name", "noise_level"])
             .reindex(key).image_category.map(cls).values)
    h = h.assign(item=key.get_indexer(pd.MultiIndex.from_frame(h[["image_name", "noise_level"]])),
                 lab=h.participant_classification.map(cls).values,
                 conf_i=h.confidence.map({c: i for i, c in enumerate(CONF)}).values,
                 noise_i=h.noise_level.map({n: i for i, n in enumerate(NOISE)}).values,
                 true=h.image_category.map(cls).values)
    return h, items, P, truth, dropped


# ---------------------------------------------------------------- confusion rows
def conf_rows(r):
    cnt = np.zeros((4, 16, 3, 16))
    np.add.at(cnt, (r.noise_i.values, r.lab.values, r.conf_i.values, r.true.values), 1)
    return (cnt + 1) / (cnt.sum(-1, keepdims=True) + 16), cnt


def human_vecs(r, rows):
    return clipren(rows[r.noise_i.values, r.lab.values, r.conf_i.values])


def pooled_human(r, hv, n_items):
    """Equal-weight log-linear pool (normalized product) of each item's human rows."""
    z = np.zeros((n_items, 16))
    np.add.at(z, r.item.values, np.log(hv))
    return clipren(softmax(z))


# ---------------------------------------------------------------- main
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    h, items, P, truth, dropped = load()
    images = np.array(sorted(items.image_name.unique()))
    assert len(images) == 1200

    sel = images[np.sort(np.random.default_rng(SEED_A).choice(1200, 300, replace=False))]
    test = np.setdiff1d(images, sel)
    pilot = test[np.sort(np.random.default_rng(SEED_B).choice(len(test), 270, replace=False))]
    perm = np.random.default_rng(SEED_A).permutation(len(pilot))
    fold_of_image = {pilot[j]: pos % 5 for pos, j in enumerate(perm)}
    assert len(set(sel) & set(pilot)) == 0 and len(test) == 900

    # ---- §9.3 baseline (a) on the selection set
    sel_mask = items.image_name.isin(sel).values
    sel_rows = []
    for v, p in P.items():
        a, lv = v.split("_")
        sel_rows.append(dict(variant=v, architecture=a, level=lv,
                             selection_log_loss=losses(p[sel_mask], truth[sel_mask]).mean(),
                             selection_accuracy=(p[sel_mask].argmax(1) == truth[sel_mask]).mean()))
    sel_tab = pd.DataFrame(sel_rows).sort_values("selection_log_loss").reset_index(drop=True)
    sel_tab.insert(0, "rank", range(1, 21))
    sel_tab.to_csv(TAB / "selection_set.csv", index=False)
    base = sel_tab.variant[0]
    arch_a, lv_a = base.split("_")
    ref_members = [f"{a}_{lv_a}" for a in ARCH if a != arch_a]

    # ---- pilot items and ratings
    pil_items = items[items.image_name.isin(pilot)].copy()
    pil_idx = pil_items.index.values                         # global item ids
    n_it = len(pil_idx)
    loc = {g: i for i, g in enumerate(pil_idx)}
    pil_items["fold"] = pil_items.image_name.map(fold_of_image).values
    y = truth[pil_idx]
    fold_it = pil_items.fold.values
    r = h[h.image_name.isin(pilot)].copy()
    r["item"] = r.item.map(loc)
    r["fold"] = r.image_name.map(fold_of_image)
    assert r.item.notna().all() and (r.true.values == y[r.item.values]).all()

    la = np.log(P[base][pil_idx])
    ref = clipren(softmax(sum(np.log(P[m][pil_idx]) for m in ref_members)))
    lr = np.log(ref)

    # outputs per item (loss of each forecast) and per rating
    L = {k: np.full(n_it, np.nan) for k in ["raw_a", "F0", "F_ar", "F_ah_pool", "F_arh_pool", "hpool_alone"]}
    Lr = {k: np.full(len(r), np.nan) for k in ["F_ah_ind", "F_arh_ind", "h_alone"]}
    PR = {k: np.zeros((n_it, 16)) for k in ["raw_a", "F0", "F_ar", "F_ah_pool", "F_arh_pool"]}
    PRr = {k: np.zeros((len(r), 16)) for k in ["F_ah_ind", "h_alone"]}
    L["raw_a"] = losses(P[base][pil_idx], y)
    PR["raw_a"] = P[base][pil_idx]
    weights, ok_all = [], True
    rows_by_fold = {}

    r_pos = np.arange(len(r))
    for k in range(5):
        tr_it, te_it = fold_it != k, fold_it == k
        tr_r, te_r = (r.fold != k).values, (r.fold == k).values

        # test-fold human vectors: rows fitted on the four training folds
        rows_k, cnt_k = conf_rows(r[tr_r])
        rows_by_fold[k] = cnt_k.sum(-1)
        hv = np.zeros((len(r), 16))
        hv[te_r] = human_vecs(r[te_r], rows_k)
        # training-fold human vectors: nested, rows fitted on the other three training folds
        for j in range(5):
            if j == k:
                continue
            inner = ((r.fold != k) & (r.fold != j)).values
            rows_kj, _ = conf_rows(r[inner])
            m = (r.fold == j).values
            hv[m] = human_vecs(r[m], rows_kj)
        hp = pooled_human(r, hv, n_it)       # uses only same-item vectors, so fold-respecting
        lh, lhp = np.log(hv), np.log(hp)
        ri = r.item.values

        w_F0, s0 = fit([la[tr_it]], y[tr_it], [1.0])
        w_ar, s1 = fit([la[tr_it], lr[tr_it]], y[tr_it], [0.5, 0.5])
        w_ahp, s2 = fit([la[tr_it], lhp[tr_it]], y[tr_it], [1.0, 0.1])
        w_arhp, s3 = fit([la[tr_it], lr[tr_it], lhp[tr_it]], y[tr_it], [0.5, 0.5, 0.1])
        w_ahi, s4 = fit([la[ri[tr_r]], lh[tr_r]], y[ri[tr_r]], [1.0, 0.5])
        w_arhi, s5 = fit([la[ri[tr_r]], lr[ri[tr_r]], lh[tr_r]], y[ri[tr_r]], [0.5, 0.5, 0.5])
        ok_all &= all([s0, s1, s2, s3, s4, s5])

        for name, w, logs in [("F0", w_F0, [la]), ("F_ar", w_ar, [la, lr]),
                              ("F_ah_pool", w_ahp, [la, lhp]), ("F_arh_pool", w_arhp, [la, lr, lhp])]:
            p = pool([x[te_it] for x in logs], w)
            PR[name][te_it] = p
            L[name][te_it] = losses(p, y[te_it])
        L["hpool_alone"][te_it] = losses(hp[te_it], y[te_it])
        te_ri = ri[te_r]
        for name, w, logs in [("F_ah_ind", w_ahi, [la[te_ri], lh[te_r]]),
                              ("F_arh_ind", w_arhi, [la[te_ri], lr[te_ri], lh[te_r]])]:
            p = pool(logs, w)
            Lr[name][te_r] = losses(p, y[te_ri])
            if name in PRr:
                PRr[name][te_r] = p
        Lr["h_alone"][te_r] = losses(hv[te_r], y[te_ri])
        PRr["h_alone"][te_r] = hv[te_r]

        for combo, w in [("F0 (a)", w_F0), ("F(a, r_a)", w_ar), ("F(a, h) individual", w_ahi),
                         ("F(a, h) pooled", w_ahp), ("F(a, r_a, h) individual", w_arhi),
                         ("F(a, r_a, h) pooled", w_arhp)]:
            names = {1: ["w_m"], 2: ["w_m", "w_r" if "r_a" in combo else "w_h"], 3: ["w_m", "w_r", "w_h"]}[len(w)]
            weights.append(dict(fold=k + 1, combiner=combo, **{n: float(x) for n, x in zip(names, w)},
                                n_train_items=int(tr_it.sum()), n_train_ratings=int(tr_r.sum())))

    assert all(np.isfinite(v).all() for v in L.values()) and all(np.isfinite(v).all() for v in Lr.values())
    wt = pd.DataFrame(weights)
    wt.to_csv(TAB / "combiner_weights.csv", index=False)

    # ---- per-item quantities
    ri = r.item.values
    ind_ah = np.bincount(ri, L["F0"][ri] - Lr["F_ah_ind"], n_it) / np.bincount(ri, minlength=n_it)
    ind_arh = np.bincount(ri, L["F_ar"][ri] - Lr["F_arh_ind"], n_it) / np.bincount(ri, minlength=n_it)
    Q = pd.DataFrame(dict(
        image_name=pil_items.image_name.values, noise_level=pil_items.noise_level.values, fold=fold_it,
        n_humans=np.bincount(ri, minlength=n_it),
        I_h_a_individual=ind_ah, I_h_a_pooled=L["F0"] - L["F_ah_pool"],
        I_r_a=L["F0"] - L["F_ar"]))
    Q["D_h_a_individual"] = Q.I_h_a_individual - Q.I_r_a
    Q["D_h_a_pooled"] = Q.I_h_a_pooled - Q.I_r_a
    Q["I_h_a_r_individual"] = ind_arh
    Q["I_h_a_r_pooled"] = L["F_ar"] - L["F_arh_pool"]
    Q.to_csv(ROOT / "data" / "pilot_item_values.csv", index=False)

    # ---- cluster bootstrap over pilot images
    img_codes, img_of_item = np.unique(Q.image_name.values, return_inverse=True)
    n_img = len(img_codes)
    draws = np.random.default_rng(SEED_B).integers(0, n_img, (N_BOOT, n_img))
    cnt_img = np.bincount(img_of_item, minlength=n_img)

    def boot(v):
        s = np.bincount(img_of_item, v, n_img)
        est = s.sum() / cnt_img.sum()
        b = s[draws].sum(1) / cnt_img[draws].sum(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return dict(mean=float(est), ci_lo=float(lo), ci_hi=float(hi), share_boot_le_0=float((b <= 0).mean()))

    res = {}
    for col in ["I_h_a_individual", "I_h_a_pooled", "I_r_a", "D_h_a_individual", "D_h_a_pooled",
                "I_h_a_r_individual", "I_h_a_r_pooled"]:
        res[col] = boot(Q[col].values)

    # item-level mean losses (descriptive)
    item_mean = lambda v: float(np.mean(np.bincount(ri, v, n_it) / np.bincount(ri, minlength=n_it)))  # noqa: E731
    loss_tab = {
        "raw baseline (a)": float(L["raw_a"].mean()), "F0: recalibrated (a)": float(L["F0"].mean()),
        "F(a, r_a)": float(L["F_ar"].mean()),
        "human confusion row alone (individual, item mean)": item_mean(Lr["h_alone"]),
        "pooled human alone": float(L["hpool_alone"].mean()),
        "F(a, h) individual (item mean)": item_mean(Lr["F_ah_ind"]),
        "F(a, h) pooled": float(L["F_ah_pool"].mean()),
        "F(a, r_a, h) individual (item mean)": item_mean(Lr["F_arh_ind"]),
        "F(a, r_a, h) pooled": float(L["F_arh_pool"].mean()),
    }
    acc = dict(raw_a=float((PR["raw_a"].argmax(1) == y).mean()),
               human_individual=float((r.lab.values == r.true.values).mean()))

    # ---- §9.4 confusion rows: display fit on all 270 pilot images + fold deviations
    rows_all, cnt_all = conf_rows(r)
    n_all = cnt_all.sum(-1)
    maxdev = np.zeros((4, 3))
    for k in range(5):
        rk, _ = conf_rows(r[(r.fold != k).values])
        maxdev = np.maximum(maxdev, np.abs(rk - rows_all).max(axis=(1, 3)))
    recs = []
    for ni, nz in enumerate(NOISE):
        for ci, cf in enumerate(CONF):
            for li, lab in enumerate(C16):
                recs.append(dict(noise_level=nz, confidence=cf, human_label=lab,
                                 n_all_pilot=int(n_all[ni, li, ci]),
                                 n_train_min_over_folds=int(min(rows_by_fold[k][ni, li, ci] for k in range(5))),
                                 **{f"P_true_{c}": float(rows_all[ni, li, ci, j]) for j, c in enumerate(C16)}))
    cr = pd.DataFrame(recs)
    cr.to_csv(TAB / "confusion_rows_pilot.csv", index=False)

    # ---- calibration (all 16 class probabilities vs one-hot outcome), per noise level
    edges = np.array([0, .01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001])

    def calib(prob, yy, noise):
        out = []
        onehot = np.eye(16)[yy]
        for nz in NOISE:
            m = noise == nz
            p, o = prob[m].ravel(), onehot[m].ravel()
            b = np.digitize(p, edges) - 1
            for bi in range(len(edges) - 1):
                s = b == bi
                if s.sum():
                    out.append(dict(noise_level=nz, bin_lo=edges[bi], bin_hi=min(edges[bi + 1], 1.0),
                                    n=int(s.sum()), mean_pred=float(p[s].mean()), obs_freq=float(o[s].mean())))
        return pd.DataFrame(out)

    noise_it = pil_items.noise_level.values
    noise_r = r.noise_level.values
    cal = pd.concat([
        calib(PRr["h_alone"], r.true.values, noise_r).assign(forecast="human confusion row (individual)"),
        calib(PR["raw_a"], y, noise_it).assign(forecast="raw baseline (a)"),
        calib(PR["F0"], y, noise_it).assign(forecast="F0: recalibrated (a)"),
        calib(PRr["F_ah_ind"], r.true.values, noise_r).assign(forecast="F(a, h) individual"),
        calib(PR["F_ah_pool"], y, noise_it).assign(forecast="F(a, h) pooled"),
        calib(PR["F_ar"], y, noise_it).assign(forecast="F(a, r_a)"),
        calib(PR["F_arh_pool"], y, noise_it).assign(forecast="F(a, r_a, h) pooled"),
    ])
    cal.to_csv(TAB / "calibration_bins.csv", index=False)
    figures(cal)

    out = dict(
        baseline_a=base, runner_up=sel_tab.variant[1], reference_members=ref_members,
        selection_images=len(sel), test_images=len(test), pilot_images=len(pilot), pilot_items=int(n_it),
        pilot_ratings=int(len(r)), pilot_persons=int(r.code.nunique()),
        humans_per_item=Q.n_humans.value_counts().sort_index().to_dict(),
        ratings_per_person_pilot=dict(min=int(r.groupby("code").size().min()),
                                      median=float(r.groupby("code").size().median()),
                                      max=int(r.groupby("code").size().max())),
        duplicates_dropped_total=int(len(dropped)),
        duplicates_dropped_in_pilot=int(dropped.image_name.isin(pilot).sum()),
        images_per_fold=pd.Series(list(fold_of_image.values())).value_counts().sort_index().to_dict(),
        items_per_fold=pd.Series(fold_it).value_counts().sort_index().to_dict(),
        ratings_per_fold=r.fold.value_counts().sort_index().to_dict(),
        optimizer_all_converged=bool(ok_all),
        negative_weights=wt[(wt[["w_m", "w_r", "w_h"]] < 0).any(axis=1)].to_dict("records"),
        results=res, mean_losses=loss_tab, accuracy=acc,
        confusion_rows=dict(rows_total=48 * 4, rows_n0=int((n_all == 0).sum()),
                            rows_n_lt30=int((n_all < 30).sum()),
                            max_abs_fold_deviation_by_noise_conf=maxdev.round(4).tolist()),
    )
    (REP / "pilot_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]


def panel_figure(cal, forecasts, title, path):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), dpi=160, sharex=True, sharey=True, facecolor=SURF)
    for ax, nz in zip(axes, NOISE):
        ax.set_facecolor(SURF)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        for i, f in enumerate(forecasts):
            d = cal[(cal.forecast == f) & (cal.noise_level == nz) & (cal.n >= 20)]
            ax.plot(d.mean_pred, d.obs_freq, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                    mec=SURF, mew=1.2, label=f, zorder=3 + i)
        ax.set_title(f"noise {nz}", color=INK, fontsize=10, loc="left")
        ax.grid(color=GRID, lw=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    axes[0].set_ylabel("observed frequency", color=INK2, fontsize=9)
    fig.supxlabel("forecast probability (all 16 class probabilities, binned)", color=INK2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    if len(forecasts) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=len(forecasts),
                   frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def figures(cal):
    panel_figure(cal, ["human confusion row (individual)"],
                 "Calibration of cross-fitted human confusion rows, pilot images (bins with n ≥ 20)",
                 FIG / "calibration_human_rows.png")
    panel_figure(cal, ["raw baseline (a)", "F0: recalibrated (a)", "F(a, h) individual"],
                 "Calibration: baseline (a), recalibrated F0, and F(a, h) individual",
                 FIG / "calibration_baseline_individual.png")
    panel_figure(cal, ["F(a, h) pooled", "F(a, r_a)", "F(a, r_a, h) pooled"],
                 "Calibration: pooled human and reference combiners",
                 FIG / "calibration_pooled_reference.png")


if __name__ == "__main__":
    main()
