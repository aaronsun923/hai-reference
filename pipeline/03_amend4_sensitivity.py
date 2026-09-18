"""Amendment 4 item 1: H1a and H1b with the strength grid extended to {1, ..., 128}.

Diagnostic only. The locked results (grid {1, 2, 4, 8, 16}) are produced by pipeline/02_full_run.py
and are not touched. This script reuses that module's functions, changing only STRENGTHS, and writes
reports/amend4_sensitivity.json and reports/full_tables/amend4_strengths.csv.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_run", ROOT / "pipeline" / "02_full_run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

EXTENDED = np.array([1, 2, 4, 8, 16, 32, 64, 128], float)


def prepare():
    """Same test set, folds and ratings as the full run."""
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
            P[f"{a}_{lv}"] = m.clipren(g[m.C16].values.astype(float))
    images = np.array(sorted(items.image_name.unique()))
    sel = images[np.sort(np.random.default_rng(m.SEED_A).choice(1200, 300, replace=False))]
    test = np.setdiff1d(images, sel)
    fm = m.fold_map(test)
    sel_mask = items.image_name.isin(sel).values
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
    base = min(P, key=lambda v: m.score(P[v][sel_mask], truth[sel_mask]).mean())
    arch, lv = base.split("_")
    ref = np.log(m.clipren(m.softmax(sum(np.log(P[f"{b}_{lv}"][tidx]) for b in m.ARCH if b != arch))))
    return dict(r=r, ri=r.item.values.astype(int), y=truth[tidx], fold_it=ti.image_name.map(fm).values,
                fold_r=r.fold.values, noise_it=ti.noise_level.values, noise_r=r.noise_level.values,
                img_it=ti.image_name.values, lm=np.log(P[base][tidx]), lr=ref, base=base)


def main():
    d = prepare()
    m.STRENGTHS = EXTENDED
    test, nested, log_test, log_nested = m.h16_human_vectors(d["r"], "structured")
    res = m.h1_engine(d["lm"], d["lr"], test, nested, d["ri"], d["y"], d["fold_it"], d["fold_r"],
                      d["noise_it"], d["noise_r"], label="amend4_extended_grid")
    H1 = {c: m.cluster_boot(res["q"][c].values, d["img_it"])[0] for c in m.QCOLS}
    rows = []
    for tag, log in [("outer", log_test), ("nested", log_nested)]:
        for e in log:
            rows.append(dict(fit=tag, **{f"strength_noise_{nz}": e["strengths"][i] for i, nz in enumerate(m.NOISE)},
                             **{k: v for k, v in e.items() if k != "strengths"}))
    st = pd.DataFrame(rows)
    st.to_csv(m.TAB / "amend4_strengths.csv", index=False)
    alone = float(m.score(test, d["r"].true.values).mean())
    out = dict(grid=EXTENDED.tolist(), baseline_a=d["base"], H1=H1, human_alone_log_loss=alone,
               strength_counts={f"noise_{nz}": st[f"strength_noise_{nz}"].value_counts().sort_index().to_dict()
                                for nz in m.NOISE},
               optimizer_all_converged=bool(res["ok"]),
               diag_mean=float(test[np.arange(len(d["r"])), d["r"].lab.values].mean()),
               accuracy=float((d["r"].lab.values == d["r"].true.values).mean()),
               losses={k: float(v) for k, v in res["losses"].items()})
    (ROOT / "reports" / "amend4_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
