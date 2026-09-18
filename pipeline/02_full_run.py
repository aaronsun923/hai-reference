"""Full run (SPEC v1 with Amendments 1-3): H1a, H1b, H1c, H2, H3a-H3c and robustness items 1-7.

Inputs: data/*_deid.csv (step 0) and the four normalized 2ntrf prediction files read from RAW.
Outputs: reports/full_results.json, reports/full_tables/*, reports/full_figures/*,
data/full_*.csv (item-, rating- and participant-level values; gitignored).

Readings confirmed in Amendment 3 item 3 (random draws, nested fitting, weight-fitting units,
optimizer) are carried over from pipeline/01_pilot_h1.py unchanged. Readings new to the full run
are listed in the full report.
"""
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path.home() / "Desktop" / "hai-reference-raw"
REP = ROOT / "reports"
TAB = REP / "full_tables"
FIG = REP / "full_figures"
DATA = ROOT / "data"

C16 = ["airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
       "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"]
CLS = {c: i for i, c in enumerate(C16)}
ARCH = ["alexnet", "densenet161", "googlenet", "resnet152", "vgg19"]
LEVELS = ["baseline", "epoch00", "epoch01", "epoch10"]
NOISE = [80, 95, 110, 125]
CONF = ["low", "medium", "high"]
TLEVELS = ["below human", "at human", "above human"]
STRENGTHS = np.array([1, 2, 4, 8, 16], float)
CLIP = 1e-3
SEED_A, SEED_B = 20260915, 20260916
N_BOOT = 2000
K = 5


# ================================================================ scoring and pooling
def clipren(p):
    p = np.clip(p, CLIP, None)
    return p / p.sum(-1, keepdims=True)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def score(p, y, rule="log"):
    if rule == "log":
        return -np.log(p[np.arange(len(y)), y])
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1
    return ((p - oh) ** 2).sum(1)


def pool(logs, w):
    return clipren(softmax(np.tensordot(np.asarray(w, float), np.stack(logs), axes=1)))


def fit(logs, y, w0, rule="log"):
    X = np.stack(logs)

    def obj(w, clip):
        p = softmax(np.tensordot(w, X, axes=1))
        return score(clipren(p) if clip else p, y, rule).mean()

    r1 = minimize(obj, np.asarray(w0, float), args=(False,), method="L-BFGS-B")
    r2 = minimize(obj, r1.x, args=(True,), method="Nelder-Mead",
                  options=dict(xatol=1e-6, fatol=1e-10, maxiter=20000))
    return r2.x, bool(r2.success)


def fold_map(images, seed=SEED_A):
    images = np.array(sorted(images))
    perm = np.random.default_rng(seed).permutation(len(images))
    return {images[j]: pos % K for pos, j in enumerate(perm)}


# ================================================================ ImageNet-16H human vectors
def counts(r):
    n = np.zeros((4, 16, 3, 16))
    np.add.at(n, (r.noise_i.values, r.lab.values, r.conf_i.values, r.true.values), 1)
    return n


def rows_structured(n, s_by_noise):
    """Amendment 3.1: prior = (label, confidence) row pooled across noise levels; strength per noise."""
    N = n.sum(0)
    tot = N.sum(-1, keepdims=True)
    prior = np.where(tot > 0, N / np.where(tot > 0, tot, 1), 1 / 16)
    s = np.asarray(s_by_noise, float)[:, None, None, None]
    return (n + s * prior[None]) / (n.sum(-1, keepdims=True) + s), int((tot[..., 0] == 0).sum())


def rows_alpha1(n):
    return (n + 1) / (n.sum(-1, keepdims=True) + 16)


def rows_uniform_c(r):
    rows = np.zeros((4, 16, 3, 16))
    for ni in range(4):
        for ci in range(3):
            m = (r.noise_i.values == ni) & (r.conf_i.values == ci)
            c = (r.lab.values[m] == r.true.values[m]).mean() if m.any() else 1 / 16
            rows[ni, :, ci, :] = (1 - c) / 15
            rows[ni, np.arange(16), ci, np.arange(16)] = c
    return rows


def vecs(r, rows):
    return clipren(rows[r.noise_i.values, r.lab.values, r.conf_i.values])


def fit_h16_rows(tr, rule, log):
    """Rows fitted on ratings `tr`. Structured rule chooses strength per noise level by inner CV."""
    if rule == "alpha1":
        return rows_alpha1(counts(tr))
    if rule == "uniform_c":
        return rows_uniform_c(tr)
    inner = tr.image_name.map(fold_map(tr.image_name.unique())).values
    loss = np.zeros((4, len(STRENGTHS)))
    for j in range(K):
        n = counts(tr[inner != j])
        held = tr[inner == j]
        for si, s in enumerate(STRENGTHS):
            rows, _ = rows_structured(n, [s] * 4)
            ll = score(vecs(held, rows), held.true.values)
            loss[:, si] += np.bincount(held.noise_i.values, ll, 4)
    s_best = STRENGTHS[loss.argmin(1)]
    rows, n_empty = rows_structured(counts(tr), s_best)
    log.append(dict(strengths=s_best.tolist(), empty_prior_rows=n_empty))
    return rows


def h16_human_vectors(r, rule):
    """Cross-fitted vectors: test[k] for fold-k ratings; nested[k] for the other folds' ratings."""
    fold = r.fold.values
    test = np.zeros((len(r), 16))
    nested = {k: np.zeros((len(r), 16)) for k in range(K)}
    log_test, log_nested = [], []
    for k in range(K):
        tlog = []
        rows = fit_h16_rows(r[fold != k], rule, tlog)
        test[fold == k] = vecs(r[fold == k], rows)
        if tlog:
            log_test.append(dict(outer_fold=k + 1, **tlog[0]))
        for j in range(K):
            if j == k:
                continue
            nlog = []
            rows_kj = fit_h16_rows(r[(fold != k) & (fold != j)], rule, nlog)
            nested[k][fold == j] = vecs(r[fold == j], rows_kj)
            if nlog:
                log_nested.append(dict(outer_fold=k + 1, nested_fold=j + 1, **nlog[0]))
    return test, nested, log_test, log_nested


def pooled(ri, hv, n_items):
    z = np.zeros((n_items, 16))
    np.add.at(z, ri, np.log(hv))
    return clipren(softmax(z))


# ================================================================ H1 engine
def h1_engine(lm, lr, hv_test, hv_nested, ri, y, fold_it, fold_r, noise_it, noise_r,
              rule="log", per_noise=False, label=""):
    n_it, n_r = len(y), len(ri)
    out = {k: np.full(n_it, np.nan) for k in ["F0", "Fr", "Fhp", "Frhp"]}
    outr = {k: np.full(n_r, np.nan) for k in ["Fh", "Frh"]}
    probs = {k: np.zeros((n_it, 16)) for k in ["F0", "Fr", "Fhp", "Frhp"]}
    probs_r = {"Fh": np.zeros((n_r, 16))}
    wrows, ok = [], True
    hp_test = pooled(ri, hv_test, n_it)
    groups = [(None, None)] if not per_noise else [(nz, nz) for nz in NOISE]
    for k in range(K):
        hp_nest = pooled(ri[fold_r != k], hv_nested[k][fold_r != k], n_it)
        with np.errstate(divide="ignore"):   # nested[k] is zero on fold-k ratings, which are never used
            lh_test, lh_nest = np.log(hv_test), np.log(hv_nested[k])
        lhp_test, lhp_nest = np.log(hp_test), np.log(hp_nest)
        for nz, _ in groups:
            tr_it = (fold_it != k) & (True if nz is None else noise_it == nz)
            te_it = (fold_it == k) & (True if nz is None else noise_it == nz)
            tr_r = (fold_r != k) & (True if nz is None else noise_r == nz)
            te_r = (fold_r == k) & (True if nz is None else noise_r == nz)
            specs = [
                ("F0", "items", [lm], [lm], [1.0]),
                ("Fr", "items", [lm, lr], [lm, lr], [0.7, 0.3]),
                ("Fhp", "items", [lm, lhp_nest], [lm, lhp_test], [1.0, 0.5]),
                ("Frhp", "items", [lm, lr, lhp_nest], [lm, lr, lhp_test], [0.7, 0.3, 0.5]),
            ]
            for name, _, tr_logs, te_logs, w0 in specs:
                w, s = fit([x[tr_it] for x in tr_logs], y[tr_it], w0, rule)
                ok &= s
                p = pool([x[te_it] for x in te_logs], w)
                probs[name][te_it] = p
                out[name][te_it] = score(p, y[te_it], rule)
                wrows.append(dict(run=label, fold=k + 1, noise=nz or "all", combiner=name, weights=w.tolist()))
            ri_tr, ri_te = ri[tr_r], ri[te_r]
            for name, tr_logs, te_logs, w0 in [
                ("Fh", [lm[ri_tr], lh_nest[tr_r]], [lm[ri_te], lh_test[te_r]], [1.0, 1.0]),
                ("Frh", [lm[ri_tr], lr[ri_tr], lh_nest[tr_r]], [lm[ri_te], lr[ri_te], lh_test[te_r]], [0.7, 0.3, 1.0]),
            ]:
                w, s = fit(tr_logs, y[ri_tr], w0, rule)
                ok &= s
                p = pool(te_logs, w)
                if name in probs_r:
                    probs_r[name][te_r] = p
                outr[name][te_r] = score(p, y[ri_te], rule)
                wrows.append(dict(run=label, fold=k + 1, noise=nz or "all", combiner=name, weights=w.tolist()))
    assert all(np.isfinite(v).all() for v in out.values()) and all(np.isfinite(v).all() for v in outr.values())
    cnt = np.bincount(ri, minlength=n_it)
    q = pd.DataFrame(dict(
        I_h_ind=np.bincount(ri, out["F0"][ri] - outr["Fh"], n_it) / cnt,
        I_h_pool=out["F0"] - out["Fhp"],
        I_r=out["F0"] - out["Fr"],
        I_hr_ind=np.bincount(ri, out["Fr"][ri] - outr["Frh"], n_it) / cnt,
        I_hr_pool=out["Fr"] - out["Frhp"]))
    q["D_ind"] = q.I_h_ind - q.I_r
    q["D_pool"] = q.I_h_pool - q.I_r
    rating_inc = out["F0"][ri] - outr["Fh"]
    return dict(q=q, rating_inc=rating_inc, weights=wrows, ok=ok, probs=probs, probs_r=probs_r,
                losses={k: float(v.mean()) for k, v in out.items()} | {k: float(v.mean()) for k, v in outr.items()},
                hp_test=hp_test)


QCOLS = ["I_h_ind", "I_h_pool", "I_r", "D_ind", "D_pool", "I_hr_ind", "I_hr_pool"]


def cluster_boot(values, groups, seed=SEED_B, n_boot=N_BOOT, one_sided=False):
    """Mean over units; bootstrap resamples clusters. Returns estimate, 95% CI, draws."""
    g_codes, g = np.unique(groups, return_inverse=True)
    n_g = len(g_codes)
    draws = np.random.default_rng(seed).integers(0, n_g, (n_boot, n_g))
    cnt = np.bincount(g, minlength=n_g)
    V = np.atleast_2d(values.T).T if values.ndim == 1 else values
    res = []
    for col in range(V.shape[1]):
        s = np.bincount(g, V[:, col], n_g)
        est = s.sum() / cnt.sum()
        b = s[draws].sum(1) / cnt[draws].sum(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        res.append(dict(mean=float(est), ci_lo=float(lo), ci_hi=float(hi),
                        lower_one_sided_95=float(np.percentile(b, 5)), share_le_0=float((b <= 0).mean())))
    return res


# ================================================================ H1c leverage rule
def ols_flags(x, yv):
    n, p = len(x), 2
    xc = x - x.mean()
    b1 = (xc * (yv - yv.mean())).sum() / (xc ** 2).sum()
    beta = np.array([yv.mean() - b1 * x.mean(), b1])
    lev = 1 / n + xc ** 2 / (xc ** 2).sum()          # simple-regression hat values
    resid = yv - beta[0] - beta[1] * x
    s2 = (resid ** 2).sum() / (n - p)
    cook = resid ** 2 / (p * s2) * lev / (1 - lev) ** 2
    flag = (lev > 2 * p / n) | (cook > 4 / n)
    return beta, lev, cook, flag


def slope(x, yv):
    xc = x - x.mean()
    return float((xc * (yv - yv.mean())).sum() / (xc ** 2).sum())


# ================================================================ Tejeda
def load_tejeda(design):
    d = pd.read_csv(DATA / f"tejeda_{design}_deid.csv")
    g = d.groupby("code")
    dup_tn = g.task_number.apply(lambda s: s.duplicated().any())
    rep = d.groupby(["code", "image_name"]).size().groupby("code").max() > (1 if design == "concurrent" else 2)
    few = g.model_on.apply(lambda s: (s == 0).sum() < 30)
    two_tab = sorted(dup_tn[dup_tn & rep].index)
    few_unassisted = sorted(few[few & ~few.index.isin(two_tab)].index)
    d = d[~d.code.isin(two_tab + few_unassisted)].copy()
    d["lab"] = d.participant_classification.map(CLS)
    d["true"] = d.image_category.map(CLS)
    d["conf_i"] = d.confidence.map({c: i for i, c in enumerate(CONF)})
    d["lv"] = d.model_performance_level.map({l: i for i, l in enumerate(TLEVELS)})
    M = clipren(d[[f"model_pred_{c}" for c in C16]].values.astype(float))
    d = d.reset_index(drop=True)
    return d, M, dict(two_tab=two_tab, few_unassisted=few_unassisted)


def tcounts(t):
    n = np.zeros((16, 3, 16))
    np.add.at(n, (t.lab.values, t.conf_i.values, t.true.values), 1)
    return n


def tvecs(t, rows):
    return clipren(rows[t.lab.values, t.conf_i.values])


def t_uniform_c(t):
    rows = np.zeros((16, 3, 16))
    for ci in range(3):
        m = t.conf_i.values == ci
        c = (t.lab.values[m] == t.true.values[m]).mean() if m.any() else 1 / 16
        rows[:, ci, :] = (1 - c) / 15
        rows[np.arange(16), ci, np.arange(16)] = c
    return rows


def tejeda_rows(tr, rule, log):
    """Row sets for one design fitted on training trials `tr`: U (unassisted), A[lv] (assisted)."""
    un, asg = tr[tr.model_on == 0], tr[tr.model_on == 1]
    if rule == "uniform_c":
        return t_uniform_c(un), {lv: t_uniform_c(asg[asg.lv == lv]) for lv in range(3)}
    if rule == "alpha1":
        nU = tcounts(un)
        U = (nU + 1) / (nU.sum(-1, keepdims=True) + 16)
        nP = tcounts(asg)
        P = (nP + 1) / (nP.sum(-1, keepdims=True) + 16)
        A, fb = {}, 0
        for lv in range(3):
            nA = tcounts(asg[asg.lv == lv])
            rows = (nA + 1) / (nA.sum(-1, keepdims=True) + 16)
            small = nA.sum(-1) < 30          # Amendment 1.3 fallback, kept for the α = 1 rule
            rows[small] = P[small]
            fb += int(small.sum())
            A[lv] = rows
        log.append(dict(fallback_rows=fb))
        return U, A
    inner = tr.image_name.map(fold_map(tr.image_name.unique())).values
    # unassisted: uniform prior, strength by inner CV
    lossU = np.zeros(len(STRENGTHS))
    for j in range(K):
        nU = tcounts(un[inner[tr.model_on.values == 0] != j])
        held = un[inner[tr.model_on.values == 0] == j]
        for si, s in enumerate(STRENGTHS):
            rows = (nU + s / 16) / (nU.sum(-1, keepdims=True) + s)
            lossU[si] += score(tvecs(held, rows), held.true.values).sum()
    sU = STRENGTHS[lossU.argmin()]
    # assisted per level: prior = unassisted row (strength sU) fitted on the same training trials
    lossA = np.zeros((3, len(STRENGTHS)))
    inner_a = inner[tr.model_on.values == 1]
    for j in range(K):
        nU = tcounts(un[inner[tr.model_on.values == 0] != j])
        Uj = (nU + sU / 16) / (nU.sum(-1, keepdims=True) + sU)
        for lv in range(3):
            a_tr = asg[(inner_a != j) & (asg.lv.values == lv)]
            held = asg[(inner_a == j) & (asg.lv.values == lv)]
            nA = tcounts(a_tr)
            for si, s in enumerate(STRENGTHS):
                rows = (nA + s * Uj) / (nA.sum(-1, keepdims=True) + s)
                lossA[lv, si] += score(tvecs(held, rows), held.true.values).sum()
    sA = STRENGTHS[lossA.argmin(1)]
    nU = tcounts(un)
    U = (nU + sU / 16) / (nU.sum(-1, keepdims=True) + sU)
    A = {}
    for lv in range(3):
        nA = tcounts(asg[asg.lv == lv])
        A[lv] = (nA + sA[lv] * U) / (nA.sum(-1, keepdims=True) + sA[lv])
    log.append(dict(strength_unassisted=float(sU), **{f"strength_assisted_{TLEVELS[lv]}": float(sA[lv]) for lv in range(3)}))
    return U, A


def tejeda_vectors(d, rule):
    fm = fold_map(d.image_name.unique())
    fold = d.image_name.map(fm).values
    hv = np.zeros((len(d), 16))
    logs = []
    for k in range(K):
        lg = []
        U, A = tejeda_rows(d[fold != k], rule, lg)
        te = d[fold == k]
        v = np.zeros((len(te), 16))
        un = te.model_on.values == 0
        v[un] = tvecs(te[un], U)
        for lv in range(3):
            m = (~un) & (te.lv.values == lv)
            v[m] = tvecs(te[m], A[lv])
        hv[fold == k] = v
        if lg:
            logs.append(dict(outer_fold=k + 1, **lg[0]))
    return hv, logs


def h3_participants(d, M, hv, design, rule="log"):
    G = score(M, d.true.values, rule) - score(hv, d.true.values, rule)
    d = d.assign(G=G)
    rng = np.random.default_rng(SEED_B)
    rows = []
    for code, g in d.sort_values(["code", "task_order"]).groupby("code", sort=True):
        un, asg = g[g.model_on == 0], g[g.model_on == 1]
        if design == "sequential":
            imgs = np.array(sorted(un.image_name.unique()))
            perm = rng.permutation(len(imgs))
            h1 = set(imgs[perm[: len(imgs) // 2]])
            u1, u2 = un[un.image_name.isin(h1)], un[~un.image_name.isin(h1)]
            b2 = asg[~asg.image_name.isin(h1)]
        else:
            perm = rng.permutation(len(un))
            u1, u2 = un.iloc[perm[: len(un) // 2]], un.iloc[perm[len(un) // 2:]]
            b2 = asg
        rows.append(dict(code=code, level=g.model_performance_level.iloc[0], n_unassisted=len(un), n_assisted=len(asg),
                         half1=len(u1), half2=len(u2), IA=un.G.mean(), IB=asg.G.mean(), IA1=u1.G.mean(),
                         IA2=u2.G.mean(), IB_half2=b2.G.mean()))
    p = pd.DataFrame(rows)
    p["Delta"] = p.IB - p.IA
    p["Delta_half2"] = p.IB_half2 - p.IA2
    return p


def h3_stats(p, one_sided_b=True):
    rng_draws = {}
    out = []
    for lv in TLEVELS:
        g = p[p.level == lv].reset_index(drop=True)
        n = len(g)
        draws = np.random.default_rng(SEED_B).integers(0, n, (N_BOOT, n))
        rng_draws[lv] = draws

        def boot_mean(col):
            v = g[col].values
            b = v[draws].mean(1)
            return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), np.percentile(b, 5)

        dm, dlo, dhi, _ = boot_mean("Delta")
        bm, blo, bhi, b5 = boot_mean("IB")
        am, alo, ahi, _ = boot_mean("IA")
        seen_y = "IB_half2"
        bs = slope(g.IA1.values, g[seen_y].values)
        bu = slope(g.IA1.values, g.IA2.values)
        bd = slope(g.IA1.values, g.Delta_half2.values)
        diffs, dhs, bss, bus = [], [], [], []
        for dr in draws:
            x = g.IA1.values[dr]
            if np.ptp(x) == 0:
                continue
            s1, s2 = slope(x, g[seen_y].values[dr]), slope(x, g.IA2.values[dr])
            diffs.append(s1 - s2)
            bss.append(s1)
            bus.append(s2)
            dhs.append(slope(x, g.Delta_half2.values[dr]))
        out.append(dict(level=lv, n=n, IA=am, IA_lo=alo, IA_hi=ahi, Delta=dm, Delta_lo=dlo, Delta_hi=dhi,
                        IB=bm, IB_lo=blo, IB_hi=bhi, IB_lower_one_sided_95=b5,
                        beta_seen=bs, beta_seen_lo=np.percentile(bss, 2.5), beta_seen_hi=np.percentile(bss, 97.5),
                        beta_unseen=bu, beta_unseen_lo=np.percentile(bus, 2.5), beta_unseen_hi=np.percentile(bus, 97.5),
                        beta_diff=bs - bu, beta_diff_lo=np.percentile(diffs, 2.5), beta_diff_hi=np.percentile(diffs, 97.5),
                        slope_delta_half2=bd, slope_delta_half2_lo=np.percentile(dhs, 2.5),
                        slope_delta_half2_hi=np.percentile(dhs, 97.5),
                        half_min=int(min(g.half1.min(), g.half2.min())), half_median=float(pd.concat([g.half1, g.half2]).median())))
    return pd.DataFrame(out)


# ================================================================ H2
def h2(r, rating_inc, test_images, min_items=40):
    v = pd.DataFrame(dict(code=r.code.values, item=r.item.values, image=r.image_name.values, v=rating_inc))
    s = v.groupby("item").v.transform("sum")
    n = v.groupby("item").v.transform("size")
    v["dm"] = v.v - (s - v.v) / (n - 1)
    perm = np.random.default_rng(SEED_B).permutation(len(test_images))
    half_a = set(np.array(sorted(test_images))[perm[: len(test_images) // 2]])
    v["half"] = np.where(v.image.isin(half_a), "A", "B")
    n_items = v.groupby("code").size()
    keep = n_items[n_items >= min_items].index
    pp = v[v.code.isin(keep)].pivot_table(index="code", columns="half", values="dm", aggfunc="mean")
    ni = v[v.code.isin(keep)].pivot_table(index="code", columns="half", values="dm", aggfunc="size")
    pp = pp.dropna()
    x, yv = pp.A.values, pp.B.values
    draws = np.random.default_rng(SEED_B).integers(0, len(pp), (N_BOOT, len(pp)))
    bp = np.array([pearsonr(x[d], yv[d])[0] for d in draws])
    bs = np.array([spearmanr(x[d], yv[d])[0] for d in draws])
    return dict(persons=int(len(pp)), persons_total=int(v.code.nunique()), excluded_lt_min=int((n_items < min_items).sum()),
                items_per_half_min=int(ni.min().min()), items_per_half_median=float(np.median(ni.values)),
                pearson=float(pearsonr(x, yv)[0]), pearson_lo=float(np.nanpercentile(bp, 2.5)), pearson_hi=float(np.nanpercentile(bp, 97.5)),
                spearman=float(spearmanr(x, yv)[0]), spearman_lo=float(np.nanpercentile(bs, 2.5)), spearman_hi=float(np.nanpercentile(bs, 97.5)))


# ================================================================ main
def main():
    t0 = time.time()
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    R = {}

    # ---------------- ImageNet-16H data
    h = pd.read_csv(DATA / "h16_deid.csv").sort_values(["code", "task_order"])
    h = h[~h.duplicated(["code", "image_name", "noise_level"], keep="first")]
    items = h[["image_name", "noise_level", "image_category"]].drop_duplicates(["image_name", "noise_level"])
    items = items.sort_values(["image_name", "noise_level"]).reset_index(drop=True)
    key = pd.MultiIndex.from_frame(items[["image_name", "noise_level"]])
    truth = items.image_category.map(CLS).values
    P = {}
    for lv in LEVELS:
        dd = pd.read_csv(RAW / "2ntrf" / f"hai_{lv}_model_preds_max_normalized.csv")
        dd = dd[(dd.noise_type == "phase") & dd.noise_level.isin(NOISE)]
        for a in ARCH:
            g = dd[dd.model_name == a].set_index(["image_name", "noise_level"]).reindex(key)
            assert g[C16].notna().all().all() and (g.category.map(CLS).values == truth).all()
            P[f"{a}_{lv}"] = clipren(g[C16].values.astype(float))
    VARIANTS = [f"{a}_{lv}" for a in ARCH for lv in LEVELS]

    images = np.array(sorted(items.image_name.unique()))
    sel = images[np.sort(np.random.default_rng(SEED_A).choice(1200, 300, replace=False))]
    test = np.setdiff1d(images, sel)
    fm = fold_map(test)
    sel_mask = items.image_name.isin(sel).values
    test_mask = ~sel_mask

    # selection table (log loss and Brier)
    st = pd.DataFrame([dict(variant=v, selection_log_loss=score(P[v][sel_mask], truth[sel_mask]).mean(),
                            selection_brier=score(P[v][sel_mask], truth[sel_mask], "brier").mean(),
                            selection_accuracy=(P[v][sel_mask].argmax(1) == truth[sel_mask]).mean()) for v in VARIANTS])
    st = st.sort_values("selection_log_loss").reset_index(drop=True)
    st.to_csv(TAB / "selection_set.csv", index=False)
    base = st.variant[0]
    Q = dict(zip(st.variant, st.selection_log_loss))
    R["baseline_a"], R["runner_up"] = base, st.variant[1]
    R["brier_selection_best"] = st.sort_values("selection_brier").variant.iloc[0]

    ti = items[test_mask].copy()
    tidx = ti.index.values
    n_it = len(tidx)
    loc = {g: i for i, g in enumerate(tidx)}
    y = truth[tidx]
    fold_it = ti.image_name.map(fm).values
    noise_it = ti.noise_level.values
    r = h[h.image_name.isin(test)].copy()
    r["item"] = pd.MultiIndex.from_frame(r[["image_name", "noise_level"]]).map(
        dict(zip(zip(ti.image_name, ti.noise_level), range(n_it))))
    r = r.assign(lab=r.participant_classification.map(CLS).values, true=r.image_category.map(CLS).values,
                 conf_i=r.confidence.map({c: i for i, c in enumerate(CONF)}).values,
                 noise_i=r.noise_level.map({n: i for i, n in enumerate(NOISE)}).values,
                 fold=r.image_name.map(fm).values).reset_index(drop=True)
    ri, fold_r, noise_r = r.item.values.astype(int), r.fold.values, r.noise_level.values
    assert (r.true.values == y[ri]).all()
    R["counts"] = dict(test_images=len(test), test_items=n_it, test_ratings=len(r), persons=int(r.code.nunique()),
                       images_per_fold=pd.Series(list(fm.values())).value_counts().sort_index().tolist(),
                       humans_per_item=pd.Series(np.bincount(ri)).value_counts().sort_index().to_dict())
    logs = {v: np.log(P[v][tidx]) for v in VARIANTS}

    def ref_same_level(v, members=None):
        a, lv = v.split("_")
        members = members or [f"{b}_{lv}" for b in ARCH if b != a]
        return np.log(clipren(softmax(sum(logs[m] for m in members))))

    # ---------------- human vectors (structured, uniform_c, alpha1)
    HV = {}
    for rule in ["structured", "uniform_c", "alpha1"]:
        HV[rule] = h16_human_vectors(r, rule)
        print(f"human vectors {rule}: {time.time() - t0:.0f}s", flush=True)
    st_log = pd.DataFrame(HV["structured"][2])
    st_log_n = pd.DataFrame(HV["structured"][3])
    for df, name in [(st_log, "strengths_h16_outer.csv"), (st_log_n, "strengths_h16_nested.csv")]:
        df = df.copy()
        for ni, nz in enumerate(NOISE):
            df[f"strength_noise_{nz}"] = df.strengths.map(lambda s: s[ni])
        df.drop(columns="strengths").to_csv(TAB / name, index=False)
    R["h16_empty_prior_rows"] = int(st_log.empty_prior_rows.sum() + st_log_n.empty_prior_rows.sum())

    # ---------------- H1 main: all 20 variants
    runs, weights = {}, []
    ok_all = True
    for v in VARIANTS:
        res = h1_engine(logs[v], ref_same_level(v), HV["structured"][0], HV["structured"][1], ri, y,
                        fold_it, fold_r, noise_it, noise_r, label=f"main:{v}")
        runs[v] = res
        weights += res["weights"]
        ok_all &= res["ok"]
        print(f"H1 {v}: {time.time() - t0:.0f}s", flush=True)
    main = runs[base]
    img_it = ti.image_name.values
    R["H1"] = {c: cluster_boot(main["q"][c].values, img_it)[0] for c in QCOLS}
    R["H1_losses"] = main["losses"]
    qa = main["q"].assign(image_name=img_it, noise_level=noise_it, fold=fold_it)
    qa.to_csv(DATA / "full_item_values_baseline_a.csv", index=False)

    # ---------------- H1c
    curve = []
    boots = {}
    for v in VARIANTS:
        q = runs[v]["q"]
        b = cluster_boot(q[QCOLS].values, img_it)
        a, lv = v.split("_")
        curve.append(dict(variant=v, architecture=a, level=lv, Q=Q[v],
                          **{c: b[i]["mean"] for i, c in enumerate(QCOLS)},
                          **{f"{c}_lo": b[i]["ci_lo"] for i, c in enumerate(QCOLS)},
                          **{f"{c}_hi": b[i]["ci_hi"] for i, c in enumerate(QCOLS)}))
    curve = pd.DataFrame(curve)
    # image-cluster bootstrap of each variant's mean, shared draws, for slope intervals
    g_codes, g = np.unique(img_it, return_inverse=True)
    draws = np.random.default_rng(SEED_B).integers(0, len(g_codes), (N_BOOT, len(g_codes)))
    cnt = np.bincount(g, minlength=len(g_codes))
    den = cnt[draws].sum(1)
    boot_means = {c: np.stack([np.bincount(g, runs[v]["q"][c].values, len(g_codes))[draws].sum(1) / den
                               for v in curve.variant]) for c in QCOLS}   # (20, N_BOOT)
    x = curve.Q.values
    slopes = []
    for c in QCOLS:
        beta, lev, cook, flag = ols_flags(x, curve[c].values)
        curve[f"{c}_leverage"], curve[f"{c}_cook"], curve[f"{c}_flag"] = lev, cook, flag
        bm = boot_means[c]
        b_all = np.array([slope(x, bm[:, i]) for i in range(N_BOOT)])
        uf = ~flag
        b_uf = np.array([slope(x[uf], bm[uf, i]) for i in range(N_BOOT)])
        s_all, s_uf = slope(x, curve[c].values), slope(x[uf], curve[c].values[uf])
        slopes.append(dict(quantity=c, slope_all=s_all, all_lo=np.percentile(b_all, 2.5), all_hi=np.percentile(b_all, 97.5),
                           n_flagged=int(flag.sum()), flagged=", ".join(curve.variant[flag]),
                           slope_unflagged=s_uf, unflagged_lo=np.percentile(b_uf, 2.5), unflagged_hi=np.percentile(b_uf, 97.5),
                           sign_holds_unflagged=bool(np.sign(s_all) == np.sign(s_uf))))
    curve.to_csv(TAB / "h1c_curve.csv", index=False)
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(TAB / "h1c_slopes.csv", index=False)
    R["h1c_slopes"] = slopes.to_dict("records")
    h1c_figure(curve, slopes)

    # ---------------- H2
    R["H2"] = h2(r, main["rating_inc"], test)

    # ---------------- robustness on H1a/H1b/H2
    lra = ref_same_level(base)
    rob = {}
    for name, kw in [
        ("R1_uniform_c", dict(hv="uniform_c")),
        ("R2_brier", dict(rule="brier")),
        ("R3_ref_all19", dict(ref=[v for v in VARIANTS if v != base])),
        ("R4_weights_per_noise", dict(per_noise=True)),
        ("R7_alpha1", dict(hv="alpha1")),
    ]:
        hvr = HV[kw.get("hv", "structured")]
        lr_ = ref_same_level(base, kw["ref"]) if "ref" in kw else lra
        res = h1_engine(logs[base], lr_, hvr[0], hvr[1], ri, y, fold_it, fold_r, noise_it, noise_r,
                        rule=kw.get("rule", "log"), per_noise=kw.get("per_noise", False), label=name)
        weights += res["weights"]
        ok_all &= res["ok"]
        rob[name] = dict(H1={c: cluster_boot(res["q"][c].values, img_it)[0] for c in QCOLS},
                         H2=h2(r, res["rating_inc"], test) if name != "R3_ref_all19" else None)
        print(f"robustness {name}: {time.time() - t0:.0f}s", flush=True)

    # R6: 128 overlap items restricted to the test set
    tej_items = pd.concat([pd.read_csv(DATA / f"tejeda_{k}_deid.csv", usecols=["image_name", "noise_level"])
                           for k in ["concurrent", "sequential"]]).drop_duplicates()
    ov = set(map(tuple, tej_items[tej_items.noise_level.isin(NOISE)].values))
    ov_mask = np.array([(a, b) in ov for a, b in zip(img_it, noise_it)])
    ov_sel = sum((a, b) in ov for a, b in zip(items.image_name[sel_mask], items.noise_level[sel_mask]))
    ov_r = ov_mask[ri]
    npp = pd.Series(r.code.values[ov_r]).value_counts()
    rob["R6_overlap"] = dict(
        items_in_test=int(ov_mask.sum()), items_in_selection=int(ov_sel), images_in_test=int(len(set(img_it[ov_mask]))),
        H1={c: cluster_boot(main["q"][c].values[ov_mask], img_it[ov_mask])[0] for c in QCOLS},
        H2=dict(computable=False, max_overlap_items_per_person=int(npp.max()), persons_with_ge_40=int((npp >= 40).sum())))
    R["robustness_h1_h2"] = rob

    wt = pd.DataFrame(weights)
    wt.to_csv(TAB / "combiner_weights.csv", index=False)
    neg = wt[wt.weights.map(lambda w: min(w) < 0)]
    R["negative_weights"] = neg.to_dict("records")
    R["optimizer_all_converged"] = bool(ok_all)

    # ---------------- calibration (baseline a, structured)
    cal = pd.concat([
        calib(HV["structured"][0], r.true.values, noise_r, "human confusion row (individual)"),
        calib(P[base][tidx], y, noise_it, "raw baseline (a)"),
        calib(main["probs"]["F0"], y, noise_it, "F0: recalibrated (a)"),
        calib(main["probs_r"]["Fh"], r.true.values, noise_r, "F(a, h) individual"),
        calib(main["probs"]["Fhp"], y, noise_it, "F(a, h) pooled"),
        calib(main["probs"]["Fr"], y, noise_it, "F(a, r_a)"),
        calib(main["probs"]["Frhp"], y, noise_it, "F(a, r_a, h) pooled"),
        calib(HV["alpha1"][0], r.true.values, noise_r, "α = 1 uniform row (robustness 7)"),
    ])
    cal.to_csv(TAB / "calibration_bins.csv", index=False)
    panel_figure(cal, ["human confusion row (individual)", "α = 1 uniform row (robustness 7)"], "noise_level", NOISE,
                 "Calibration of cross-fitted human rows: structured prior vs α = 1 uniform", FIG / "calibration_human_rows.png")
    panel_figure(cal, ["raw baseline (a)", "F0: recalibrated (a)", "F(a, h) individual"], "noise_level", NOISE,
                 "Calibration: baseline (a), recalibrated F0, and F(a, h) individual", FIG / "calibration_baseline_individual.png")
    panel_figure(cal, ["F(a, h) pooled", "F(a, r_a)", "F(a, r_a, h) pooled"], "noise_level", NOISE,
                 "Calibration: pooled human and reference combiners", FIG / "calibration_pooled_reference.png")
    diag = []
    for nz in NOISE:
        for ci, cf in enumerate(CONF):
            m = (noise_r == nz) & (r.conf_i.values == ci)
            hv = HV["structured"][0][m]
            diag.append(dict(noise=nz, confidence=cf, ratings=int(m.sum()),
                             accuracy=float((r.lab.values[m] == r.true.values[m]).mean()),
                             diag_structured=float(hv[np.arange(m.sum()), r.lab.values[m]].mean()),
                             diag_alpha1=float(HV["alpha1"][0][m][np.arange(m.sum()), r.lab.values[m]].mean())))
    pd.DataFrame(diag).to_csv(TAB / "row_diagonal_vs_accuracy.csv", index=False)
    # descriptive diagnostic for the strength grid: human-alone log loss by rule and by fixed strength
    alone = [dict(rows=rule, strength="inner CV" if rule == "structured" else "—",
                  human_alone_log_loss=float(score(HV[rule][0], r.true.values).mean()))
             for rule in ["structured", "uniform_c", "alpha1"]]
    for s_fix in [16, 32, 64, 128]:
        tot = 0.0
        for k in range(K):
            rows_k, _ = rows_structured(counts(r[fold_r != k]), [s_fix] * 4)
            te = r[fold_r == k]
            tot += score(vecs(te, rows_k), te.true.values).sum()
        alone.append(dict(rows="structured prior, fixed strength (outside the locked grid if > 16)", strength=s_fix,
                          human_alone_log_loss=tot / len(r)))
    pd.DataFrame(alone).to_csv(TAB / "human_alone_losses.csv", index=False)
    R["human_alone_losses"] = alone
    R["diag_overall"] = dict(
        accuracy=float((r.lab.values == r.true.values).mean()),
        structured=float(HV["structured"][0][np.arange(len(r)), r.lab.values].mean()),
        alpha1=float(HV["alpha1"][0][np.arange(len(r)), r.lab.values].mean()))

    # display rows on all 900 test images (structured, inner-CV strengths)
    dlog = []
    rows_disp = fit_h16_rows(r, "structured", dlog)
    n_disp = counts(r).sum(-1)
    R["display_strengths"] = dlog[0]
    recs = []
    for ni, nz in enumerate(NOISE):
        for ci, cf in enumerate(CONF):
            for li, lab in enumerate(C16):
                recs.append(dict(noise_level=nz, confidence=cf, human_label=lab, n=int(n_disp[ni, li, ci]),
                                 **{f"P_true_{c}": float(rows_disp[ni, li, ci, j]) for j, c in enumerate(C16)}))
    pd.DataFrame(recs).to_csv(TAB / "confusion_rows_h16.csv", index=False)
    print(f"H1/H2 done: {time.time() - t0:.0f}s", flush=True)

    # ---------------- Tejeda H3
    T = {}
    tcal = []
    for design in ["sequential", "concurrent"]:
        d, M, excl = load_tejeda(design)
        dres = dict(exclusions=excl, participants=int(d.code.nunique()),
                    participants_by_level=d.groupby("model_performance_level").code.nunique().reindex(TLEVELS).tolist(),
                    trials=int(len(d)))
        if design == "sequential":
            pairs = d.groupby(["code", "image_name"]).model_on.agg(lambda s: tuple(sorted(s)))
            dres["pairs_complete"] = int((pairs == (0, 1)).sum())
            dres["pairs_other"] = int((pairs != (0, 1)).sum())
        vec, per_rule = {}, {}
        for rule in ["structured", "uniform_c", "alpha1"]:
            hv, lg = tejeda_vectors(d, rule)
            vec[rule] = hv
            if lg:
                pd.DataFrame(lg).to_csv(TAB / f"strengths_tejeda_{design}_{rule}.csv", index=False)
                per_rule[rule] = lg
        variants = [("main", "structured", "log"), ("R1_uniform_c", "uniform_c", "log"),
                    ("R2_brier", "structured", "brier"), ("R7_alpha1", "alpha1", "log")]
        stats = {}
        for name, rule, sr in variants:
            p = h3_participants(d, M, vec[rule], design, sr)
            if name == "main":
                p.to_csv(DATA / f"full_h3_participants_{design}.csv", index=False)
            stats[name] = h3_stats(p).to_dict("records")
            if design == "concurrent" and name == "main":
                pw = p[~p.code.str.startswith("W")]
                stats["main_without_h16_overlap"] = h3_stats(pw).to_dict("records")
                dres["h16_overlap_kept"] = int(p.code.str.startswith("W").sum())
        dres["stats"] = stats
        dres["strength_logs"] = per_rule
        un = d.model_on.values == 0
        tcal.append(calib(vec["structured"][un], d.true.values[un], np.full(un.sum(), 0), f"{design}: unassisted rows").assign(design=design))
        tcal.append(calib(vec["structured"][~un], d.true.values[~un], np.full((~un).sum(), 0), f"{design}: assisted rows").assign(design=design))
        T[design] = dres
        # display rows on all images
        lg = []
        U, A = tejeda_rows(d, "structured", lg)
        dres["display_strengths"] = lg[0]
        nA = {lv: tcounts(d[(d.model_on == 1) & (d.lv == lv)]).sum(-1) for lv in range(3)}
        nU = tcounts(d[d.model_on == 0]).sum(-1)
        recs = []
        for lv in [None, 0, 1, 2]:
            rows, nn = (U, nU) if lv is None else (A[lv], nA[lv])
            for ci, cf in enumerate(CONF):
                for li, lab in enumerate(C16):
                    recs.append(dict(design=design, condition="unassisted" if lv is None else "assisted",
                                     ai_level="all" if lv is None else TLEVELS[lv], confidence=cf, human_label=lab,
                                     n=int(nn[li, ci]), **{f"P_true_{c}": float(rows[li, ci, j]) for j, c in enumerate(C16)}))
        pd.DataFrame(recs).to_csv(TAB / f"confusion_rows_tejeda_{design}.csv", index=False)
        print(f"H3 {design}: {time.time() - t0:.0f}s", flush=True)
    R["H3"] = T
    tcal = pd.concat(tcal)
    tcal["panel"] = tcal.design
    tcal.to_csv(TAB / "calibration_bins_tejeda.csv", index=False)
    tcal2 = tcal.copy()
    tcal2["forecast"] = tcal2.forecast.str.replace(r"^\w+: ", "", regex=True)
    panel_figure(tcal2, ["unassisted rows", "assisted rows"], "panel", ["sequential", "concurrent"],
                 "Calibration of cross-fitted Tejeda confusion rows (all 256 items)", FIG / "calibration_tejeda_rows.png")

    R["runtime_s"] = round(time.time() - t0, 1)
    (REP / "full_results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"done: {R['runtime_s']}s")


# ================================================================ calibration and figures
EDGES = np.array([0, .01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001])


def calib(prob, yy, noise, name):
    out = []
    onehot = np.eye(16)[yy]
    for nz in np.unique(noise):
        m = noise == nz
        p, o = prob[m].ravel(), onehot[m].ravel()
        b = np.digitize(p, EDGES) - 1
        for bi in range(len(EDGES) - 1):
            s = b == bi
            if s.sum():
                out.append(dict(forecast=name, noise_level=nz, bin_lo=EDGES[bi], bin_hi=min(EDGES[bi + 1], 1.0),
                                n=int(s.sum()), mean_pred=float(p[s].mean()), obs_freq=float(o[s].mean())))
    return pd.DataFrame(out)


INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def panel_figure(cal, forecasts, panel_col, panels, title, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 0.3, 3.9), dpi=160,
                             sharex=True, sharey=True, facecolor=SURF)
    for ax, pv in zip(np.atleast_1d(axes), panels):
        style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        for i, f in enumerate(forecasts):
            dd = cal[(cal.forecast == f) & (cal[panel_col] == pv) & (cal.n >= 20)]
            ax.plot(dd.mean_pred, dd.obs_freq, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                    mec=SURF, mew=1.2, label=f, zorder=3 + i)
        ax.set_title(f"noise {pv}" if panel_col == "noise_level" else str(pv), color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("observed frequency", color=INK2, fontsize=9)
    fig.supxlabel("forecast probability (all 16 class probabilities, binned; bins with n ≥ 20)", color=INK2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper right", ncol=len(forecasts),
               frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.995, 0.93 if len(panels) < 3 else 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.86 if len(panels) < 3 else 0.93))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def h1c_figure(curve, slopes):
    rows = [("I_h", "I(h | m)"), ("D", "D(h | m)"), ("I_hr", "I(h | m, r_m)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), dpi=160, sharex=True, facecolor=SURF)
    xs = np.linspace(curve.Q.min() - 0.05, curve.Q.max() + 0.05, 50)
    for ri_, version in enumerate(["ind", "pool"]):
        for ci_, (stem, lab) in enumerate(rows):
            c = f"{stem}_{version}"
            ax = axes[ri_, ci_]
            style(ax)
            ax.axhline(0, color=AXIS, lw=1, zorder=1)
            for i, a in enumerate(ARCH):
                g = curve[curve.architecture == a].set_index("level").reindex(LEVELS)
                ax.plot(g.Q, g[c], color=SERIES[i], lw=2, marker=MARKERS[i], ms=6, mec=SURF, mew=1.2,
                        label=a, zorder=3)
            fl = curve[curve[f"{c}_flag"]]
            ax.scatter(fl.Q, fl[c], s=170, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4,
                       label="flagged (leverage or Cook's D)")
            s = slopes.set_index("quantity").loc[c]
            X = curve.Q.values
            b_all = np.polyfit(X, curve[c].values, 1)
            uf = ~curve[f"{c}_flag"].values
            b_uf = np.polyfit(X[uf], curve[c].values[uf], 1)
            ax.plot(xs, np.polyval(b_all, xs), color=MUTED, lw=1.5, zorder=2, label="fit, all 20 points")
            ax.plot(xs, np.polyval(b_uf, xs), color=INK, lw=1.5, ls=(0, (5, 3)), zorder=2, label="fit, unflagged points")
            ax.set_title(f"{lab}, {'individual' if version == 'ind' else 'pooled'}", color=INK, fontsize=10, loc="left")
            ax.text(0.02, 0.97, f"slope all {s.slope_all:+.3f}\nslope unflagged {s.slope_unflagged:+.3f}",
                    transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("Q_m: selection-set log loss of baseline m", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("nats per item", color=INK2, fontsize=9)
    fig.suptitle("H1c: human increment against baseline quality, 20 variants (levels joined by architecture)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK2, bbox_to_anchor=(0.995, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG / "h1c_curve.png", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()
