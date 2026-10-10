"""SPEC v2 (LOCKED 2026-10-09, commit afd91a6): reliance behavior on the Tejeda sequential data.

Reads the SPEC v1 Tejeda sequential frame through pipeline/02_full_run.py:load_tejeda (same file,
same exclusions, same participants). No v1 file is modified.

Order: (1) frame check against reports/full_results.json and the H3 frame in full_REPORT (72
participants, 24 per level, 13,824 trial pairs); (2) reproduction check: overall switch rate on
disagreement trials rises from A to C (Tejeda 2022, Fig. 7 bottom row); the script stops after
writing the report if either check fails; (3) quantities 1-6 with participant-cluster bootstrap
intervals; (4) reports/reliance_v2_REPORT.md, reports/reliance_v2_results.json,
reports/figures/reliance_by_level.png.

Path: the spec names code/reliance_v2.py; this repository keeps its scripts in pipeline/.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_run", ROOT / "pipeline" / "02_full_run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SPEC_COMMIT = "afd91a6b8f6f74f6a30966412c416cb652a3b03d"
REP = ROOT / "reports"
FIG = REP / "figures"
SEED = 20261009
N_BOOT = 2000
LV = {"below human": "A", "at human": "B", "above human": "C"}
LEVELS = ["A", "B", "C"]
PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]
CONF = m.CONF
TERC = ["fast", "middle", "slow"]


# ================================================================ trial pairs
def trial_pairs(d):
    """One row per trial: initial (model_on 0) and final (model_on 1) rows, consecutive by task_number."""
    s = d.sort_values(["code", "task_number"]).reset_index(drop=True)
    a, b = s.iloc[0::2].reset_index(drop=True), s.iloc[1::2].reset_index(drop=True)
    ok = ((a.code == b.code) & (a.image_name == b.image_name) & (a.model_on == 0) & (b.model_on == 1)
          & (a.model_prediction == b.model_prediction)).all()
    if not ok:
        sys.exit("trial pairing failed: initial and final rows do not line up")
    probs = b[[f"model_pred_{c}" for c in m.C16]].values
    t = pd.DataFrame(dict(
        code=a.code, level=a.model_performance_level.map(LV), image=a.image_name, true=a.image_category,
        init=a.participant_classification, init_conf=a.confidence, ai=b.model_prediction,
        final=b.participant_classification, final_conf=b.confidence,
        rt_init=a.classification_time, rt_final=b.classification_time))
    t["ai_is_argmax"] = np.array(m.C16)[probs.argmax(1)] == t.ai.values
    t["dis"] = t.init != t.ai
    t["init_ok"] = t.init == t.true
    t["ai_ok"] = t.ai == t.true
    t["final_ok"] = t.final == t.true
    t["sw"] = t.final == t.ai
    t["stay"] = t.final == t.init
    # hesitation analog: final-stage decision time (AI shown -> final click), tercile within participant
    # over that participant's disagreement trials
    t["rt_terc"] = None
    for c, g in t[t.dis].groupby("code"):
        t.loc[g.index, "rt_terc"] = pd.qcut(g.rt_final.rank(method="first"), 3, labels=TERC).astype(str)
    return t


# ================================================================ quantities
def cells(t):
    """Every rate as (numerator mask, denominator mask). All rates pool trials across participants."""
    D = t.dis
    q = {
        "switch": (D & t.sw, D),
        "switch_ai_correct": (D & t.ai_ok & t.sw, D & t.ai_ok),
        "switch_ai_incorrect": (D & ~t.ai_ok & t.sw, D & ~t.ai_ok),
        "RAIR": (~t.init_ok & t.ai_ok & t.sw, ~t.init_ok & t.ai_ok),
        "RSR": (t.init_ok & ~t.ai_ok & t.stay, t.init_ok & ~t.ai_ok),
        "overreliance": (t.init_ok & ~t.ai_ok & t.sw, t.init_ok & ~t.ai_ok),
        "underreliance": (~t.init_ok & t.ai_ok & ~t.sw, ~t.init_ok & t.ai_ok),
        "third_class_after_correct_initial": (t.init_ok & ~t.ai_ok & ~t.sw & ~t.stay, t.init_ok & ~t.ai_ok),
        "dis_initial_acc": (D & t.init_ok, D),
        "dis_ai_acc": (D & t.ai_ok, D),
        "dis_final_acc": (D & t.final_ok, D),
        "disagreement_share": (D, pd.Series(True, index=t.index)),
    }
    for c in CONF:
        for k, a in (("ai_correct", t.ai_ok), ("ai_incorrect", ~t.ai_ok)):
            sel = D & a & (t.init_conf == c)
            q[f"conf_{c}_{k}"] = (sel & t.sw, sel)
    for r in TERC:
        for k, a in (("ai_correct", t.ai_ok), ("ai_incorrect", ~t.ai_ok)):
            sel = D & a & (t.rt_terc == r)
            q[f"rt_{r}_{k}"] = (sel & t.sw, sel)
    return q


def per_participant_counts(t):
    """Numerator and denominator per participant per rate: arrays [participants x rates]."""
    q = cells(t)
    names = list(q)
    codes = t.code.values
    pc = pd.Index(sorted(t.code.unique()))
    idx = pc.get_indexer(codes)
    num = np.zeros((len(pc), len(names)))
    den = np.zeros((len(pc), len(names)))
    for j, k in enumerate(names):
        nm, dm = q[k]
        np.add.at(num[:, j], idx[nm.values], 1)
        np.add.at(den[:, j], idx[dm.values], 1)
    lev = t.groupby("code")["level"].first().reindex(pc).values
    return names, pc, lev, num, den


def ratio(n, d):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / d, np.nan)


def bootstrap(names, lev, num, den):
    """Participant-cluster bootstrap, resampling within level (levels are between-participant).
    Pooled draws stack the three level resamples of the same draw, so level differences are paired."""
    rng = np.random.default_rng(SEED)
    rows = {L: np.where(lev == L)[0] for L in LEVELS}
    draws = {L: np.empty((N_BOOT, len(names))) for L in LEVELS + ["pooled"]}
    for b in range(N_BOOT):
        tot_n = np.zeros(len(names)); tot_d = np.zeros(len(names))
        for L in LEVELS:
            pick = rng.choice(rows[L], size=len(rows[L]), replace=True)
            n, d = num[pick].sum(0), den[pick].sum(0)
            draws[L][b] = ratio(n, d)
            tot_n += n; tot_d += d
        draws["pooled"][b] = ratio(tot_n, tot_d)
    return draws


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [None, None]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize(names, lev, num, den, draws):
    out = {}
    for L in LEVELS + ["pooled"]:
        sel = np.ones(len(lev), bool) if L == "pooled" else lev == L
        n, d = num[sel].sum(0), den[sel].sum(0)
        out[L] = {k: dict(rate=None if d[j] == 0 else float(n[j] / d[j]), num=int(n[j]), den=int(d[j]),
                          ci=ci(draws[L][:, j]), nan_draws=int(np.isnan(draws[L][:, j]).sum()))
                  for j, k in enumerate(names)}
    diffs = {}
    for a, b in PAIRS:
        pt = {k: (out[b][k]["rate"] - out[a][k]["rate"]) if None not in (out[a][k]["rate"], out[b][k]["rate"])
              else None for k in names}
        dd = draws[b] - draws[a]
        diffs[f"{b}-{a}"] = {k: dict(diff=pt[k], ci=ci(dd[:, j]), nan_draws=int(np.isnan(dd[:, j]).sum()))
                             for j, k in enumerate(names)}
    return out, diffs


# ================================================================ report helpers
def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def fs(x):
    return "n/a" if x is None else f"{x:+.3f}"


def cell(r):
    lo, hi = r["ci"]
    s = f"{f3(r['rate'])} [{f3(lo)}, {f3(hi)}]"
    if r["nan_draws"]:
        s += f" ({r['nan_draws']} empty draws)"
    return s


def dcell(r):
    lo, hi = r["ci"]
    flag = "" if lo is None or (lo <= 0 <= hi) else " *"
    return f"{fs(r['diff'])} [{fs(lo)}, {fs(hi)}]{flag}"


def table(out, keys, labels, cols=LEVELS + ["pooled"]):
    head = "| quantity | " + " | ".join(f"{c} (k / n)" if c != "pooled" else "pooled (k / n)" for c in cols) + " |"
    lines = [head, "|---|" + "---|" * len(cols)]
    for k, lab in zip(keys, labels):
        lines.append(f"| {lab} | " + " | ".join(f"{cell(out[c][k])} ({out[c][k]['num']} / {out[c][k]['den']})"
                                               for c in cols) + " |")
    return "\n".join(lines)


def figure(out, path):
    x = np.arange(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, lab, col, mk in (("switch", "switch rate (disagreement trials)", "#1f4e79", "o"),
                            ("RAIR", "RAIR", "#2e7d32", "s"), ("RSR", "RSR", "#c62828", "^")):
        y = np.array([out[L][k]["rate"] for L in LEVELS])
        lo = np.array([out[L][k]["ci"][0] for L in LEVELS]); hi = np.array([out[L][k]["ci"][1] for L in LEVELS])
        a1.errorbar(x, y, yerr=[y - lo, hi - y], marker=mk, color=col, capsize=4, label=lab)
    a1.set_xticks(x, ["A (below human)", "B (at human)", "C (above human)"])
    a1.set_ylim(0, 1); a1.set_ylabel("proportion"); a1.set_title("Reliance measures by classifier level")
    a1.legend(frameon=False, fontsize=8)
    cols = {"A": "#7b8fa6", "B": "#3b6a9a", "C": "#0b2f5b"}
    for j, L in enumerate(LEVELS):
        for k, ls, mk in (("ai_correct", "-", "o"), ("ai_incorrect", "--", "x")):
            y = np.array([out[L][f"conf_{c}_{k}"]["rate"] for c in CONF], float)
            lo = np.array([np.nan if out[L][f"conf_{c}_{k}"]["ci"][0] is None else out[L][f"conf_{c}_{k}"]["ci"][0]
                           for c in CONF]); hi = np.array([np.nan if out[L][f"conf_{c}_{k}"]["ci"][1] is None
                                                           else out[L][f"conf_{c}_{k}"]["ci"][1] for c in CONF])
            xs = x + (j - 1) * 0.08
            a2.errorbar(xs, y, yerr=[y - lo, hi - y], ls=ls, marker=mk, color=cols[L], capsize=3,
                        label=f"{L}, classifier {'correct' if k == 'ai_correct' else 'incorrect'}")
    a2.set_xticks(x, ["low", "medium", "high"]); a2.set_xlabel("initial confidence")
    a2.set_ylim(0, 1); a2.set_ylabel("switch rate"); a2.set_title("Hesitation analog: switch rate by initial confidence")
    a2.legend(frameon=False, fontsize=7, ncol=2)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def excl0(r):
    lo, hi = r["ci"]
    return lo is not None and not (lo <= 0 <= hi)


def readings(out, diffs, extra):
    """The spec's readings, fixed before the run, set against the results."""
    exp = (("switch", "rises", +1), ("RSR", "falls", -1), ("RAIR", "rises", +1))
    lines = ["", "## Readings fixed before the run, against the results", ""]
    for k, word, sgn in exp:
        vals = ", ".join(f"{L} {out[L][k]['rate']:.3f}" for L in LEVELS)
        parts = []
        for p in ("B-A", "C-B", "C-A"):
            r = diffs[p][k]
            agree = r["diff"] * sgn > 0
            parts.append(f"{p.replace('-', ' − ')} {fs(r['diff'])} ({'in the expected direction' if agree else 'against'}"
                         f"{', interval excludes zero' if excl0(r) else ', interval includes zero'})")
        lines.append(f"- **Expected: {k} {word} with level.** Observed {vals}. " + "; ".join(parts) + ".")
    # distinct rates only: switch_ai_correct is RAIR and underreliance is 1 - RAIR
    rel = ["switch", "switch_ai_incorrect", "RAIR", "RSR", "overreliance"]
    moved = {p: [k for k in rel if excl0(diffs[p][k])] for p in ("B-A", "C-B", "C-A")}
    acc = extra["ai_acc_all_trials"]
    lines += ["",
              "**The paper's claim, as the intervals allow it.** "
              f"Between A and B, {len(moved['B-A'])} of {len(rel)} distinct reliance rates (switch, switch with the classifier incorrect, RAIR, RSR, overreliance) move with an interval excluding zero; "
              f"between A and C, {len(moved['C-A'])} of {len(rel)}; between B and C, {len(moved['C-B'])} of {len(rel)}. "
              "Within one task and one participant pool, the reliance measures move with classifier level from A to "
              "the two better classifiers, so a reliance figure reported for a single AI is a point on a curve. "
              "Between B and C every level-difference interval includes zero: the measures are flat across that part "
              f"of the range (classifier accuracy {acc['B']:.0%} to {acc['C']:.0%} on these trials). "
              f"The full range is {acc['A']:.0%} to {acc['C']:.0%}, not the 56% to 65% in the spec text "
              "(see the reproduction check)." if not moved["C-B"] else
              "**The paper's claim, as the intervals allow it.** Level differences: " +
              "; ".join(f"{p}: {', '.join(v) or 'none'}" for p, v in moved.items()) + ".",
              "",
              "**Hesitation analog.** Descriptive only. No claim about the CHI 2026 interaction-log indicators "
              "follows from it.",
              "",
              "**Multiplicity.** Three levels, the rates above and the confidence and RT breakdowns are reported as "
              "tables and a figure, not as a list of tests."]
    return lines


STEP0 = """\
Searched 2026-10-09: all 54 works citing Tejeda et al. (2022) in OpenAlex (Semantic Scholar's API was rate-limited); \
OpenAlex full-text search for "ImageNet-16H"; Steyvers' publication list (2022 to 2026); the reference lists of \
Schemmer et al. (2023) and the Eckhardt et al. AI-reliance survey (ACM CSUR, doi 10.1145/3776528), cross-checked \
against the citing set; and the works citing the two HHAI papers from the same group.

**Result: no published or preprint analysis found that reports switch rates split by classifier correctness, or RAIR/RSR, \
by classifier level on the Tejeda sequential data or ImageNet-16H.** The closest items:

- Tejeda, Kumar & Steyvers (HHAI 2023, doi 10.3233/FAIA230087), "How Displaying AI Confidence Affects Reliance". Same \
group, same interface, sequential judge-advisor design, three classifiers; the confidence-displayed arm has 66 \
participants (23/23/20), against 75 recruited (25 per level) in Tejeda (2022), so it may overlap with these data but \
is not identical. Its Fig. 5 reports switch probability per classifier by noise level and by initial confidence, \
**not** split by classifier correctness, with no RAIR/RSR. Its confidence row is the unsplit counterpart of \
quantity 5 here.
- Tejeda, Kumar & Steyvers (HHAI 2022, doi 10.3233/FAIA220201): concurrent design; optimal-combination and \
automation-bias comparisons by confidence. No switch measures.
- Steyvers & Kumar (2024, Perspectives on Psychological Science): cites Tejeda (2022) descriptively; no reanalysis.
- Li & Steyvers (2025 arXiv 2507.22365; 2026 J. Math. Psych.): ImageNet-16H used for metacognitive sensitivity and \
combined accuracy; no switch or reliance rates.
- Not read in full: Boidot & Mourato (SSRN 7294165, August 2026), "What can the Judge-Advisor System tell about \
appropriate reliance…". The abstract describes a measurement framework and three models for categorical JAS \
tasks and names no dataset; the full text was behind a Cloudflare check and could not be retrieved. It cites \
Tejeda (2022). **Open item:** whether it illustrates its measures on the Tejeda data.
"""


def main():
    d, M, excl = m.load_tejeda("sequential")
    t = trial_pairs(d)
    full = json.load(open(REP / "full_results.json"))["H3"]["sequential"]
    per_level_p = t.groupby("level")["code"].nunique().to_dict()
    per_p_trials = t.groupby("code").size()
    frame = dict(participants=int(t.code.nunique()), per_level=per_level_p, trial_pairs=int(len(t)),
                 trials_rows=int(len(d)), trials_per_participant=sorted(set(per_p_trials.tolist())),
                 exclusions=excl,
                 expected=dict(participants=full["participants"], trials_rows=full["trials"],
                               trial_pairs=full["pairs_complete"], per_level=24))
    frame_ok = (frame["participants"] == full["participants"] and frame["trials_rows"] == full["trials"]
                and frame["trial_pairs"] == full["pairs_complete"] and all(v == 24 for v in per_level_p.values()))
    frame["ok"] = bool(frame_ok)
    extra = dict(ai_is_argmax=float(t.ai_is_argmax.mean()),
                 final_acc_all_trials={L: float(t[t.level == L].final_ok.mean()) for L in LEVELS},
                 ai_acc_all_trials={L: float(t[t.level == L].ai_ok.mean()) for L in LEVELS},
                 initial_acc_all_trials={L: float(t[t.level == L].init_ok.mean()) for L in LEVELS})

    names, pc, lev, num, den = per_participant_counts(t)
    point = {L: ratio(num[lev == L].sum(0), den[lev == L].sum(0)) for L in LEVELS}
    sw = [float(point[L][names.index("switch")]) for L in LEVELS]
    repro_ok = sw[0] < sw[1] < sw[2]
    repro = dict(switch=dict(zip(LEVELS, sw)), rising=bool(repro_ok))

    head_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = dict(spec_commit=SPEC_COMMIT, head_at_run=head_sha, seed=SEED, n_boot=N_BOOT, frame=frame,
               reproduction=repro, extra_checks=extra)
    L1 = [f"# SPEC v2: reliance behavior on the Tejeda sequential data, across classifier levels", "",
          f"SPEC v2, LOCKED 2026-10-09, spec commit `{SPEC_COMMIT}`. Repo HEAD at run: `{head_sha}`.", "",
          "- **Deviation from the spec wording.** The spec names `code/reliance_v2.py`. This repository keeps its "
          "scripts in `pipeline/`, so the script is `pipeline/reliance_v2.py`.",
          "- Levels: A = below human, B = at human, C = above human (between participants).", "",
          "## Step 0: literature check", "", STEP0, "## Frame check", "",
          f"- Participants: {frame['participants']} (H3 frame: {full['participants']}); per level "
          f"{', '.join(f'{k} {v}' for k, v in sorted(per_level_p.items()))} (H3 frame: 24 each).",
          f"- Trial rows: {frame['trials_rows']} (H3 frame: {full['trials']}); initial/final pairs: "
          f"{frame['trial_pairs']} (H3 frame: {full['pairs_complete']}); pairs per participant: "
          f"{frame['trials_per_participant']}.",
          f"- Exclusions (v1 Amendment 1 item 4, via `load_tejeda`): {', '.join(excl['two_tab'])} (two-tab).",
          f"- **Frame check: {'passed' if frame_ok else 'FAILED'}.**", "",
          "## Reproduction check (Tejeda 2022, Fig. 7)", "",
          "Fig. 7 (bottom row) of Tejeda et al. (2022) plots switch proportions on disagreement trials by human "
          "confidence and classifier-confidence bin, without printed values, and the text reports that advice is "
          "more likely to be taken from more accurate classifiers. The check is therefore the pattern: overall "
          "switch rate on disagreement trials rising from A to C.", "",
          f"- Switch rate: A {sw[0]:.3f}, B {sw[1]:.3f}, C {sw[2]:.3f}. **{'Rising A < B < C: passed' if repro_ok else 'Not rising: FAILED'}.**",
          f"- Extra check (not in the spec): final accuracy on all trials A {extra['final_acc_all_trials']['A']:.3f}, "
          f"B {extra['final_acc_all_trials']['B']:.3f}, C {extra['final_acc_all_trials']['C']:.3f}; Tejeda (2022) "
          "report 56%, 61%, 65% for the sequential paradigm (before their exclusions, 75 participants).",
          f"- Classifier accuracy on all trials: A {extra['ai_acc_all_trials']['A']:.3f}, B {extra['ai_acc_all_trials']['B']:.3f}, "
          f"C {extra['ai_acc_all_trials']['C']:.3f}. The spec's \"56% to 65%\" is Tejeda's human-with-AI accuracy, "
          "not classifier accuracy; the classifier range on these trials is the one above.",
          f"- Classifier class = argmax of the stored probability vector on {extra['ai_is_argmax']:.1%} of trials.", ""]

    if not (frame_ok and repro_ok):
        L1 += ["## Stopped", "", "A reproduction check failed. Under the spec, no quantity is reported."]
        (REP / "reliance_v2_REPORT.md").write_text("\n".join(L1) + "\n")
        json.dump(res, open(REP / "reliance_v2_results.json", "w"), indent=1)
        sys.exit("reproduction check failed; report written, quantities not computed")

    draws = bootstrap(names, lev, num, den)
    out, diffs = summarize(names, lev, num, den, draws)
    res.update(rates=out, level_differences=diffs)
    FIG.mkdir(exist_ok=True)
    figure(out, FIG / "reliance_by_level.png")

    q2 = ["switch_ai_correct", "switch_ai_incorrect"]
    L1 += [
        "## Readings made at run time", "",
        "- **Rates pool trials.** Each rate is (trials meeting the condition) / (trials in the denominator), "
        "pooled over the participants in the level; participants with more disagreement trials weigh more. "
        "This is the proportion Tejeda plot.",
        "- **Bootstrap.** Participants resampled with replacement within level (levels are between participants), "
        "2,000 draws, seed 20261009, percentile 95%. Pooled draws stack the three level resamples of the same draw. "
        "Level differences are taken draw by draw on these resamples (the spec's paired difference; with "
        "different people at each level the pairing is by draw, not by person). A draw whose denominator is "
        "empty is dropped from that interval and counted.",
        "- **Overreliance.** Computed as defined, the share of initial-right / classifier-wrong trials whose final "
        "answer is the classifier's. It equals 1 − RSR only when nobody moves to a third class; the third-class "
        "share is reported so the gap is visible.",
        "- **RT analog.** Response time exists. The hesitation analog uses the final-stage classification time "
        "(classifier shown to final click, `classification_time` on the final row), split into terciles within "
        "participant over that participant's disagreement trials (ties broken by order).",
        "- RAIR and RSR trials are disagreement trials by construction (one of initial and classifier is right, the other wrong).",
        "- **Quantity 2 (classifier correct) is RAIR.** On a disagreement trial where the classifier is right, the "
        "initial answer is wrong, so the two denominators are the same trials and the rates are identical "
        "(underreliance is its complement). Quantity 2 (classifier incorrect) is not RSR: it also holds the trials "
        "where both answers are wrong and differ.",
        "- `*` in the level-difference table marks an interval that excludes zero. No test is run.", "",
        "## 1. Switch rate on disagreement trials", "",
        table(out, ["disagreement_share", "switch"], ["share of trials that are disagreement trials", "switch rate"]), "",
        "## 2. Switch rate by classifier correctness", "",
        table(out, q2, ["switch, classifier correct", "switch, classifier incorrect"]), "",
        "## 3. RAIR and RSR", "",
        table(out, ["RAIR", "RSR"], ["RAIR (initial wrong, classifier right: final = classifier)",
                                     "RSR (initial right, classifier wrong: final = initial)"]), "",
        "## 4. Over- and underreliance", "",
        table(out, ["overreliance", "underreliance", "third_class_after_correct_initial"],
              ["overreliance (initial right, classifier wrong: final = classifier)",
               "underreliance (initial wrong, classifier right: final ≠ classifier)",
               "third class (initial right, classifier wrong: final is neither)"]), "",
        "## 5. Hesitation analog (labelled as an analog, not the CHI 2026 indicator)", "",
        "Switch rate on disagreement trials by initial confidence, split by classifier correctness.", "",
        table(out, [f"conf_{c}_{k}" for k in ("ai_correct", "ai_incorrect") for c in CONF],
              [f"{c} confidence, classifier {'correct' if k == 'ai_correct' else 'incorrect'}"
               for k in ("ai_correct", "ai_incorrect") for c in CONF]), "",
        "Switch rate on disagreement trials by final-stage response-time tercile within participant.", "",
        table(out, [f"rt_{r}_{k}" for k in ("ai_correct", "ai_incorrect") for r in TERC],
              [f"{r} tercile, classifier {'correct' if k == 'ai_correct' else 'incorrect'}"
               for k in ("ai_correct", "ai_incorrect") for r in TERC]), "",
        "## 6. Accuracy on disagreement trials", "",
        table(out, ["dis_initial_acc", "dis_ai_acc", "dis_final_acc"],
              ["initial-answer accuracy", "classifier accuracy", "final-answer accuracy"]), "",
        "## Level differences (later level minus earlier level)", "",
        "| quantity | B − A | C − B | C − A |", "|---|---|---|---|"]
    keys = ["switch", "switch_ai_correct", "switch_ai_incorrect", "RAIR", "RSR", "overreliance", "underreliance",
            "dis_initial_acc", "dis_ai_acc", "dis_final_acc"]
    for k in keys:
        L1.append(f"| {k} | " + " | ".join(dcell(diffs[p][k]) for p in ("B-A", "C-B", "C-A")) + " |")
    L1 += readings(out, diffs, extra)
    L1 += ["", "## Figure", "", "![Reliance by classifier level](figures/reliance_by_level.png)", "",
           "*Left:* switch rate (disagreement trials), RAIR and RSR by classifier level, 95% participant-cluster "
           "bootstrap intervals. *Right:* switch rate by initial confidence; solid = classifier correct, dashed = "
           "classifier incorrect; shade = level.", ""]
    (REP / "reliance_v2_REPORT.md").write_text("\n".join(L1) + "\n")
    json.dump(res, open(REP / "reliance_v2_results.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
