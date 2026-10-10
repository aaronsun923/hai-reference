# hai-reference: SPEC v2 (reliance behavior on the Tejeda sequential data, across classifier levels)

Status: LOCKED 2026-10-09, committed before any computation.

Aaron commits; Claude Code writes the code and the report. SPEC v1 and its amendments are untouched; nothing here changes a v1 estimate.

## Purpose

Liu, Zhou, Shen, Liu, Wu and Chen (CHI 2026, arXiv 2602.11567) define behavioral indicators of overreliance from interaction logs (copy-paste counts, scrolling, window switches, idle time). The Tejeda et al. (2022) sequential data (answer, see the classifier, revise) has no interaction logs, so those indicators cannot be computed on it. What the sequential structure does support is the trial-level reliance measures of the reliance literature (Eckhardt et al. 2024 survey; Schemmer et al. 2023), split by whether the classifier was correct and by classifier level. Tejeda et al. report only the overall switch proportion per classifier (their Figure 7) and no split by classifier correctness. This spec computes the split, across the three classifier levels, with intervals. One analog of the CHI "hesitation before adopting" pattern is computed and labelled as an analog, not as their indicator.

## Step 0: literature check, before anything else

Search for any published or preprint analysis of the Tejeda 2022 sequential data (or Steyvers et al. 2022 ImageNet-16H) that reports switch rates split by classifier correctness, or RAIR/RSR, by classifier level. Check: papers citing Tejeda 2022 (Google Scholar or Semantic Scholar), Steyvers' publication list, Schemmer 2023 and the Eckhardt survey's reference list. If such an analysis exists, stop and report it; this spec is not run.

## Data

The Tejeda sequential rows already loaded by hai-reference for SPEC v1 H3 (same files, same inclusion rules, same participant set). Required fields per trial: participant, classifier level (A, B, C), image, true class, initial answer, initial confidence (low, medium, high), classifier's top class, final answer, final confidence, response times if present. Step 1 is to confirm which of these fields exist; if response time is absent, the RT analog is dropped and the report says so.

Frame check: the trial and participant counts per level must equal the H3 frame in full_REPORT before any quantity is reported.

## Quantities (per classifier level, and pooled)

Disagreement trials: initial answer differs from the classifier's top class. All measures below are on disagreement trials unless stated.

1. Switch rate: share of disagreement trials where the final answer equals the classifier's class. Reproduction check: must agree with Tejeda's Figure 7 pattern (rising from A to C); report the three values.
2. Switch rate when the classifier is correct, and when it is incorrect (two rates per level).
3. RAIR (Schemmer 2023): among trials where the initial answer is wrong and the classifier is right, the share where the final answer is the classifier's. RSR: among trials where the initial answer is right and the classifier is wrong, the share where the final answer stays the initial one.
4. Overreliance rate: among trials where the initial answer is right and the classifier is wrong, the share switching to the classifier (1 minus RSR). Underreliance rate: among trials where the initial answer is wrong and the classifier is right, the share not switching (1 minus RAIR).
5. Hesitation analog (labelled as such): switch rate on disagreement trials by initial confidence (low, medium, high), separately for classifier-correct and classifier-incorrect trials. If response time exists: the same by RT tercile within participant.
6. Final accuracy on disagreement trials: initial-answer accuracy, classifier accuracy, final-answer accuracy, per level (the increment context for the reliance numbers).

Intervals: participant-cluster bootstrap, 2,000 draws, seed 20261009, percentile 95%. Levels are compared by the paired difference of each rate between levels inside the bootstrap (A vs B, B vs C, A vs C).

## Readings, fixed before the run

- Expected: switch rate rises with classifier level; RSR falls with level (people keep their own correct answer less often as the classifier gets better); RAIR rises with level. The report states these as expectations and reports whatever appears.
- The paper's claim, if the intervals support it: the reliance measures move with classifier level within one task and one participant pool, so a reliance figure reported for a single AI is a point on a curve. If the level differences include zero, the report says the measures are flat across this range of classifier accuracy (56% to 65%).
- The hesitation analog is descriptive. No claim about the CHI indicators follows from it.
- Multiplicity: three levels, six rates, and the confidence breakdown are reported as a table and a figure, not as a list of tests.

## Output

- code/reliance_v2.py (reads the v1 Tejeda frame through the existing loader; no v1 file modified).
- reports/reliance_v2_REPORT.md: the Step 0 result; the frame check; one table per quantity with intervals; the level-difference table; the figure.
- reports/figures/reliance_by_level.png: switch rate, RAIR, RSR against classifier level, with intervals; a second panel for the confidence breakdown.
- reports/reliance_v2_results.json.

Reproduction checks: the H3 frame counts; the Tejeda Figure 7 pattern for the overall switch rate. If either fails, stop and report.
