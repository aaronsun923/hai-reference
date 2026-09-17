# SPEC v1: Reanalysis of ImageNet-16H and Tejeda et al. (2022): the Human Increment Against an Ensemble, and the Cost of Seeing the Model

Status: DRAFT 2026-09-15. Becomes LOCKED when §11 is confirmed and the file is pushed to a public repository before any analysis runs.

Reanalysis of the ImageNet-16H and Tejeda et al. (2022) OSF data is by permission of M. Steyvers (email, 15 September 2026). The public repository carries a `PERMISSIONS.md` stating the date, the scope (reanalysis, results public, data not redistributed) and the conditions the author adds; the email itself stays local.

## 0. Note to the implementer

Rules of the earlier SPECs apply: [LOCKED] is a design decision, disagreement is reported, pilot before full run, no subgroup fishing. Work in `~/Desktop/hai-reference/`. The public behavioral files contain MTurk worker ids; the first step of the pipeline replaces them with random codes and the raw files are never copied into the repository or redistributed in any form.

## 1. Objective

The forecasting study (SPEC v1, v2) left three things open. Whether the human increment survives an ensemble reference in a second domain. Whether the shape of the increment against baseline quality holds where baseline quality is varied by design rather than by vendor. And what happens to a person's increment when they see the model. This dataset answers all three without new data collection.

- ImageNet-16H (OSF `2ntrf`): 1,200 ImageNet images at 4 phase-noise levels, 4,800 items; 145 participants, 28,997 classifications, 6 to 7 per item, each with a 16-way label and a confidence rating; no AI shown. 20 classifier variants (5 architectures × 4 fine-tuning levels), each with a full 16-class probability vector on every item.
- Tejeda et al. (2022) (OSF `9xbpu`): 256 items; participants who classified first without and then with an AI's prediction shown (concurrent and sequential designs), each participant assigned to one of three VGG-19 fine-tuning levels chosen to sit below, near, and above human accuracy; full 16-class probabilities for the three levels.

Three questions:

- H1. Against the best classifier, does the human label carry information the classifier does not? Does it survive an ensemble of the other classifiers? How does it change across the four fine-tuning levels?
- H2. Is the per-person increment stable across a split of items?
- H3. When a person sees the model's prediction, does their gain over that model change, does it depend on the model's level, and does it depend on the person's own increment before seeing it?

## 2. Data

- **[LOCKED]** Snapshot: the OSF files as listed in `INVENTORY.md` §2.2 and §1.3 row 71, with SHA-256 recorded at download. The normalized prediction files are used, as in the paper.
- **[LOCKED]** Items: for ImageNet-16H, the 4,800 (image, noise) items with human classifications. For Tejeda, the 256 items; the 160 that overlap ImageNet-16H's noise levels carry all 20 variants, the 96 at noise 140 to 170 carry only the three VGG levels, and every analysis states which item set it uses.
- Ground truth: the ILSVRC label.
- The implementer confirms from the files: the mapping of `epoch00` to the between-0-and-1-epoch level; whether the Tejeda VGG levels A/B/C correspond to `2ntrf` checkpoints; whether concurrent-design `model_on` trials are interleaved or blocked within a session. Each is reported before any hypothesis is run; the third determines whether §7 H3 needs a split-sample correction (§7).

## 3. Turning a label into a forecast

Humans give a label and a three-level confidence, not a probability vector. Human errors are structured (car and truck, cat and dog are confused with each other), so spreading the remaining mass evenly over the other 15 classes would hand the human a worse vector than the data support and depress every log-loss quantity below.

**[LOCKED]** The human forecast is the corresponding row of a cross-fitted confusion matrix: `P(true class | human label, confidence level)`, estimated with Dirichlet smoothing (α = 1 per cell) on the training folds of a 5-fold split **by image** (seed 20260915), so that no image's own outcomes enter its row. This is the human-side model of Kerrigan, Smyth and Steyvers (2021) and of the PNAS combination model.

- **[LOCKED]** Rows are estimated separately for each noise level. Human accuracy differs sharply across the four levels, and a pooled row misplaces the vector at every level.
- **[LOCKED]** For Tejeda's assisted trials, rows are estimated separately from the unassisted trials, on assisted trials only. A person who copies the model and then reports high confidence is not the same instrument as the same person unassisted.
- Robustness: the uniform off-diagonal rule (mass `c` on the label, `(1 − c)/15` elsewhere, `c` cross-fitted per confidence level and noise level), which is the special case of the confusion row with a uniform off-diagonal.

Loss: 16-class log loss with probabilities clipped at 0.001. Multiclass Brier is the robustness scoring rule.

## 4. Baselines and reference

- **[LOCKED]** Every split, fold and bootstrap in this SPEC is by **image**, not by item. The 4,800 items are 1,200 images × 4 noise levels; the four items of one image share content, and a classifier's performance on the same image at adjacent noise levels is highly correlated. Splitting by item would leak and narrow every interval.
- **[LOCKED]** Baseline (a): the classifier variant with the lowest log loss on a **selection set** of 300 images (1,200 items) drawn at random (seed 20260915) and excluded from every test; the remaining 900 images (3,600 items) are the test set. Reported: the chosen variant, its selection-set loss, and the runner-up.
- **[LOCKED]** For the curve (§7 H1c), every one of the 20 variants serves in turn as the baseline; `Q_m` = its selection-set log loss.
- **[LOCKED]** Reference `r_m` for variant m: the log-linear pool (normalized product) of the other four architectures at the **same fine-tuning level**. This matches the forecasting study's matched-condition rule. A second reference, all other 19 variants pooled, is a robustness item.

## 5. Quantities

For item q and a human classification h on it:

- **[LOCKED]** Combiner: a log-linear pool of m's vector and h's vector with **two** fitted weights `(w_m, w_h)`, normalized, fitted on the training folds of the 5-fold split by image (seed 20260915). The no-human comparison `F0` is the same pool with `w_h = 0` and `w_m` fitted, so that the increment is measured against a recalibrated baseline, not the raw one (the correction made in the problem statement §2). A single global pair of weights is primary; weights fitted per noise level are a robustness item, because a global pair understates the human where the human is strong (low noise) and overstates where weak.
- `I(h | m)` = out-of-sample reduction in log loss from adding h, relative to `F0`. Two versions, both reported: the **individual** version, averaged over the 6 to 7 humans on each item, and the **pooled** version, in which h is the log-linear pool of the confusion rows of all humans on the item. The pooled version is the counterpart of the forecasting study's human median and is the one compared with it.
- `I(r_m | m)` = the same with the reference in place of the human.
- `D(h | m) = I(h | m) − I(r_m | m)`.
- `I(h | m, r_m)` = the increment of h given both m and r_m in the combiner, the "still need a person after the ensemble" quantity.
- Per person p: `I_p` = mean `I(h | m)` over the items p classified, against baseline (a).
- Tejeda: per participant, `IA_p` (unassisted trials) and `IB_p` (assisted trials), each the mean gain over the model that participant was assigned, `G = loss(m) − loss(h)`; `Δ_p = IB_p − IA_p`.

## 6. Pilot

**[LOCKED]** A random 30% of test items (seed 20260916), all humans on them. H1 and the §5 machinery only. Deliver §9 for the pilot; stop; full run on confirmation.

## 7. Hypotheses

- **H1a.** `I(h | a) > 0` against baseline (a), individual and pooled. Cluster bootstrap over test **images**, 2,000 draws, seed 20260916.
- **H1b.** `D(h | a)` and `I(h | a, r_a)`, same bootstrap. The reading is fixed in advance: the claim that humans carry information the classifiers lack is supported only where `I(h | a, r_a)` is bounded away from zero.
- **H1c.** The curve: `I(h | m)`, `D(h | m)` and `I(h | m, r_m)` against `Q_m` for all 20 variants, with the two scaffold-pairs of the forecasting study replaced by the four fine-tuning levels of each architecture joined by lines; the leverage rule of the forecasting SPEC v2 Amendment 1 applies (leverage > 2p/n or Cook's D > 4/n; unflagged-points refit is co-primary). Expected direction: `I` rises as `Q` worsens; whether `I(h | m, r_m)` falls to zero at the best level is the open question.
- **H2.** Per-person split-half of `I_p` (images split at random per person, seed 20260916, leave-one-out item demeaning as in forecasting SPEC v1 §5.5), Pearson and Spearman with bootstrap over persons. Only persons with at least 40 test items; with about 200 items per person this excludes almost no one.
- **H3a.** Tejeda: mean `Δ_p` by assigned AI level (below, near, above human accuracy), bootstrap over participants. Two-sided. Expected direction recorded: positive for the above-human level, undetermined for the others.
- **H3b.** Mean `IB_p` by level, one-sided against zero: do people who see the model beat it?
- **H3c.** Does a person's unassisted increment predict how much of it survives when the model is shown? **[LOCKED]** There is no no-AI control arm, so `Δ_p` and `IA_p` share the unassisted trials and a naive slope carries regression to the mean. The estimator: split each participant's unassisted trials at random into halves (seed 20260916), `IA1_p` and `IA2_p`. Fit two slopes by level: `β_seen` from `IB_p ~ IA1_p` and `β_unseen` from `IA2_p ~ IA1_p`. Regression to the mean is the same in both; `β_seen − β_unseen` is the effect of seeing the model on the persistence of the individual increment, with a bootstrap CI over participants. A slope difference below zero means people who had more to add keep less of it once they see the model. Expected direction: negative, strongest at the below-human level. The `Δ_p`-on-half version is a robustness item. Before the lock, the implementer reports trials per participant per condition; if the unassisted half has fewer than about 30 trials, H3c is demoted to descriptive.
- If the implementer finds that `model_on` trials are blocked rather than interleaved in the concurrent design, order effects are confounded with seeing the model and H3a is reported as descriptive only; the sequential design is then the primary source for H3.

**[LOCKED]** No other tests.

## 8. Robustness (six items only)

1. Uniform off-diagonal human vector (§3).
2. Multiclass Brier in place of log loss.
3. Reference = all other 19 variants pooled.
4. Combiner weights fitted per noise level (§5).
5. H3c with `Δ_p` on the held-out half in place of the two-slope estimator.
6. H1 and H2 restricted to the 160 Tejeda-overlap items, so the two datasets are compared on the same images.

## 9. Deliverables

1. Environment, snapshot hashes, the three confirmations of §2.
2. The de-identification step and a statement that no worker id or raw file is in the repository.
3. Baseline (a): selection-set table.
4. The cross-fitted confusion rows (one table per noise level and confidence level, and the assisted-trial rows for Tejeda) and the fitted combiner weights, with calibration plots.
5. H1a, H1b tables; H1c figure with leverage marks and the unflagged refit.
6. H2 with intervals and the person count.
7. H3a to H3c tables by level, with the interleaving confirmation stated next to them.
8. The six robustness items.
9. A short statement of limits: 6 to 7 humans per item; labels not probabilities; MTurk population; one image domain; the Tejeda participants each saw one level.

## 10. Stop rule

Stop after the pilot. Stop after the full run. The designer decides whether this becomes a section of a methods note (with the 4-of-74 documentation finding) or a standalone short paper.

## 11. Parameters awaiting the designer's confirmation

| Parameter | Proposed | Location |
|---|---|---|
| Human vector rule | cross-fitted confusion row per noise and confidence level, Dirichlet α = 1 | §3 |
| Loss | 16-class log loss, clip 0.001 | §3 |
| Selection set | 300 of 1,200 images, seed 20260915; all splits by image | §4 |
| Reference | other four architectures at the same level, log-linear pool | §4 |
| Combiner | log-linear pool, two fitted weights, F0 = recalibrated baseline, 5 folds by image | §5 |
| Minimum items per person for H2 | 40 | §7 |
| Pilot | 30% of test items, seed 20260916 | §6 |
| Bootstrap draws | 2,000 | §7 |
| H3c estimator | two-slope (β_seen − β_unseen) on halves of unassisted trials; demote if < 30 trials per half | §7 |
