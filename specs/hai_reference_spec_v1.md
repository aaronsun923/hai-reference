# SPEC v1: Reanalysis of ImageNet-16H and Tejeda et al. (2022): the Human Increment Against an Ensemble, and the Cost of Seeing the Model

Status: LOCKED 2026-09-15, §11 parameters confirmed by the designer. Governing from the commit that adds Amendment 1 (2026-09-16); see the amendment for the record of the earlier push.

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
- **[LOCKED]** For Tejeda's assisted trials, rows are estimated separately from the unassisted trials, on assisted trials only, because a person who copies the model and then reports high confidence is not the same instrument as the same person unassisted. Copying a below-human model and copying an above-human model are also different behaviors, so rows are estimated per (condition × AI level), pooled across noise levels because Tejeda has only 256 items. Fallback rule, fixed in advance: any row with fewer than 30 training observations falls back to the row pooled across AI levels for that condition; the report lists every row that fell back.
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

**[LOCKED]** A random 30% of test **images** (about 270 images, 1,080 items; seed 20260916), all humans on them. H1 and the §5 machinery only. Deliver §9 for the pilot; stop; full run on confirmation.

## 7. Hypotheses

- **H1a.** `I(h | a) > 0` against baseline (a), individual and pooled. Cluster bootstrap over test **images**, 2,000 draws, seed 20260916.
- **H1b.** `D(h | a)` and `I(h | a, r_a)`, same bootstrap. The reading is fixed in advance: the claim that humans carry information the classifiers lack is supported only where `I(h | a, r_a)` is bounded away from zero.
- **H1c.** The curve: `I(h | m)`, `D(h | m)` and `I(h | m, r_m)` against `Q_m` for all 20 variants, with the two scaffold-pairs of the forecasting study replaced by the four fine-tuning levels of each architecture joined by lines; the leverage rule of the forecasting SPEC v2 Amendment 1 applies (leverage > 2p/n or Cook's D > 4/n; unflagged-points refit is co-primary). Expected direction: `I` rises as `Q` worsens; whether `I(h | m, r_m)` falls to zero at the best level is the open question.
- **H2.** Per-person split-half of `I_p` (images split at random per person, seed 20260916, leave-one-out item demeaning as in forecasting SPEC v1 §5.5), Pearson and Spearman with bootstrap over persons. Only persons with at least 40 test items; with about 200 items per person this excludes almost no one.
- **H3a.** Tejeda: mean `Δ_p` by assigned AI level (below, near, above human accuracy), bootstrap over participants. Two-sided. Expected direction recorded: positive for the above-human level, undetermined for the others.
- **H3b.** Mean `IB_p` by level, one-sided against zero: do people who see the model beat it?
- **H3c.** Does a person's unassisted increment predict how much of it survives when the model is shown? **[LOCKED]** There is no no-AI control arm, so `Δ_p` and `IA_p` share the unassisted trials and a naive slope carries regression to the mean. The estimator: split each participant's unassisted trials at random into halves (seed 20260916), `IA1_p` and `IA2_p`. Fit two slopes by level: `β_seen` from `IB_p ~ IA1_p` and `β_unseen` from `IA2_p ~ IA1_p`. Regression to the mean is the same in both; `β_seen − β_unseen` is the effect of seeing the model on the persistence of the individual increment, with a bootstrap CI over participants. A slope difference below zero means people who had more to add keep less of it once they see the model. Expected direction: negative, strongest at the below-human level. The `Δ_p`-on-half version is a robustness item. Before the lock, the implementer reports trials per participant per condition; if the unassisted half has fewer than about 30 trials, H3c is demoted to descriptive.
- **The two Tejeda designs measure different things and both are reported.** The concurrent design (a person judges with the model's prediction on screen) measures assisted judgment; it is the counterpart of block B in the Prolific experiment and is the primary source for H3a to H3c when its `model_on` trials are interleaved. The sequential design (a person answers, sees the model, and may revise the same item) is a within-item comparison, so item effects cancel, but it measures revision anchored on the person's own first answer; it is reported as a co-primary within-item version of H3a to H3c. If the implementer finds that the concurrent `model_on` trials are blocked rather than interleaved, order effects are confounded with seeing the model, the concurrent H3a is reported as descriptive only, and the sequential design becomes primary.

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

## 11. Parameters confirmed by the designer (2026-09-15, all [LOCKED])

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

---

## Amendment 1 (2026-09-16, before any hypothesis was run)

**Record of the push error.** The file committed at `bcbd93b` was a superseded draft (single-`c` human vector, no confusion rows, no fallback rule, item-level splits, Status still DRAFT). The version above is the one confirmed on 2026-09-15 and is the governing text from this commit on. The feasibility audit (deliverables 1 and 2 of §9) was run against the draft; every fact it established carries over, and its open items are ruled on here. No hypothesis had been run when this amendment was committed.

**Rulings from the audit.**

1. **Tejeda's three models are not the `2ntrf` checkpoints** (no exact vector match; the below-human level has no counterpart). H3 uses the 16-class vectors stored on each Tejeda trial row. The H1c curve uses the 20 `2ntrf` variants only. Robustness item 6 compares humans on the same images against different classifiers and says so.
2. **Concurrent `model_on` trials are blocked** (4 × 48 on, 16 off). Under §7, the sequential design is primary for H3a, H3b and H3c; every concurrent H3 quantity is reported as descriptive. In the sequential design the two responses to an image are the within-item pair, so the H3c split is **by image pair**: an image's unassisted and assisted responses go to the same half.
3. **Confusion rows for Tejeda.** Unassisted rows are estimated on Tejeda's own unassisted trials (not on ImageNet-16H, which has a narrower noise range and gave no feedback), pooled across AI levels; assisted rows per (condition × level) as §3 states; the 30-observation fallback applies to both.
4. **Participant exclusions, [LOCKED]:** exclude any participant with duplicated task numbers and repeated images within a session (the two-tab pattern; 3 concurrent, 2 sequential) and any participant with fewer than 30 unassisted trials (C202). Report the counts. The 8 concurrent participants who also took part in ImageNet-16H are kept, and every concurrent quantity is also reported without them as a sensitivity, descriptive.
5. **ImageNet-16H duplicates:** the 6 rows where a person rated the same item twice keep the first rating.
6. **Clipping, [LOCKED]:** clip every probability at 0.001, then renormalize the vector to sum to 1. Applies to human vectors, classifier vectors and pooled vectors alike.
7. **Overlap set:** 128 items (noise 80 to 125), not 160; the 32 noise-0 items have no ImageNet-16H ratings. Robustness item 6 uses the 128.
8. **Intervals:** 95% throughout; H3b one-sided at 95%.
9. **Rules imported from the forecasting SPECs**, restated here so this repository is self-contained:
   - Leverage rule (forecasting SPEC v2 Amendment 1): a point is flagged if leverage > 2p/n or Cook's distance > 4/n, applied mechanically; the refit on unflagged points is co-primary and is reported next to the all-points fit; a directional claim stands only if the sign holds with flagged points removed.
   - Split-half demeaning (forecasting SPEC v1 §5.5): before averaging within person, subtract from each item's value the leave-one-out mean of that value across all other persons on the same item; halves are assigned by a fixed random split of images with one seed for everyone.

No parameter in §11 changes.

---

## Amendment 2 (2026-09-16, before the pilot was run)

Rulings on the four pilot-implementation items left open by Amendment 1, plus the scope of the pilot report. No hypothesis and no pilot had been run when this amendment was written.

1. **Pilot unit:** 270 of the 900 test images (1,080 items), drawn at random with seed 20260916.
2. **Fold scope:** the pilot is self-contained. The 5 folds by image (seed 20260915) run over the 270 pilot images only, and the pilot's confusion rows and combiner weights are fitted within them. The full run refits over all 900 test images. This keeps the pilot a check on the pipeline rather than a preview of 30% of the final numbers. Selection-set images never feed the confusion rows or the combiner, in either run.
3. **Pooled human vector:** the equal-weight log-linear pool of the item's human confusion rows, renormalized. `w_h` is then fitted on that pooled vector exactly as on a single human vector.
4. **`I(h | m, r_m)`:** a three-weight pool `(w_m, w_r, w_h)`, compared with a two-weight pool `(w_m, w_r)` as `F0`. Weights are unconstrained; any fitted weight that comes out negative is reported.
5. **Pilot scope:** the pilot delivers §9 items 3, 4 and 9, plus H1a and H1b on the pilot images. No H1c curve, no leverage marks, no Tejeda quantities, no robustness items. §9 item 4 is limited to the ImageNet-16H confusion rows and the combiner weights.

No parameter in §11 changes.
