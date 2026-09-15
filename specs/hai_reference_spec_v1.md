# SPEC v1: Reanalysis of ImageNet-16H and Tejeda et al. (2022): the Human Increment Against an Ensemble, and the Cost of Seeing the Model

Status: DRAFT 2026-09-15. Becomes LOCKED when §11 is confirmed and the file is pushed to a public repository before any analysis runs.

Reanalysis of the ImageNet-16H and Tejeda et al. (2022) OSF data is by permission of M. Steyvers (email, 15 September 2026).

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

Humans give a label and a three-level confidence, not a probability vector. **[LOCKED]** The human forecast is a 16-class vector with mass `c` on the chosen label and `(1 − c)/15` on each other class, where `c` is the empirical accuracy of human labels at that confidence level, estimated by 5-fold cross-fitting over items (seed 20260915) so that no item's own outcome enters its `c`. The same rule, with a single `c`, is the robustness variant.

Loss: 16-class log loss with probabilities clipped at 0.001. Brier (multiclass) is the robustness scoring rule.

## 4. Baselines and reference

- **[LOCKED]** Baseline (a): the classifier variant with the lowest log loss on a **selection set** of 1,200 items drawn at random from the 4,800 (seed 20260915) and excluded from every test; the remaining 3,600 are the test set. Reported: the chosen variant, its selection-set loss, and the runner-up.
- **[LOCKED]** For the curve (§7 H1c), every one of the 20 variants serves in turn as the baseline; `Q_m` = its selection-set log loss.
- **[LOCKED]** Reference `r_m` for variant m: the log-linear pool (normalized product) of the other four architectures at the **same fine-tuning level**. This matches the forecasting study's matched-condition rule. A second reference, all other 19 variants pooled, is a robustness item.

## 5. Quantities

For item q and a human classification h on it:

- `I(h | m)` = out-of-sample reduction in log loss from adding h to baseline m, in a combiner cross-fitted over items (5 folds, seed 20260915): log-linear pool of m's vector and h's vector with one weight on h fitted in the training folds. Per item, the average over the 6 to 7 humans on that item. This is the scale-free measure of the forecasting study.
- `I(r_m | m)` = the same with the reference in place of the human.
- `D(h | m) = I(h | m) − I(r_m | m)`.
- `I(h | m, r_m)` = the increment of h given both m and r_m in the combiner, the "still need a person after the ensemble" quantity.
- Per person p: `I_p` = mean `I(h | m)` over the items p classified, against baseline (a).
- Tejeda: per participant, `IA_p` (unassisted trials) and `IB_p` (assisted trials), each the mean gain over the model that participant was assigned, `G = loss(m) − loss(h)`; `Δ_p = IB_p − IA_p`.

## 6. Pilot

**[LOCKED]** A random 30% of test items (seed 20260916), all humans on them. H1 and the §5 machinery only. Deliver §9 for the pilot; stop; full run on confirmation.

## 7. Hypotheses

- **H1a.** `I(h | a) > 0` against baseline (a). Item-level cluster bootstrap over test items, 2,000 draws, seed 20260916.
- **H1b.** `D(h | a)` and `I(h | a, r_a)`, same bootstrap. The reading is fixed in advance: the claim that humans carry information the classifiers lack is supported only where `I(h | a, r_a)` is bounded away from zero.
- **H1c.** The curve: `I(h | m)`, `D(h | m)` and `I(h | m, r_m)` against `Q_m` for all 20 variants, with the two scaffold-pairs of the forecasting study replaced by the four fine-tuning levels of each architecture joined by lines; the leverage rule of the forecasting SPEC v2 Amendment 1 applies (leverage > 2p/n or Cook's D > 4/n; unflagged-points refit is co-primary). Expected direction: `I` rises as `Q` worsens; whether `I(h | m, r_m)` falls to zero at the best level is the open question.
- **H2.** Per-person split-half of `I_p` (items split at random, seed 20260916, leave-one-out item demeaning as in forecasting SPEC v1 §5.5), Pearson and Spearman with bootstrap over persons. Only persons with at least 40 test items.
- **H3a.** Tejeda: mean `Δ_p` by assigned AI level (below, near, above human accuracy), bootstrap over participants. Two-sided. Expected direction recorded: positive for the above-human level, undetermined for the others.
- **H3b.** Mean `IB_p` by level, one-sided against zero: do people who see the model beat it?
- **H3c.** Slope of `Δ_p` on `IA_p` by level. **[LOCKED]** Because there is no no-AI control arm, `Δ_p` and `IA_p` share the unassisted trials and the naive slope carries regression to the mean. The unassisted trials are split at random per participant (seed 20260916): `IA_p` is estimated on one half and `Δ_p` uses the other half. If the implementer finds that `model_on` trials are blocked rather than interleaved, order effects are confounded with seeing the model and H3a is reported as descriptive only. Expected direction: negative, strongest at the below-human level where about half of participants have `IA_p > 0`.

**[LOCKED]** No other tests.

## 8. Robustness (four items only)

1. Single-`c` human vector (§3).
2. Multiclass Brier in place of log loss.
3. Reference = all other 19 variants pooled.
4. H1 and H2 restricted to the 160 Tejeda-overlap items, so the two datasets are compared on the same images.

## 9. Deliverables

1. Environment, snapshot hashes, the three confirmations of §2.
2. The de-identification step and a statement that no worker id or raw file is in the repository.
3. Baseline (a): selection-set table.
4. The cross-fitted confidence-to-probability map `c` and the fitted combiner weights, with calibration plots.
5. H1a, H1b tables; H1c figure with leverage marks and the unflagged refit.
6. H2 with intervals and the person count.
7. H3a to H3c tables by level, with the interleaving confirmation stated next to them.
8. The four robustness items.
9. A short statement of limits: 6 to 7 humans per item; labels not probabilities; MTurk population; one image domain; the Tejeda participants each saw one level.

## 10. Stop rule

Stop after the pilot. Stop after the full run. The designer decides whether this becomes a section of a methods note (with the 4-of-74 documentation finding) or a standalone short paper.

## 11. Parameters awaiting the designer's confirmation

| Parameter | Proposed | Location |
|---|---|---|
| Human vector rule | confidence-calibrated `c`, cross-fitted | §3 |
| Loss | 16-class log loss, clip 0.001 | §3 |
| Selection set | 1,200 of 4,800 items, seed 20260915 | §4 |
| Reference | other four architectures at the same level, log-linear pool | §4 |
| Combiner | log-linear pool with one fitted weight, 5 folds | §5 |
| Minimum items per person for H2 | 40 | §7 |
| Pilot | 30% of test items, seed 20260916 | §6 |
| Bootstrap draws | 2,000 | §7 |
| H3c split-sample rule | halves of unassisted trials per participant | §7 |
