# Deliverables 1–2: snapshot, confirmations, de-identification

SPEC v1 (`specs/hai_reference_spec_v1.md`, governing version `093d123`; `bcbd93b` was a superseded draft). Audit run 2026-09-16 (Pacific). No hypothesis has been run.

## 1. Environment

macOS 26.6.2; Python 3.9.6; pandas 2.3.3; numpy 2.0.2.

## 1. Snapshot

Downloaded 2026-09-17T06:02:08Z (UTC) to `~/Desktop/hai-reference-raw/`, outside the repository. Each local SHA-256 equals the hash OSF reports for the file. Manifest: `~/Desktop/hai-reference-raw/MANIFEST.tsv`; `SNAPSHOT.json` has the same data with timestamps. Model checkpoints, images and unnormalized prediction files were not downloaded.

| OSF | Path | Download | Bytes | SHA-256 |
|---|---|---|---:|---|
| 2ntrf | Behavioral Data/Preprocessed/human_only_classification_6per_img_preprocessed.csv | osf.io/download/75u26 | 14,348,661 | `391476d2354ff1fd06b89679f917d01a0cbb43c95689cb1c5e238d0dd2abed12` |
| 2ntrf | Behavioral Data/human_only_classification_6per_img_export.csv (row-count check only) | osf.io/download/k2syr | 13,985,546 | `cb8321e05a99b1df8956ce3a7c2c88074898a23e028d928cfa7d118a7f560ec8` |
| 2ntrf | Machine Classifier Predictions/hai_baseline_model_preds_max_normalized.csv | osf.io/download/up8dj | 14,009,900 | `4f197144e299fbe24efb43c68eb65f9f45b082bc651f78ab2b1cd8678f07ad2e` |
| 2ntrf | Machine Classifier Predictions/hai_epoch00_model_preds_max_normalized.csv | osf.io/download/9gqc6 | 39,846,150 | `4cf106c57fffc4ada2dc3d4205d685288a05e14d05ac7f215c4869ad045a5ec9` |
| 2ntrf | Machine Classifier Predictions/hai_epoch01_model_preds_max_normalized.csv | osf.io/download/6hb3y | 14,202,399 | `4753345afd4403f2038c4e05cf34b357720af7b39fceaa65a15d87efc37e4d7f` |
| 2ntrf | Machine Classifier Predictions/hai_epoch10_model_preds_max_normalized.csv | osf.io/download/dfhvs | 40,336,446 | `27a716f3a61c178178866d1370b0330d69e37428c37df88410e32fc0028713bc` |
| 9xbpu | Behavioral Data/data_concurrent_paradigm.csv | osf.io/download/4n7xw | 12,825,097 | `69b63b7242a485965f1afccfd6335dc37b32b3c46477143a75346bbe13026327` |
| 9xbpu | Behavioral Data/data_sequential_paradigm.csv | osf.io/download/k528n | 20,739,715 | `5da2f91a82347744518b8f182c2b3ea04e4d6cadd11a2715a9faa0f5c55c481f` |

`9xbpu` holds only these two CSVs, plus two JAGS text files and wiki images. It has no separate prediction files. The three VGG-19 probability vectors are stored on every trial row (`model_pred_<class>`).

Preprocessed vs raw export (2ntrf): both 28,997 rows, same `id` set. The preprocessed file adds `Unnamed: 0`, `image_category_int`, `participant_classification_int`, `confidence_int`. The 32 shared columns are identical on every row.

## 1. The three confirmations of §2

**(a) `epoch00` = the level between 0 and 1 epoch. Confirmed by documentation and data, not stated in the file names.**
- PNAS Methods (PMC8931210): models fine-tuned "for either 0 epochs (baseline), between 0 and 1 epochs, 1 epoch, and 10 epochs. The second level of fine tuning (0 to 1 epochs) was based on a checkpoint during training before 1 epoch was reached."
- Files are named `baseline`, `epoch00`, `epoch01`, `epoch10`, so `epoch00` is the only file left for that level.
- Data, noise 80–125, `correct` by `model_name`: for every architecture, accuracy runs baseline < epoch00 < epoch01, and log loss (clip 0.001) baseline > epoch00 > epoch01. Accuracy: alexnet .277/.503/.684/.707, densenet161 .478/.706/.772/.833, googlenet .441/.593/.745/.731, resnet152 .434/.625/.751/.731, vgg19 .348/.647/.795/.850. `epoch10` is not uniformly best: googlenet and resnet152 have higher log loss at epoch10 than epoch01.

**(b) Tejeda levels A/B/C = the `2ntrf` VGG-19 checkpoints. Not confirmed; the data say they are different models.**
- Tejeda et al. (2022), AI Predictions: A = "fine-tuning for less than one epoch (10% of batches of the first epoch)", B = one epoch, C = 10 epochs, "All models were trained with all levels of phase noise" (which include 140–170, absent from every `2ntrf` file).
- Files: `9xbpu` `model_performance_level` ∈ {below human, at human, above human} (`model_level_int` −1/0/1) = A/B/C. One vector per (level, image): 768 vectors, identical across trials and across the two files.
- Match on (`image_name`, `noise_level`) against `2ntrf` `model_name == vgg19`, `noise_type == phase`, 160 items per level. No level matches any file exactly (row share within 1e−6: at most 13%). Closest per level (median max-abs difference; argmax agreement):
  - above human vs epoch10: 0.008; 89%
  - at human vs epoch10: 0.024; 86% (epoch01: 0.043; 83%)
  - below human vs every file: ≥ 0.12; ≤ 70%
- Conclusion: A/B/C were retrained or re-evaluated models. H3 must use the per-trial `model_pred_*` vectors in `9xbpu`, never the `2ntrf` files.

**(c) Concurrent `model_on` trials: blocked, not interleaved.**
- Tejeda et al. (2022) Procedure: "4 blocks where each block consisted of 48 consecutive trials in which AI assistance was turned on, and 16 consecutive trials without AI assistance." The OSF wiki says the same.
- Data (`model_on` ordered by `task_number` / `task_end_time`): 59 of 65 participants follow (48 on, 16 off) × 4 exactly. The other 6 are truncated or anomalous (see the audit notes), and none is interleaved. Unassisted trials always come after 48 assisted trials in the same block.
- Under §7 this makes H3a descriptive only.
- Sequential design: every trial is an unassisted initial response followed by an assisted final response on the same image. That holds for 14,097 of 14,154 pairs; the exceptions all belong to 2 participants.

## 2. De-identification

`pipeline/00_deidentify.py` checks the eight hashes, then writes `data/h16_deid.csv` (28,997 rows, 145 codes), `data/tejeda_concurrent_deid.csv` (16,363 rows, 65 codes) and `data/tejeda_sequential_deid.csv` (28,416 rows, 74 codes). `data/` is gitignored.

- `worker_id` and `participant_id` are replaced by a random code drawn with `secrets`: H/C/S by file, W for the 8 workers who appear in both ImageNet-16H and the concurrent design. The key is stored only at `~/Desktop/hai-reference-raw/deid_key.csv`.
- Dropped: `device_type`, `device_os`, `device_browser`, all wall-clock timestamps, the database row `id`, and the pandas index column. Order is kept as `task_order` and `t_end_s` (seconds since that participant's first response).
- Prediction files carry no personal data and are read from the raw folder; they are not copied.
- Check: all 276 raw worker ids were searched for in every file under the repository working tree (including `data/`), with no hit. The script also asserts this for each output. No raw file and no worker id is in the repository or in git history.
