# Amendment 5: human increment against the raw classifier (sensitivity only)

SPEC v1 Amendment 5, LOCKED 2026-10-08, spec commit `3fa0858630f12d4edd4f085a6899b309ceeac343`. Repo HEAD at run: `3fa0858630f12d4edd4f085a6899b309ceeac343`.

- **Status.** After results, sensitivity only. No locked estimate changes; every `I_recal` below is the locked H1c value.
- **Deviation from the spec wording.** The spec names `code/amendment5_rawF0.py`. This repository keeps its scripts in `pipeline/`, so the script is `pipeline/amendment5_rawF0.py`.
- **Raw F0.** F0's single weight is fixed at w_m = 1 (no fitted weight, no recalibration), so F0 is the classifier vector itself. To avoid log(0) it carries the spec's existing clip: probabilities clipped below at 1e-3 and renormalized (`clipren`, CLIP = 1e-3), the same treatment every classifier vector already has in the locked run. The with-human pools are unchanged: two fitted weights, the same 5 folds by image (seed 20260915), the same clipping.
- **Intervals.** Image-cluster bootstrap, 2000 draws, seed 20260916, percentile 95%, the full run's draws. The difference `I_raw − I_recal` is bootstrapped per item on the same draws, so its interval is paired. Slope intervals use the same draws with flags fixed from the point estimates.
- **Reproduction check.** With the flag off, all 154 H1c values compared (I_h_ind and I_h_pool, mean and interval for 20 variants; Q; both slopes with intervals) are bit-identical to `reports/full_tables/h1c_curve.csv` and `h1c_slopes.csv`.
- **Individual and pooled differences coincide.** Only F0 changes; the with-human pools do not. Per item, both increments therefore shift by the same F0 loss difference, and the I_raw − I_recal columns of the two tables are identical. Rule (a) can still differ between them, because its 20% threshold is taken on each version's own I_recal.
- Optimizer converged in every fit: yes.

Units: nats per item. Positive increments mean the human lowers out-of-fold loss.

## Locked reading (restated)

- The difference is expected to be positive wherever the raw classifier is miscalibrated (recalibration takes credit the human would otherwise get).
- The choice is material, and becomes a numbered item in the reporting standard, if (a) for at least one variant the paired interval on I_raw − I_recal excludes zero and the difference exceeds 20% of I_recal, or (b) the slope of I on baseline loss changes by more than the half-width of its locked interval. Otherwise the item moves to the discussion as a recommendation without a demonstration.

## Verdict

**Material.** The choice of no-human term is material under the locked rule: rule (a) holds for 3 variant-quantity pairs; rule (b) holds for I_h_ind slope_all, I_h_ind slope_unflagged, I_h_pool slope_all, I_h_pool slope_unflagged. It becomes a numbered item in the reporting standard.

## Per variant, individual `I(h | m)`

Ordered by baseline loss Q (selection-set log loss). Rule (a): paired interval excludes zero and difference > 0.2 × I_recal.

| variant | baseline loss Q | I_recal [95% CI] | I_raw [95% CI] | I_raw − I_recal [paired 95% CI] | rule (a) |
|---|---|---|---|---|---|
| vgg19_epoch10 | 0.436 | +0.131 [+0.107, +0.157] | +0.131 [+0.107, +0.157] | +0.000 [−0.000, +0.000] | no |
| densenet161_epoch10 | 0.477 | +0.151 [+0.125, +0.178] | +0.150 [+0.124, +0.178] | −0.000 [−0.001, +0.000] | no |
| vgg19_epoch01 | 0.608 | +0.187 [+0.159, +0.216] | +0.187 [+0.158, +0.217] | −0.000 [−0.000, +0.000] | no |
| resnet152_epoch01 | 0.690 | +0.274 [+0.240, +0.309] | +0.281 [+0.242, +0.321] | +0.007 [+0.001, +0.013] | no |
| densenet161_epoch01 | 0.702 | +0.236 [+0.203, +0.270] | +0.237 [+0.202, +0.273] | +0.001 [−0.002, +0.004] | no |
| googlenet_epoch01 | 0.749 | +0.276 [+0.241, +0.311] | +0.278 [+0.239, +0.315] | +0.002 [−0.002, +0.005] | no |
| resnet152_epoch10 | 0.808 | +0.292 [+0.256, +0.330] | +0.308 [+0.264, +0.354] | +0.016 [+0.007, +0.026] | no |
| alexnet_epoch10 | 0.829 | +0.351 [+0.313, +0.387] | +0.351 [+0.312, +0.387] | −0.000 [−0.000, −0.000] | no |
| googlenet_epoch10 | 0.840 | +0.295 [+0.259, +0.333] | +0.299 [+0.260, +0.341] | +0.004 [−0.001, +0.009] | no |
| densenet161_epoch00 | 0.884 | +0.370 [+0.330, +0.414] | +0.383 [+0.336, +0.432] | +0.013 [+0.004, +0.022] | no |
| alexnet_epoch01 | 0.942 | +0.408 [+0.370, +0.449] | +0.408 [+0.370, +0.448] | −0.000 [−0.001, +0.001] | no |
| vgg19_epoch00 | 1.075 | +0.474 [+0.436, +0.515] | +0.474 [+0.435, +0.515] | −0.000 [−0.000, +0.000] | no |
| resnet152_epoch00 | 1.177 | +0.514 [+0.470, +0.561] | +0.554 [+0.499, +0.615] | +0.039 [+0.025, +0.055] | no |
| googlenet_epoch00 | 1.223 | +0.581 [+0.533, +0.632] | +0.586 [+0.533, +0.639] | +0.004 [−0.001, +0.009] | no |
| alexnet_epoch00 | 1.377 | +0.800 [+0.749, +0.854] | +0.800 [+0.749, +0.853] | −0.000 [−0.001, +0.000] | no |
| googlenet_baseline | 1.753 | +0.955 [+0.904, +1.006] | +0.958 [+0.904, +1.011] | +0.002 [−0.001, +0.006] | no |
| densenet161_baseline | 1.840 | +0.880 [+0.828, +0.931] | +1.036 [+0.963, +1.107] | +0.156 [+0.129, +0.183] | no |
| resnet152_baseline | 2.130 | +1.039 [+0.983, +1.095] | +1.275 [+1.190, +1.361] | +0.237 [+0.202, +0.274] | yes |
| vgg19_baseline | 2.313 | +1.247 [+1.188, +1.306] | +1.501 [+1.414, +1.590] | +0.254 [+0.215, +0.293] | yes |
| alexnet_baseline | 2.556 | +1.409 [+1.355, +1.461] | +1.721 [+1.630, +1.809] | +0.313 [+0.265, +0.359] | yes |

## Per variant, pooled `I(h | m)`

Ordered by baseline loss Q (selection-set log loss). Rule (a): paired interval excludes zero and difference > 0.2 × I_recal.

| variant | baseline loss Q | I_recal [95% CI] | I_raw [95% CI] | I_raw − I_recal [paired 95% CI] | rule (a) |
|---|---|---|---|---|---|
| vgg19_epoch10 | 0.436 | +0.224 [+0.183, +0.264] | +0.224 [+0.183, +0.264] | +0.000 [−0.000, +0.000] | no |
| densenet161_epoch10 | 0.477 | +0.254 [+0.213, +0.299] | +0.254 [+0.213, +0.299] | −0.000 [−0.001, +0.000] | no |
| vgg19_epoch01 | 0.608 | +0.311 [+0.266, +0.358] | +0.311 [+0.266, +0.358] | −0.000 [−0.000, +0.000] | no |
| resnet152_epoch01 | 0.690 | +0.446 [+0.395, +0.498] | +0.453 [+0.396, +0.509] | +0.007 [+0.001, +0.013] | no |
| densenet161_epoch01 | 0.702 | +0.386 [+0.333, +0.438] | +0.386 [+0.331, +0.441] | +0.001 [−0.002, +0.004] | no |
| googlenet_epoch01 | 0.749 | +0.437 [+0.382, +0.494] | +0.439 [+0.381, +0.499] | +0.002 [−0.002, +0.005] | no |
| resnet152_epoch10 | 0.808 | +0.478 [+0.421, +0.537] | +0.494 [+0.430, +0.560] | +0.016 [+0.007, +0.026] | no |
| alexnet_epoch10 | 0.829 | +0.531 [+0.474, +0.584] | +0.531 [+0.474, +0.584] | −0.000 [−0.000, −0.000] | no |
| googlenet_epoch10 | 0.840 | +0.473 [+0.417, +0.530] | +0.477 [+0.417, +0.538] | +0.004 [−0.001, +0.009] | no |
| densenet161_epoch00 | 0.884 | +0.559 [+0.501, +0.621] | +0.572 [+0.507, +0.641] | +0.013 [+0.004, +0.022] | no |
| alexnet_epoch01 | 0.942 | +0.592 [+0.535, +0.650] | +0.591 [+0.535, +0.649] | −0.000 [−0.001, +0.001] | no |
| vgg19_epoch00 | 1.075 | +0.706 [+0.649, +0.769] | +0.706 [+0.648, +0.769] | −0.000 [−0.000, +0.000] | no |
| resnet152_epoch00 | 1.177 | +0.780 [+0.714, +0.847] | +0.820 [+0.742, +0.899] | +0.039 [+0.025, +0.055] | no |
| googlenet_epoch00 | 1.223 | +0.832 [+0.766, +0.899] | +0.836 [+0.766, +0.906] | +0.004 [−0.001, +0.009] | no |
| alexnet_epoch00 | 1.377 | +1.093 [+1.025, +1.160] | +1.093 [+1.024, +1.159] | −0.000 [−0.001, +0.000] | no |
| googlenet_baseline | 1.753 | +1.309 [+1.241, +1.375] | +1.311 [+1.241, +1.379] | +0.002 [−0.001, +0.006] | no |
| densenet161_baseline | 1.840 | +1.233 [+1.164, +1.298] | +1.389 [+1.299, +1.477] | +0.156 [+0.129, +0.183] | no |
| resnet152_baseline | 2.130 | +1.407 [+1.336, +1.480] | +1.644 [+1.545, +1.741] | +0.237 [+0.202, +0.274] | no |
| vgg19_baseline | 2.313 | +1.637 [+1.563, +1.709] | +1.891 [+1.786, +1.991] | +0.254 [+0.215, +0.293] | no |
| alexnet_baseline | 2.556 | +1.813 [+1.746, +1.878] | +2.125 [+2.021, +2.219] | +0.313 [+0.265, +0.359] | no |

## Slopes on baseline loss

Rule (b): |slope_raw − slope_recal| > half-width of the locked interval. All points and unflagged points are co-primary, as in H1c.

| quantity | slope | I_recal (locked) [95% CI] | I_raw [95% CI] | change | locked half-width | flagged recal / raw | rule (b) |
|---|---|---|---|---|---|---|---|
| I_h_ind | slope_all | +0.603 [+0.575, +0.631] | +0.743 [+0.703, +0.780] | +0.140 | 0.028 | 4 / 2 | yes |
| I_h_ind | slope_unflagged | +0.591 [+0.557, +0.625] | +0.685 [+0.642, +0.729] | +0.095 | 0.034 | 4 / 2 | yes |
| I_h_pool | slope_all | +0.756 [+0.722, +0.790] | +0.896 [+0.850, +0.939] | +0.140 | 0.034 | 4 / 2 | yes |
| I_h_pool | slope_unflagged | +0.780 [+0.738, +0.823] | +0.860 [+0.809, +0.911] | +0.080 | 0.042 | 4 / 2 | yes |

Full values: `reports/amendment5_results.json`.
