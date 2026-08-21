# TE distillation metrics

## Data audit and gates

- Original accepted training data: 50 shards / 50,000 samples from
  `QC_50K_MANIFEST.json`.
- Independent validation: 13 shards / 6,500 samples after applying the QC
  exclusion list.
- Stress data: 3 shards / 2,150 high-step speed-boundary samples.
- The new `model_44196_full_route/FORMAL` upload contains 44 strict schema-v2
  shards / 44,000 samples covering ten terrain labels, with the expected
  privileged-teacher SHA and unclipped raw actions.
- Exact array comparison shows all 44 formal episodes already exist in
  `model_44196_raw_v2/CANDIDATE_50K`: 37 episodes were already accepted, five
  were previously rejected for falling/incomplete traversal, and high-step
  episodes 7040/7042 have catastrophic action saturation. The last 2,000
  samples were excluded and the upload was not counted as new independent
  evidence.

The poor previous result was therefore a closed-loop covariate-shift problem,
not a shortage solved by repeating the formal samples. Directly fine-tuning a
single network repaired the ramps but regressed the already successful stairs
and high-step cases. The selected solution keeps both policies and learns a
hard full-observation gate.

## Final offline comparison

| Metric | Final gated 441-D | Previous primary | Proprio-only |
|---|---:|---:|---:|
| Validation raw-action MAE | 0.193127 | 0.187210 | 0.199307 |
| Validation RMSE | 0.447723 | 0.444039 | 0.461332 |
| Validation p95 absolute error | 0.932715 | 0.893589 | 0.958570 |
| Decoded leg-target MAE (rad) | 0.043358 | 0.041690 | 0.040370 |
| Decoded wheel-target MAE (rad/s) | 0.847219 | 0.843906 | 1.102932 |
| High-step stress MAE | 0.178402 | 0.174378 | — |
| High-step stress RMSE | 0.291461 | 0.286240 | — |
| High-step stress p95 | 0.579387 | 0.561507 | — |

The final model is better than the proprio-only reference overall but wins
only 5 of 13 validation strata, so the old configured per-stratum reference
gate reports false. This is an intentional, documented selection based on
closed-loop route recovery and exact preservation of the successful primary
branch, rather than on offline MAE alone.

## Terrain ablations

- Baseline final MAE: 0.193127.
- Zero-terrain MAE: 1.035432; mean/p95 action L2 change: 4.982011/5.934599.
- Shuffled-terrain MAE: 0.245837; mean/p95 action L2 change:
  0.748229/2.159836.
- Torch/ONNX maximum absolute error: `2.62261e-6` across dynamic batches 1,
  7, and 32.

## Hard-gate training

- Primary branch: previous selected flat DAgger5 policy.
- Recovery branch: route-focused flat policy after five low-learning-rate
  DAgger epochs.
- Gate input: all 441 observations; hidden widths 512 and 256.
- Positive/ramp states: 12,354; negative/preserved states: 22,000.
- Training classification errors: 0/34,354; minimum logit margin: 2.001785.
- Gate output contract: `logit > 0` selects recovery, otherwise primary;
  `Where` returns one complete 16-D action and never interpolates actions.

## Closed loop

All runs use 1,000 policy steps after the same stand-up sequence. Gate branch
decisions were evaluated on every saved observation; every row has zero branch
switches.

| Start/segment | Command vx | Selected branch | Displacement / path (m) | z min/final/max (m) | tilt p95/max (rad) |
|---|---:|---|---:|---:|---:|
| Official start | 0.5 | primary | 10.644 / 10.696 | 0.284 / 0.308 / 0.407 | 0.069 / 0.139 |
| Ramp 7002 | 0.6 | recovery | 6.876 / 7.776 | 0.341 / 0.790 / 0.853 | 0.208 / 0.411 |
| Ramp 7003 | 0.6 | recovery | 12.543 / 12.585 | 0.340 / 0.818 / 0.822 | 0.153 / 0.216 |
| Ramp 7005 | 0.6 | recovery | 12.118 / 12.166 | 0.339 / 0.818 / 0.822 | 0.142 / 0.172 |
| Stairs 7026 | 0.6 | primary | 6.298 / 7.310 | 0.781 / 1.499 / 1.527 | 0.328 / 0.374 |
| High step 7041 | 0.6 | primary | 2.726 / 4.616 | 1.459 / 2.255 / 2.281 | 0.431 / 0.606 |
| High-step recovery 7043 | 0.6 | primary | 1.848 / 5.851 | 1.483 / 1.647 / 1.939 | 0.411 / 0.481 |
| Long descent 7036 | 0.6 | primary | 9.586 / 10.184 | 2.055 / 2.055 / 2.745 | 0.231 / 0.434 |
| Steep descent 7031 | 0.6 | primary | 11.535 / 11.845 | 0.773 / 0.910 / 1.557 | 0.256 / 0.305 |

The previous primary policy ended ramp 7002 at z=0.249 m with 0.768 rad
maximum tilt and fell before step 700 on ramp 7003 and ramp 7005. The recovery
branch completes all three. Because the final gate makes the same hard branch
choice on all frames, the non-ramp rows exactly preserve the previous policy's
actions and trajectories.

## Reproduction outline

The base and route-recovery branches use `train_student.py`; the gate uses the
new `train_gate.py`. Diagnostic rollout paths are intentionally not committed,
so recollect the named positive and negative routes before running this form:

```powershell
.venv-te\Scripts\python.exe -m training.distillation.train_gate `
  --primary-checkpoint path/to/primary.pt `
  --recovery-checkpoint path/to/recovery.pt `
  --positive-shards path/to/ramp_rollout_1.npz path/to/ramp_rollout_2.npz `
  --negative-shards path/to/preserved_route_1.npz path/to/preserved_route_2.npz `
  --gate-hidden 512 256 --seed 46 `
  --output results/distillation_runs/gated_fullobs_seed46/best.pt

.venv-te\Scripts\python.exe -m training.distillation.export_student_onnx `
  --checkpoint results/distillation_runs/gated_fullobs_seed46/best.pt `
  --output models/terrain_locomotion.onnx
```

Official ROS scoring remains pending on Ubuntu 24.04 with ROS 2 Jazzy.
