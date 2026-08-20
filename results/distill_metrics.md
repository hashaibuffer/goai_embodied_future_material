# TE distillation metrics

## Data gates

- Accepted training data: 50 shards / 50,000 samples from
  `QC_50K_MANIFEST.json`.
- Independent validation: 13 shards / 6,500 samples after applying the QC
  exclusion list.
- Stress data: 3 shards / 2,150 high-step speed-boundary samples.
- Every accepted shard is schema v2, pickle-free, `float32`, finite, labeled
  by `privileged_mujoco`, tied to the frozen teacher SHA, and uses
  `teacher_action_raw` without `[-1, 1]` clipping.

## Final comparison

| Metric | Learned 441-D | Proprio-only 441-signature |
|---|---:|---:|
| Validation raw-action MAE | 0.187210 | 0.199307 |
| Validation RMSE | 0.444039 | — |
| Validation p95 absolute error | 0.893589 | — |
| Decoded leg-target MAE (rad) | 0.041690 | — |
| Decoded wheel-target MAE (rad/s) | 0.843906 | — |
| High-step stress MAE | 0.174378 | — |
| High-step stress RMSE | 0.286240 | — |
| High-step stress p95 | 0.561507 | — |

The learned model improves overall validation MAE by about 6.1%. It wins 6
of 13 terrain strata and passes the configured overall, per-stratum, height
usage, stress, and ONNX parity gates.

| Validation stratum | Learned MAE | Proprio-only MAE |
|---|---:|---:|
| high_step | 0.077753 | 0.112269 |
| regular_stairs_up | 0.114838 | 0.188633 |
| gentle_slope_down | 0.016250 | 0.013127 |
| high_plateau_wp09_10 | 0.050680 | 0.039009 |
| high_step_recovery | 0.087840 | 0.125160 |
| long_down_wp20_21 | 0.035741 | 0.034353 |
| low_plateau_wp13_14 | 0.016642 | 0.014869 |
| mild_up_wp05_06 | 0.044051 | 0.049339 |
| start_long_ramp_wp00_01 | 0.080482 | 0.075709 |
| steep_down_wp12_13 | 0.056687 | 0.058550 |
| steep_up_wp22_23 | 0.449626 | 0.309873 |
| steep_up_wp24_25 | 0.946705 | 0.916699 |
| steep_up_wp26_27 | 0.456435 | 0.653396 |

## Ablations

- Baseline learned MAE: 0.187210.
- Zero-terrain MAE: 1.036896; mean action L2 change: 4.993148.
- Shuffled-terrain MAE: 0.239679.
- Changing forward command from 0.3 to 0.6 changes all sampled actions;
  mean/p95 action L2 change is 0.080292/0.083704.

## Closed loop

All rows use 1,000 policy steps after the same stand-up sequence, command
`[0.5, 0, 0]`, and seed 42.

| Policy | Displacement (m) | z mean/min/final (m) | tilt p95/max (rad) |
|---|---:|---:|---:|
| Privileged teacher | 10.280 | 0.360 / 0.359 / 0.359 | 0.015 / 0.047 |
| Learned before DAgger | 15.300 | 0.319 / 0.309 / 0.309 | 0.097 / 0.103 |
| Learned after DAgger5 (selected) | 10.644 | 0.319 / 0.284 / 0.308 | 0.057 / 0.115 |
| Learned after DAgger10 (rejected) | 7.678 | 0.309 / 0.294 / 0.306 | 0.093 / 0.117 |

The selected checkpoint balances teacher-like speed and upright stability;
the ten-epoch candidate was rejected as too slow and weaker per stratum.

## Reproduction

```powershell
.venv-te\Scripts\python.exe -m training.distillation.train_student `
  --controller learned --architecture flat --flat-hidden 512 256 `
  --output-dir results/distillation_runs/learned_flat512_rawcmd_seed42 `
  --seed 42 --epochs 150 --patience 20

.venv-te\Scripts\python.exe -m training.distillation.train_student `
  --controller proprio_clone --architecture branch `
  --output-dir results/distillation_runs/proprio_clone_rawcmd_seed42 `
  --seed 42 --epochs 150 --patience 20
```

For DAgger, run `scripts/collect_mujoco_d_priv.py` with both
`--teacher-onnx` and `--rollout-policy`, then pass the resulting shard through
`--extra-shards` and the base checkpoint through `--init-checkpoint`. The
selected fine-tune used five epochs at learning rate `3e-5`.

```powershell
.venv-te\Scripts\python.exe -m training.distillation.train_student `
  --controller learned --architecture flat --flat-hidden 512 256 `
  --output-dir results/distillation_runs/learned_flat512_rawcmd_seed42_dagger5 `
  --seed 4242 --epochs 5 --patience 5 --learning-rate 0.00003 `
  --init-checkpoint results/distillation_runs/learned_flat512_rawcmd_seed42/best.pt `
  --extra-shards path/to/D_rollout.npz
```

Official ROS scoring remains pending on Ubuntu 24.04 with ROS 2 Jazzy.
