# TE terrain locomotion student

## Intended use

`terrain_locomotion.onnx` is the selected TE deployment policy for the S10
terrain-aware controller. `proprio_clone.onnx` is only a separately trained
proprioception-only comparison model.

## Frozen interface

- Input: `obs`, `float32`, dynamic shape `[batch, 441]`.
- Layout: official proprioception 57 + height values 192 + validity bits 192.
- Command slots `obs[:, 6:9]` and validity bits remain in their bounded raw
  representation. Other sensor/height values are normalized in the graph.
- Output: `actions`, `float32`, dynamic shape `[batch, 16]`.
- Output contract: raw actor action. There is no `tanh`, no `[-1, 1]`
  clipping, and the official action decoder applies scaling exactly once.
- ONNX opset: 17.

## Selected model

- Architecture: hard-gated ensemble. The primary and ramp-recovery branches
  are both `441 -> 512 -> 256 -> 16` ELU policies. A full-observation
  `441 -> 512 -> 256 -> 1` gate selects exactly one branch with `Where`; the
  action vectors are never blended.
- Parameters: 1,081,377; artifact size: 4,335,605 bytes.
- The primary branch is the previous DAgger5 policy. It is kept bit-for-bit
  inside the ensemble for the official start, stairs, high step, recovery,
  and descending segments.
- The recovery branch is a low-learning-rate DAgger fine-tune on safe
  privileged-teacher ramp states. The gate was trained on 12,354 ramp and
  22,000 non-ramp/student-visited observations.
- Gate training classification: 34,354/34,354 correct, minimum logit margin
  2.001785.
- Teacher ONNX SHA-256:
  `857f2d59c04b6ee979e3fc776e2ca00e593c4d009297ed3b84427b0a5752b9be`.
- Student ONNX SHA-256:
  `edc785e9c2859ddbfb0622aea7785e679bc8fa231fccac12b1ffe7569f99589e`.

## Formal-data audit

The 44 shards in `model_44196_full_route/FORMAL` satisfy the strict schema-v2
and teacher-hash contract, but byte-level array comparison found that every
shard duplicates an episode already present in the earlier
`model_44196_raw_v2/CANDIDATE_50K` pool. Of the 44,000 samples, 37,000 were
already in the accepted QC training set; 5,000 came from previously rejected
incomplete/fallen ramp or stair episodes; and 2,000 high-step samples
(episodes 7040 and 7042) contain near-total `+/-100` action saturation after
the first few frames and were excluded. The formal upload was therefore not
treated as 44,000 new independent training samples.

## Validation

- Independent offline validation: 6,500 samples across 13 route strata.
- Final gated MAE/RMSE/p95 absolute error: 0.193127 / 0.447723 / 0.932715.
- Proprio-only MAE: 0.199307. The gated policy is better overall, but wins
  only 5/13 strata, so it does not pass the older per-stratum reference gate.
- High-step stress MAE/RMSE/p95: 0.178402 / 0.291461 / 0.579387.
- Zeroing terrain raises MAE to 1.035432 and changes actions by mean L2
  4.982011, confirming that terrain inputs remain active.
- Torch/ONNX maximum absolute parity error over dynamic batches 1, 7, and 32:
  `2.62261e-6` (acceptance tolerance `1e-4`).
- Local ONNX Runtime CPU microbenchmark: about 54.09 microseconds per batch-1
  inference (5,000 runs; machine-specific).

## Headless closed-loop checks

Every row completed 1,000 policy steps after the normal stand-up sequence.
The gate selected the expected branch for every frame, with zero switches.

| Start/segment | Branch | Displacement (m) | z min/final/max (m) | tilt p95/max (rad) |
|---|---|---:|---:|---:|
| Official start, vx=0.5 | primary | 10.644 | 0.284 / 0.308 / 0.407 | 0.069 / 0.139 |
| Ramp episode 7002 | recovery | 6.876 | 0.341 / 0.790 / 0.853 | 0.208 / 0.411 |
| Ramp episode 7003 | recovery | 12.543 | 0.340 / 0.818 / 0.822 | 0.153 / 0.216 |
| Ramp episode 7005 | recovery | 12.118 | 0.339 / 0.818 / 0.822 | 0.142 / 0.172 |
| Stairs episode 7026 | primary | 6.298 | 0.781 / 1.499 / 1.527 | 0.328 / 0.374 |
| High step episode 7041 | primary | 2.726 | 1.459 / 2.255 / 2.281 | 0.431 / 0.606 |
| High-step recovery 7043 | primary | 1.848 | 1.483 / 1.647 / 1.939 | 0.411 / 0.481 |
| Long descent 7036 | primary | 9.586 | 2.055 / 2.055 / 2.745 | 0.231 / 0.434 |
| Steep descent 7031 | primary | 11.535 | 0.773 / 0.910 / 1.557 | 0.256 / 0.305 |

The previous primary policy fell or ended badly tilted on ramp episodes 7002,
7003, and 7005. The selected hard-gated policy completes all three while
preserving the successful primary-policy trajectories on the other checks.

## Limitations

- These are ROS-free Windows headless MuJoCo checks. Official Ubuntu 24.04 /
  ROS 2 Jazzy replay and competition scoring are still required.
- Offline teacher-trajectory MAE and closed-loop robustness disagree for this
  change: the final model deliberately accepts a small offline regression to
  repair student-visited ramp states.
- `steep_up_wp22_23` and `steep_up_wp24_25` remain weak offline strata and need
  special attention during Jazzy replay.
- The hard gate is validated on the collected starts and commands, not every
  possible transition. Turning/lateral commands and noisy/failing terrain
  sensors still need broader rollout coverage.

Detailed metrics are in `results/distill_metrics.md`.
