# TE terrain locomotion student

## Intended use

`terrain_locomotion.onnx` is the learned TE deployment policy for the S10
terrain-aware controller. `proprio_clone.onnx` is a separately trained
proprio-only comparison/fallback model; it is not the selected controller.

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

- Architecture: flat MLP `441 -> 512 -> 256 -> 16`, ELU activations.
- Parameters: 361,744; artifact size: 1,454,532 bytes.
- Training: 50,000 strict schema-v2 privileged-teacher samples, followed by
  five low-learning-rate DAgger epochs using 2,000 student-visited states.
- Teacher ONNX SHA-256:
  `857f2d59c04b6ee979e3fc776e2ca00e593c4d009297ed3b84427b0a5752b9be`.
- Student ONNX SHA-256:
  `a3fea3138912f6a72e70231c634035c03ceba6125123391ef068aec8250b1fb3`.
- Constant normalized sensor dimensions are masked and standardized values
  are clamped to `[-10, 10]` to prevent unseen inputs from amplifying
  untrained weights. Raw commands remain active.

## Validation

- Independent validation: 6,500 samples across 13 terrain/route strata.
- Learned raw-action MAE: 0.187210; proprio-only MAE: 0.199307.
- Learned RMSE / p95 absolute error: 0.444039 / 0.893589.
- Physical decoded MAE: 0.041690 rad for leg targets and 0.843906 rad/s for
  wheel targets.
- High-step stress set (2,150 samples): MAE 0.174378, RMSE 0.286240.
- Zeroing terrain increases MAE to 1.036896 and changes actions by mean L2
  4.993148. The model therefore does not collapse to a proprio-only policy.
- Torch/ONNX maximum absolute parity error over dynamic batches 1, 7, and 32:
  `1.90735e-6` (gate: `1e-4`).
- Headless MuJoCo at command `[0.5, 0, 0]`: 1,000 policy steps, 10.644 m
  displacement, tilt p95/max 0.057/0.115 rad, no fall. The privileged teacher
  baseline displaced 10.280 m with tilt p95/max 0.015/0.047 rad.
- Local ONNX Runtime CPU microbenchmark: about 23.55 microseconds per
  batch-1 inference (5,000 runs; machine-specific).

## Limitations

- The closed-loop result is a ROS-free Windows MuJoCo check at one start,
  command, and seed. Official Ubuntu 24.04 / ROS 2 Jazzy replay and competition
  scoring are still required.
- The learned model is worse than the proprio-only comparison on the held-out
  `steep_up_wp22_23` and `steep_up_wp24_25` strata. These segments need special
  attention during Jazzy replay.
- The training data is dominated by forward motion. Turning/lateral command
  sweeps and sensor-fault rollouts remain useful follow-up coverage.

Detailed metrics and reproduction commands are in `results/distill_metrics.md`.
