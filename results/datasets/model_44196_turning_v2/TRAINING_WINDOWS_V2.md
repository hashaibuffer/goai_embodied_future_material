# Turning v2 iterative fine-tune protocol

Turning v2 is intentionally larger than a one-shot patch dataset. It contains
31,608 formal samples across 32 shards and three official-map locations. Combine
it with turning v1 for 38,992 labelled turning samples.

## Data groups

- 16 complete privileged-teacher demonstrations.
- 14 complete student-visitation DAgger shards.
- 2 student pre-fall boundary DAgger shards.
- Commands cover `vx 0..0.5`, `vy -0.2..0.2`, `wz -0.6..0.6`.
- Courses cover pure yaw, forward arcs, left/right switchback, turn-then-go and
  lateral/yaw coupling.
- Starts cover official start, high plateau and low plateau with pose/yaw jitter.

`PROBE` is diagnostic-only. Train only from `FORMAL`.

## Round 1: primary branch

```powershell
$turnV1 = Get-ChildItem results/datasets/model_44196_turning_v1/FORMAL/*/*.npz |
  ForEach-Object FullName
$turnV2 = Get-ChildItem results/datasets/model_44196_turning_v2/FORMAL/*/*.npz |
  ForEach-Object FullName
$turn = @($turnV1) + @($turnV2)
$primary = "results/distillation_runs/learned_flat512_rawcmd_seed42_dagger5/last.pt"
$preserve = @(
  "results/distillation_runs/diag_gated_fullobs/official_start_seed42.npz",
  "results/distillation_runs/diag_current/stairs_7026.npz",
  "results/distillation_runs/diag_current/high_step_7041.npz",
  "results/distillation_runs/diag_current/high_step_recovery_7043.npz",
  "results/distillation_runs/diag_current/long_down_7036.npz",
  "results/distillation_runs/diag_current/steep_down_7031.npz"
)

.venv-te\Scripts\python.exe -m training.distillation.train_student `
  --controller learned --architecture flat --flat-hidden 512 256 `
  --init-checkpoint $primary --extra-shards $turn `
  --preserve-checkpoint $primary --preserve-shards $preserve `
  --preserve-weight 1.0 --learning-rate 3e-5 `
  --epochs 60 --patience 10 --seed 52 `
  --output-dir results/distillation_runs/turning_v2_primary_seed52
```

Run seeds 52, 53 and 54. Reject candidates that improve turning but regress any
preservation route. Do not select by aggregate validation MAE alone.

## Round 1 gate and export

Rebuild the hard gate with the selected turning primary, the existing ramp
recovery branch, existing ramp states as positives, and successful-route plus
turning states as negatives. Run gate seeds 52–54, export ONNX, and verify
Torch/ONNX parity below `1e-4`.

## Closed-loop gates before round 2

1. Run all eight v2 courses at all three start locations for 1,000 steps.
2. Run official start AutoNav through waypoint 0 without tumble or orbiting.
3. Re-run official start, ramp, stairs, high-step, recovery and descent checks.
4. Record every candidate failure as a new teacher-labelled DAgger shard.

## Round 2

Fine-tune only after collecting candidate-policy failure states from round 1.
Use a lower learning rate (`1e-5`) and 20–40 epochs with the same preservation
loss. Rebuild the gate and repeat all closed-loop gates. Continue DAgger rounds
until the visual ROS 2/Jazzy run reaches waypoints `0..32` without teleport,
tumble, stall or out-of-bounds events.

This protocol assumes more than one fine-tune. A model is accepted by closed-loop
behavior and preservation, not by the number of completed training rounds.
