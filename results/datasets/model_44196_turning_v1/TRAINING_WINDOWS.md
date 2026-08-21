# Turning v1 Windows fine-tune

Use the training machine checkout at commit `e383d95` or later. The formal
manifest contains 7,384 strict schema-v2 teacher-labelled samples. The two
short left-turn shards are intentional student-visited pre-fall DAgger states.

## Required existing artifacts

The selected deployment ONNX is a gated ensemble, so do not fine-tune the ONNX
directly. The training machine must retain these checkpoints named in the model
metadata:

- `results/distillation_runs/learned_flat512_rawcmd_seed42_dagger5/last.pt`
- `results/distillation_runs/learned_flat512_rawcmd_seed42_dagger_route5/last.pt`
- the ramp-positive and route-preservation shards referenced by
  `models/terrain_locomotion.onnx.json`

Verify the primary and recovery checkpoint hashes against the model metadata
before training.

## 1. Fine-tune the primary branch

Run in PowerShell from the repository root:

```powershell
$turn = Get-ChildItem results/datasets/model_44196_turning_v1/FORMAL/*/*.npz |
  ForEach-Object FullName
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
  --epochs 40 --patience 8 --seed 52 `
  --output-dir results/distillation_runs/turning_v1_primary_seed52
```

Repeat seeds 53 and 54. Select by offline validation plus closed-loop turning;
do not select only by aggregate MAE.

## 2. Rebuild the hard gate

Use the selected turning primary as `--primary-checkpoint`, retain the current
ramp recovery checkpoint as `--recovery-checkpoint`, use the existing ramp
rollouts as positives, and combine the turning shards with the established
successful-route rollouts as negatives. Run `train_gate.py` with seeds 52–54.
This makes turning select the updated primary while preserving ramp recovery.

## 3. Export and acceptance

Export the selected gated checkpoint with `export_student_onnx.py`, then require:

- Torch/ONNX maximum absolute error below `1e-4` for batches 1, 7 and 32.
- No regression on official start, stairs, high step, recovery and descents.
- Both left and right turning curricula complete 1,000 steps.
- Ubuntu/ROS 2 Jazzy visual AutoNav reaches all waypoints `0..32` without
  teleport, tumble, stall or out-of-bounds events.

Only after these gates pass should `models/terrain_locomotion.onnx` be replaced.
