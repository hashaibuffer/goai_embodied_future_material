# TC placeholder model

- Purpose: deterministic deployment-chain smoke test; not a trained locomotion policy.
- Input: `obs`, `float32`, shape `[1, 441]`.
- Output: `actions`, `float32`, shape `[1, 16]`.
- Opset: 17.
- Behavior: returns sixteen zeros for every input. The official action decoder therefore commands the frozen default leg pose and zero wheel velocity.
- Contract source: `configs/policy.yaml` (T00).
- Replacement: TE replaces this artifact with the trained model without changing tensor names, shapes, ordering, units, or scaling.
