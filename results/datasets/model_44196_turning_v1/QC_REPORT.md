# model_44196 turning v1 QC

- Shards: 8
- Samples: 7384
- Teacher SHA-256: `857f2d59c04b6ee979e3fc776e2ca00e593c4d009297ed3b84427b0a5752b9be`
- All observations/actions finite: True
- Command coverage: `vx 0.0..0.5`, `vy=0`, `wz -0.6..0.6`

| Shard | Role | N | z min/final | tilt max |
|---|---|---:|---:|---:|
| FORMAL/dagger/D_priv_turn_left_curriculum_dagger_8201.npz | dagger_failure_boundary | 779 | 0.080/0.081 | 3.142 |
| FORMAL/dagger/D_priv_turn_left_transition_dagger_8203.npz | dagger_failure_boundary | 605 | 0.081/0.081 | 3.104 |
| FORMAL/dagger/D_priv_turn_right_curriculum_dagger_8202.npz | dagger | 1000 | 0.220/0.356 | 0.200 |
| FORMAL/dagger/D_priv_turn_right_transition_dagger_8204.npz | dagger | 1000 | 0.226/0.333 | 0.229 |
| FORMAL/teacher/D_priv_turn_left_curriculum_8101.npz | teacher_demo | 1000 | 0.313/0.359 | 0.069 |
| FORMAL/teacher/D_priv_turn_left_transition_8103.npz | teacher_demo | 1000 | 0.315/0.363 | 0.065 |
| FORMAL/teacher/D_priv_turn_right_curriculum_8102.npz | teacher_demo | 1000 | 0.341/0.358 | 0.030 |
| FORMAL/teacher/D_priv_turn_right_transition_8104.npz | teacher_demo | 1000 | 0.341/0.363 | 0.048 |

The two incomplete left-turn DAgger shards intentionally preserve student-visited pre-fall boundary states. They are valid teacher-labelled DAgger inputs, not complete teacher demonstrations.
