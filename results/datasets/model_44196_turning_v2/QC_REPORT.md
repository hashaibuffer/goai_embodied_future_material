# model_44196 turning v2 QC

- Formal shards: 32
- Formal samples: 31608
- Categories: `{'dagger_success': 14, 'dagger_failure_boundary': 2, 'teacher_demo': 16}`
- Teacher SHA-256: `857f2d59c04b6ee979e3fc776e2ca00e593c4d009297ed3b84427b0a5752b9be`
- Student behavior SHA-256: `edc785e9c2859ddbfb0622aea7785e679bc8fa231fccac12b1ffe7569f99589e`
- Command bounds `[vx, vy, wz]`: `[0.0, -0.20000000298023224, -0.6000000238418579]` to `[0.5, 0.20000000298023224, 0.6000000238418579]`
- Duplicate file hashes: `0`
- All observations/actions finite: `True`
- Probe shards are diagnostic-only and excluded from this manifest.

| Category | Shards | Samples |
|---|---:|---:|
| dagger_failure_boundary | 2 | 1608 |
| dagger_success | 14 | 14000 |
| teacher_demo | 16 | 16000 |

## Failure-boundary shards

| Shard | Samples | z min/final | tilt max |
|---|---:|---:|---:|
| FORMAL/dagger/D_priv_turnv2_arc_left_official_dagger_8412.npz | 693 | 0.080/0.081 | 3.138 |
| FORMAL/dagger/D_priv_turnv2_yaw_left_official_dagger_8402.npz | 915 | 0.080/0.081 | 3.110 |
