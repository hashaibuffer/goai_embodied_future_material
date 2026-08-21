#!/usr/bin/env python3
"""Export a trained TE student as a verified dynamic-batch opset-17 ONNX."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .te_dataset import ACTION_DIM, OBS_DIM, sha256_file
from .te_model import load_checkpoint, parameter_count


ROOT = Path(__file__).resolve().parents[2]


def portable_artifact_path(path: str | Path | None) -> str | None:
    """Represent build inputs without publishing a machine-specific path."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def portable_training_metadata(metadata: dict) -> dict:
    """Copy checkpoint metadata and normalize the known artifact path fields."""
    result = copy.deepcopy(metadata)
    if "training_manifest" in result:
        result["training_manifest"] = portable_artifact_path(
            result["training_manifest"])
    for field in (
            "init_checkpoint", "preserve_checkpoint", "primary_checkpoint",
            "recovery_checkpoint"):
        if result.get(field):
            result[field] = portable_artifact_path(result[field])
    for collection in (
            "extra_shards", "preserve_shards", "positive_shards",
            "negative_shards"):
        for shard in result.get(collection, []):
            if "path" in shard:
                shard["path"] = portable_artifact_path(shard["path"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-atol", type=float, default=1e-4)
    parser.add_argument("--parity-rtol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _, training_metadata = load_checkpoint(args.checkpoint)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe = torch.linspace(-1.0, 1.0, OBS_DIM, dtype=torch.float32)[None]
    torch.onnx.export(
        model,
        probe,
        args.output,
        opset_version=17,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or graph.graph.input[0].name != "obs":
        raise RuntimeError("ONNX input must be named obs")
    if len(graph.graph.output) != 1 or graph.graph.output[0].name != "actions":
        raise RuntimeError("ONNX output must be named actions")
    input_shape = graph.graph.input[0].type.tensor_type.shape.dim
    output_shape = graph.graph.output[0].type.tensor_type.shape.dim
    if not input_shape[0].dim_param or input_shape[1].dim_value != OBS_DIM:
        raise RuntimeError("ONNX input must be dynamic [batch,441]")
    if not output_shape[0].dim_param or output_shape[1].dim_value != ACTION_DIM:
        raise RuntimeError("ONNX output must be dynamic [batch,16]")
    opsets = {item.domain: item.version for item in graph.opset_import}
    if opsets.get("") != 17:
        raise RuntimeError(f"expected default opset 17, got {opsets}")

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20260820)
    parity = []
    max_abs_error = 0.0
    input_mean = model.input_mean.detach().numpy()
    input_std = model.input_std.detach().numpy()
    for batch_size in (1, 7, 32):
        # Probe the frozen raw-input distribution. Unit-normal random raw inputs
        # are wildly OOD for near-constant features and can magnify otherwise
        # harmless backend rounding by several orders of magnitude.
        observations = (
            input_mean[None]
            + rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32)
            * input_std[None]
        ).astype(np.float32)
        observations[:, 249:] = rng.integers(
            0, 2, size=(batch_size, OBS_DIM - 249)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(observations)).numpy()
        actual = session.run(["actions"], {"obs": observations})[0]
        error = float(np.max(np.abs(expected - actual)))
        max_abs_error = max(max_abs_error, error)
        if error > args.parity_atol or not np.allclose(
                expected, actual, rtol=args.parity_rtol, atol=args.parity_atol):
            raise RuntimeError(
                f"Torch/ONNX parity failed for batch {batch_size}: max error={error:.3e}")
        parity.append({"batch": batch_size, "max_abs_error": error})

    sidecar = {
        "checkpoint": portable_artifact_path(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "onnx_sha256": sha256_file(args.output),
        "controller": model.config.controller,
        "architecture": model.config.architecture,
        "parameter_count": parameter_count(model),
        "input": {"name": "obs", "dtype": "float32", "shape": ["batch", OBS_DIM]},
        "output": {
            "name": "actions",
            "dtype": "float32",
            "shape": ["batch", ACTION_DIM],
            "contract": "raw actor action; no tanh; no [-1,1] clipping",
        },
        "opset": 17,
        "torch_onnx_parity": {
            "checked": True,
            "max_abs_error": max_abs_error,
            "batches": parity,
            "atol": args.parity_atol,
            "rtol": args.parity_rtol,
        },
        "training_metadata": portable_training_metadata(training_metadata),
    }
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
