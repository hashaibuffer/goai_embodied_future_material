#!/usr/bin/env python3
"""Export an S10 asymmetric CNN-GRU checkpoint with explicit recurrent state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HeightCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(),
            # Exact fixed-grid replacement for AdaptiveAvgPool2d((5, 4)).
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 5 * 4, 128), nn.ELU(), nn.Linear(128, 64)
        )

    def forward(self, height_map: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.conv(height_map))


class RecurrentTeacher(nn.Module):
    """command, proprio, height, h_prev -> raw action, h_next."""

    def __init__(self) -> None:
        super().__init__()
        self.height_cnn = HeightCNN()
        self.gru = nn.GRU(60, 128, num_layers=1, batch_first=False)
        self.mlp = nn.Sequential(
            nn.Linear(252, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 16),
        )

    def forward(self, command, proprio, height_map, h_prev):
        terrain = self.height_cnn(height_map)
        recurrent_input = torch.cat((command, proprio), dim=-1)
        recurrent, h_next = self.gru(
            recurrent_input.unsqueeze(0), h_prev.unsqueeze(0)
        )
        latent = torch.cat(
            (command, proprio, terrain, recurrent.squeeze(0)), dim=-1
        )
        action = torch.clamp(self.mlp(latent), -100.0, 100.0)
        return action, h_next.squeeze(0)


def load_actor(checkpoint_path: Path) -> RecurrentTeacher:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path} has no actor_state_dict")
    required = ("height_cnn.conv.0.weight", "gru.weight_ih_l0", "mlp.0.weight")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(
            f"{checkpoint_path} is not an S10 asymmetric CNN-GRU Actor; missing {missing}"
        )
    model = RecurrentTeacher().eval()
    actor_state = {
        name: value
        for name, value in state.items()
        if not name.startswith("distribution.")
    }
    incompatible = model.load_state_dict(actor_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint architecture mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model = load_actor(checkpoint)
    dummy = (
        torch.zeros(1, 3),
        torch.zeros(1, 57),
        torch.zeros(1, 1, 41, 33),
        torch.zeros(1, 128),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output,
        export_params=True,
        opset_version=18,
        input_names=["command", "proprio", "height_map", "h_prev"],
        output_names=["actions", "h_next"],
        dynamic_axes={
            "command": {0: "batch"},
            "proprio": {0: "batch"},
            "height_map": {0: "batch"},
            "h_prev": {0: "batch"},
            "actions": {0: "batch"},
            "h_next": {0: "batch"},
        },
    )

    parity = {"checked": False, "reason": "onnxruntime is unavailable"}
    try:
        import onnxruntime as ort
    except ImportError:
        pass
    else:
        rng = np.random.default_rng(7)
        inputs = (
            torch.from_numpy(rng.normal(size=(3, 3)).astype(np.float32)),
            torch.from_numpy(rng.normal(size=(3, 57)).astype(np.float32)),
            torch.from_numpy(rng.normal(size=(3, 1, 41, 33)).astype(np.float32)),
            torch.from_numpy(rng.normal(size=(3, 128)).astype(np.float32)),
        )
        with torch.no_grad():
            expected = model(*inputs)
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(
            ["actions", "h_next"],
            {
                name: tensor.numpy()
                for name, tensor in zip(
                    ("command", "proprio", "height_map", "h_prev"), inputs
                )
            },
        )
        errors = [
            float(np.max(np.abs(reference.numpy() - observed)))
            for reference, observed in zip(expected, actual)
        ]
        if not all(
            np.allclose(reference.numpy(), observed, rtol=1e-4, atol=1e-5)
            for reference, observed in zip(expected, actual)
        ):
            raise RuntimeError(f"Torch/ONNX recurrent parity failed: {errors}")
        parity = {"checked": True, "max_abs_error": max(errors)}

    sidecar = {
        "protocol": "s10-asymmetric-cnn-gru-1413-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(output),
        "inputs": {
            "command": ["batch", 3],
            "proprio": ["batch", 57],
            "height_map": ["batch", 1, 41, 33],
            "h_prev": ["batch", 128],
        },
        "outputs": {"actions": ["batch", 16], "h_next": ["batch", 128]},
        "action_contract": "raw_clip_100",
        "torch_onnx_parity": parity,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
