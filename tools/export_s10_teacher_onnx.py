#!/usr/bin/env python3
"""Export a frozen RSL-RL S10 privileged actor as dynamic-batch 1413->16 ONNX."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn


class TeacherForExport(nn.Module):
    """Expose the teacher with the action contract expected by MuJoCo."""

    def __init__(self, actor, postprocess, action_limit):
        super().__init__()
        self.actor = actor
        self.postprocess = postprocess
        self.action_limit = float(action_limit)

    def forward(self, obs):
        action = self.actor(obs)
        if self.postprocess == "clip":
            return torch.clamp(action, -self.action_limit, self.action_limit)
        if self.postprocess == "tanh":
            return torch.tanh(action)
        if self.postprocess == "none":
            return action
        raise RuntimeError(f"unsupported action postprocess: {self.postprocess}")


def actor_state(checkpoint):
    if "actor_state_dict" in checkpoint:
        raw = checkpoint["actor_state_dict"]
        return {("actor." + key[4:] if key.startswith("mlp.") else key): value
                for key, value in raw.items() if key.startswith("mlp.")}
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    raise KeyError("checkpoint has neither actor_state_dict nor model_state_dict")


def build_actor(state):
    layers, index = [], 0
    while f"actor.{index}.weight" in state:
        weight, bias = state[f"actor.{index}.weight"], state[f"actor.{index}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight); linear.bias.data.copy_(bias)
        layers.append(linear)
        if f"actor.{index + 2}.weight" in state:
            layers.append(nn.ELU())
        index += 2
    if not layers:
        raise KeyError("checkpoint contains no actor MLP weights")
    return nn.Sequential(*layers)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action-postprocess", choices=("clip", "tanh", "none"), default="clip",
        help="postprocess actor mean inside ONNX (default: clip to [-1, 1])",
    )
    parser.add_argument(
        "--action-limit", type=float, default=1.0,
        help="symmetric limit for --action-postprocess clip (default: 1.0)",
    )
    args = parser.parse_args()
    if args.action_limit <= 0:
        parser.error("--action-limit must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = build_actor(actor_state(checkpoint)).eval()
    first = next(module for module in actor if isinstance(module, nn.Linear))
    last = [module for module in actor if isinstance(module, nn.Linear)][-1]
    if first.in_features != 1413 or last.out_features != 16:
        raise ValueError(
            f"not an S10 privileged teacher: actor is {first.in_features}->{last.out_features}, expected 1413->16")
    export_model = TeacherForExport(actor, args.action_postprocess, args.action_limit).eval()
    # Fail before writing an unsafe graph. The MuJoCo decoder consumes a_norm.
    probe = torch.randn(128, 1413)
    with torch.no_grad():
        probe_actions = export_model(probe)
    if not torch.isfinite(probe_actions).all():
        raise RuntimeError("exported actions contain non-finite values")
    if float(probe_actions.abs().max()) > 1.0 + 1e-5:
        raise RuntimeError(
            "exported actions exceed the MuJoCo a_norm range [-1, 1]; "
            "use --action-postprocess clip --action-limit 1 or tanh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model, torch.zeros(1, 1413), args.output, opset_version=17,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}})
    parity = {"checked": False, "reason": "onnxruntime is not installed"}
    try:
        import onnxruntime as ort
    except ImportError:
        pass
    else:
        probe = torch.linspace(-1.0, 1.0, 3 * 1413).reshape(3, 1413)
        with torch.no_grad():
            expected = export_model(probe).numpy()
        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        actual = session.run(["actions"], {"obs": probe.numpy()})[0]
        max_abs_error = float(abs(expected - actual).max())
        if not torch.allclose(torch.from_numpy(actual), torch.from_numpy(expected), rtol=1e-4, atol=1e-5):
            raise RuntimeError(f"Torch/ONNX parity failed: max_abs_error={max_abs_error:.3e}")
        parity = {"checked": True, "max_abs_error": max_abs_error}
    sidecar = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_sha256": sha256(args.output),
        "input": {"name": "obs", "shape": ["batch", 1413]},
        "output": {"name": "actions", "shape": ["batch", 16]},
        "opset": 17,
        "empirical_normalization": False,
        "action_postprocess": args.action_postprocess,
        "action_range": [
            -args.action_limit, args.action_limit
        ] if args.action_postprocess == "clip" else [-1.0, 1.0],
        "action_contract": "a_norm",
        "action_contract_adapter": args.action_postprocess != "none",
        "torch_onnx_parity": parity,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
