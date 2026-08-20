"""Small 441->16 student policies for TE distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .te_dataset import ACTION_DIM, OBS_DIM, PROPRIO_DIM, InputStatistics


@dataclass(frozen=True)
class StudentConfig:
    controller: str = "learned"
    architecture: str = "branch"
    proprio_hidden: tuple[int, ...] = (128, 64)
    terrain_hidden: tuple[int, ...] = (256, 128)
    fusion_hidden: tuple[int, ...] = (128,)
    flat_hidden: tuple[int, ...] = (256, 128)
    activation: str = "elu"
    normalization_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.controller not in {"learned", "proprio_clone"}:
            raise ValueError("controller must be learned or proprio_clone")
        if self.architecture not in {"branch", "flat"}:
            raise ValueError("architecture must be branch or flat")
        if self.normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")

    def to_dict(self) -> dict:
        value = asdict(self)
        for name in ("proprio_hidden", "terrain_hidden", "fusion_hidden", "flat_hidden"):
            value[name] = list(value[name])
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "StudentConfig":
        value = dict(value)
        for name in ("proprio_hidden", "terrain_hidden", "fusion_hidden", "flat_hidden"):
            if name in value:
                value[name] = tuple(map(int, value[name]))
        return cls(**value)


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


def _mlp(input_dim: int, hidden: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, int(width)), _activation(activation)))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class StudentPolicy(nn.Module):
    """Raw-action student with normalization embedded in the exported graph."""

    def __init__(self, config: StudentConfig, statistics: InputStatistics):
        super().__init__()
        self.config = config
        self.register_buffer(
            "input_mean", torch.as_tensor(statistics.mean, dtype=torch.float32))
        self.register_buffer(
            "input_std", torch.as_tensor(statistics.std, dtype=torch.float32))
        normalized_mask = np.asarray(statistics.normalized_mask, dtype=np.bool_)
        self.register_buffer(
            "input_normalized",
            torch.as_tensor(normalized_mask, dtype=torch.bool),
            persistent=False,
        )
        # A feature that was exactly constant in all training data has no
        # learnable response.  Feeding a novel value through its tiny std would
        # amplify an untrained random weight (for example 0.3 / 1e-4 = 3000).
        # Keep this derived mask out of checkpoints so older format-v1 payloads
        # remain loadable; validity bits are raw and always active.
        active = (~normalized_mask) | (
            statistics.std > np.float32(1.0001e-4))
        self.register_buffer(
            "input_active",
            torch.as_tensor(active, dtype=torch.float32),
            persistent=False,
        )

        if config.controller == "proprio_clone":
            self.proprio_clone = _mlp(
                PROPRIO_DIM, config.proprio_hidden, ACTION_DIM, config.activation)
        elif config.architecture == "flat":
            self.flat_policy = _mlp(
                OBS_DIM, config.flat_hidden, ACTION_DIM, config.activation)
        else:
            proprio_out = config.proprio_hidden[-1]
            terrain_out = config.terrain_hidden[-1]
            self.proprio_encoder = _mlp(
                PROPRIO_DIM, config.proprio_hidden[:-1], proprio_out, config.activation)
            self.terrain_encoder = _mlp(
                OBS_DIM - PROPRIO_DIM,
                config.terrain_hidden[:-1],
                terrain_out,
                config.activation,
            )
            self.fusion = _mlp(
                proprio_out + terrain_out,
                config.fusion_hidden,
                ACTION_DIM,
                config.activation,
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if (not torch.jit.is_tracing()
                and (observations.ndim != 2 or observations.shape[-1] != OBS_DIM)):
            raise ValueError(f"student input must be [batch,{OBS_DIM}]")
        normalized = (observations - self.input_mean) / self.input_std
        prepared = torch.where(
            self.input_normalized, normalized, observations)
        normalized = torch.clamp(
            prepared * self.input_active,
            min=-self.config.normalization_clip,
            max=self.config.normalization_clip,
        )
        if self.config.controller == "proprio_clone":
            # The ONNX signature remains 441-D, but terrain can never influence
            # this separately trained fallback model.
            return self.proprio_clone(normalized[:, :PROPRIO_DIM])
        if self.config.architecture == "flat":
            return self.flat_policy(normalized)
        proprio = self.proprio_encoder(normalized[:, :PROPRIO_DIM])
        terrain = self.terrain_encoder(normalized[:, PROPRIO_DIM:])
        return self.fusion(torch.cat((proprio, terrain), dim=-1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(
    path: Path,
    *,
    model: StudentPolicy,
    statistics: InputStatistics,
    training_metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "input_statistics": statistics.to_json(),
        "model_state_dict": model.state_dict(),
        "training_metadata": dict(training_metadata),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu"):
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported student checkpoint format")
    config = StudentConfig.from_dict(payload["model_config"])
    statistics = InputStatistics.from_json(payload["input_statistics"])
    model = StudentPolicy(config, statistics)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, statistics, dict(payload.get("training_metadata", {}))


def statistics_from_model(model: StudentPolicy) -> InputStatistics:
    mean = model.input_mean.detach().cpu().numpy().astype(np.float32)
    std = model.input_std.detach().cpu().numpy().astype(np.float32)
    mask = model.input_normalized.detach().cpu().numpy().astype(np.bool_)
    return InputStatistics(mean=mean, std=std, normalized_mask=mask)
