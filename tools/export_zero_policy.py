#!/usr/bin/env python3
"""Generate a dependency-free ONNX placeholder: [1,441] -> zeros [1,16]."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def _varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(number: int, wire: int, payload: bytes) -> bytes:
    key = _varint((number << 3) | wire)
    return key + (_varint(len(payload)) if wire == 2 else b"") + payload


def _int_field(number: int, value: int) -> bytes:
    return _field(number, 0, _varint(value))


def _str_field(number: int, value: str) -> bytes:
    return _field(number, 2, value.encode())


def _shape(*dims: int) -> bytes:
    dimension_messages = [_int_field(1, dim) for dim in dims]
    shape = b"".join(_field(1, 2, dim) for dim in dimension_messages)
    tensor_type = _int_field(1, 1) + _field(2, 2, shape)  # FLOAT + shape
    return _field(1, 2, tensor_type)


def _value_info(name: str, *dims: int) -> bytes:
    return _str_field(1, name) + _field(2, 2, _shape(*dims))


def build_model() -> bytes:
    tensor = (
        _int_field(1, 1)
        + _int_field(1, 16)
        + _int_field(2, 1)
        + _str_field(8, "zero_actions")
        + _field(9, 2, struct.pack("<16f", *([0.0] * 16)))
    )
    attribute = _str_field(1, "value") + _field(5, 2, tensor) + _int_field(20, 4)
    node = (
        _str_field(2, "actions")
        + _str_field(3, "constant_zero_actions")
        + _str_field(4, "Constant")
        + _field(5, 2, attribute)
    )
    graph = (
        _field(1, 2, node)
        + _str_field(2, "terrain_policy_placeholder")
        + _field(11, 2, _value_info("obs", 1, 441))
        + _field(12, 2, _value_info("actions", 1, 16))
    )
    opset = _int_field(2, 17)
    return (
        _int_field(1, 9)
        + _str_field(2, "s10-terrain-aware-policy")
        + _str_field(3, "0.1.0")
        + _field(7, 2, graph)
        + _field(8, 2, opset)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="models/terrain_locomotion.onnx")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(build_model())
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
