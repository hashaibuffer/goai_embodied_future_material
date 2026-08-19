#!/usr/bin/env python3
from __future__ import annotations

import argparse

from schema import validate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    count = validate_dataset(args.dataset)
    print(f"valid TD dataset: {count} samples")


if __name__ == "__main__":
    main()
