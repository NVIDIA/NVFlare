# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the exact datasets used for the reported MemCo experiments."""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _item_count(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return len(payload)


def _validate(path: Path, expected_count: int, expected_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = _item_count(path)
    if count != expected_count:
        raise ValueError(f"{path}: expected {expected_count} items, found {count}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"{path}: SHA-256 mismatch: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--family", choices=("pddl", "fever", "alfworld", "all"), default="all")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "datasets.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    families = manifest if args.family == "all" else {args.family: manifest[args.family]}
    checked = 0
    for family, config in families.items():
        base = args.data_root
        if family == "alfworld":
            base /= config["subset_dir"]
            game_root = args.data_root / config["game_root"]
            if not game_root.is_dir():
                raise FileNotFoundError(game_root)
        for relative_path, (count, digest) in config["files"].items():
            _validate(base / relative_path, count, digest)
            checked += 1
        print(f"{family}: OK ({len(config['files'])} files)")
    print(f"Validated {checked} canonical dataset files.")


if __name__ == "__main__":
    main()
