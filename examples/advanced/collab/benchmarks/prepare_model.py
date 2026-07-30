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

"""Download and validate a revision-pinned benchmark model in the configured user cache."""

import argparse
import json
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_snapshot(snapshot_path: Path) -> tuple[list[str], int]:
    required_files = ("config.json", "tokenizer_config.json")
    missing = [name for name in required_files if not (snapshot_path / name).is_file()]
    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as stream:
            weight_map = json.load(stream).get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        missing.extend(name for name in shard_names if not (snapshot_path / name).is_file())
    else:
        shard_names = sorted(path.name for path in snapshot_path.glob("*.safetensors"))
        if not shard_names:
            missing.append("model.safetensors or model.safetensors.index.json")
    if missing:
        raise FileNotFoundError(f"model snapshot is incomplete; missing: {', '.join(missing)}")
    total_bytes = sum((snapshot_path / name).stat().st_size for name in shard_names)
    return shard_names, total_bytes


def prepare_model(config_path: Path, local_files_only: bool) -> Path:
    config = load_config(config_path)
    model_id = config["model_name_or_path"]
    revision = config.get("model_revision")
    if not revision:
        raise ValueError("benchmark model preparation requires model_revision")

    hf_home = Path(config["hf_home"]).expanduser().resolve()
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"
    xdg_cache = hf_home / "xdg"
    for path in (hf_home, hub_cache, datasets_cache, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hub_cache),
            "HF_DATASETS_CACHE": str(datasets_cache),
            "XDG_CACHE_HOME": str(xdg_cache),
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )

    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(hub_cache),
            local_files_only=local_files_only,
        )
    ).resolve()
    if not snapshot_path.is_relative_to(hf_home):
        raise RuntimeError(f"resolved snapshot escaped configured user cache: {snapshot_path}")
    shard_names, weight_bytes = validate_snapshot(snapshot_path)

    manifest_dir = hf_home / "nvflare_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{model_id.replace('/', '--')}--{revision}.json"
    manifest = {
        "model_name_or_path": model_id,
        "requested_revision": revision,
        "resolved_revision": snapshot_path.name,
        "snapshot_path": str(snapshot_path),
        "cache_root": str(hf_home),
        "weight_shards": shard_names,
        "weight_bytes": weight_bytes,
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Prepared {model_id}@{revision} under user cache {hf_home}")
    print(f"Snapshot: {snapshot_path}")
    print(f"Manifest: {manifest_path}")
    return snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "pt_llm_sft_slurm_yi_9b.json",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    prepare_model(args.config.expanduser().resolve(), args.local_files_only)


if __name__ == "__main__":
    main()
