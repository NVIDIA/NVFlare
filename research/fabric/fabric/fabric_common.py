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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Portable configuration, data validation, and FLARE tensor guards."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SITES = ("CBTN_CQU", "Harvard", "EBRAINS")
EXPERIMENTS = ("uni", "virchow2")
VARIANTS = ("pooling", "topk")
LOCKED_SETTINGS = {
    "n_folds": 5,
    "rounds": 5,
    "local_epochs": 5,
    "batch_size": 32,
    "num_workers": 1,
    "max_instances": 4000,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "threshold": 0.5,
    "seed": 42,
    "device": "cuda",
    "gpu": 0,
    "amp": True,
    "embed_dim": 512,
    "attn_dim": 256,
    "dropout": 0.25,
    "dtfd_pseudo_bags": 8,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root: Path, path: str | Path) -> Path:
    """Reject output/runtime paths (including symlinks) escaping the repository."""
    root = root.resolve()
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes {root}: {candidate}")
    return candidate


def external_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve an explicitly configured data path; data may live outside Git."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def load_data_config(repo_root: Path, path: Path) -> dict:
    path = external_path(repo_root, path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data configuration not found: {path}. Copy configs/data_paths.example.json "
            "to configs/data_paths.local.json and edit only the local copy."
        )
    payload = read_json(path)
    if "manifest_root" not in payload:
        raise ValueError("Data configuration requires manifest_root")
    mapping = payload.get("feature_path_prefix_map", {})
    if not isinstance(mapping, dict) or any(not old or not new for old, new in mapping.items()):
        raise ValueError("feature_path_prefix_map must contain non-empty old:new strings")
    return {
        "config_path": str(path),
        "manifest_root": str(external_path(repo_root, payload["manifest_root"])),
        "feature_path_prefix_map": {str(old): str(external_path(repo_root, new)) for old, new in mapping.items()},
    }


def remap_feature_value(value: str, prefix_map: dict[str, str]) -> str:
    paths = []
    for raw in str(value).split(";"):
        replacement = raw
        for old in sorted(prefix_map, key=len, reverse=True):
            normalized = old.rstrip("/")
            if raw == normalized or raw.startswith(normalized + "/"):
                replacement = prefix_map[old].rstrip("/") + raw[len(normalized) :]
                break
        paths.append(replacement)
    return ";".join(paths)


def remap_feature_paths(frame, prefix_map: dict[str, str]):
    """Return a copy with path prefixes changed; never reorder rows or slides."""
    frame = frame.copy()
    if prefix_map:
        frame["feature_paths"] = frame["feature_paths"].map(lambda value: remap_feature_value(value, prefix_map))
    return frame


def load_setup(root: Path, experiment: str, variant: str = "pooling", top_k: int | None = None) -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    if top_k is not None and (variant != "topk" or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k is a positive integer and is only valid for the topk variant")
    setup = read_json(inside(root, f"configs/{experiment}.json"))
    expected = {
        "experiment": experiment,
        "model_key": f"{experiment}_dtfd",
        "encoder": {"uni": "UNI", "virchow2": "Virchow2"}[experiment],
        "model_name": "dtfd_mil",
        "input_dim": {"uni": 1024, "virchow2": 2560}[experiment],
        "sites": list(SITES),
        "folds": list(range(5)),
        "nvflare_version": "2.7.2",
    }
    for key, value in expected.items():
        if setup.get(key) != value:
            raise ValueError(f"Locked experiment field changed: {key}")
    if setup.get("settings") != LOCKED_SETTINGS:
        raise ValueError("Historical training settings changed")
    if variant == "topk":
        setup.update(
            variant="topk",
            model_name="dtfd_topk",
            model_key=f"{experiment}_dtfd_topk",
            model_options={
                "top_k": top_k if top_k is not None else 1,
                "pseudo_loss_weight": 1.0,
                "eval_group_seed": setup["settings"]["seed"],
            },
        )
    return setup


def _manifest_paths(manifest_root: Path, experiment: str, fold: int) -> list[Path]:
    folder = manifest_root / experiment / f"fold_{fold}"
    return [folder / f"client_{site}_train_manifest.csv" for site in SITES] + [folder / "global_test_manifest.csv"]


def check_inputs(setup: dict, manifest_root: Path, prefix_map: dict[str, str], *, check_features=True) -> dict:
    """Validate private manifests and optionally check feature shapes before training."""
    manifest_root = manifest_root.resolve()
    test_patients, cohort, counts, feature_files = set(), None, [], set()
    required = {"patient_id", "site", "recurrence_label", "feature_paths"}
    for fold in setup["folds"]:
        seen, fold_counts = set(), {}
        for source in _manifest_paths(manifest_root, setup["experiment"], fold):
            if not source.is_file():
                raise FileNotFoundError(source)
            with source.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or not required.issubset(rows[0]):
                raise ValueError(f"Manifest is empty or missing columns: {source}")
            ids = [row["patient_id"] for row in rows]
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError(f"Duplicate/overlapping patients: fold={fold}, file={source.name}")
            if any(row["recurrence_label"] not in {"0", "1"} for row in rows):
                raise ValueError(f"Expected binary recurrence labels: {source}")
            is_test = source.name == "global_test_manifest.csv"
            site = "test" if is_test else source.name.removeprefix("client_").removesuffix("_train_manifest.csv")
            if not is_test and (site not in SITES or any(row["site"] != site for row in rows)):
                raise ValueError(f"Client site mismatch: {source}")
            seen.update(ids)
            fold_counts[site] = len(rows)
            if is_test:
                if test_patients.intersection(ids):
                    raise ValueError("A patient occurs in more than one test fold")
                test_patients.update(ids)
            for row in rows:
                for raw in remap_feature_value(row["feature_paths"], prefix_map).split(";"):
                    feature_files.add(Path(raw))
        if len(seen) != 804 or (cohort is not None and seen != cohort):
            raise ValueError("Reproducing the reported experiments requires the 804-patient cohort")
        cohort = seen
        counts.append({"fold": fold, **fold_counts})
    if test_patients != cohort:
        raise ValueError("The five test folds must cover the cohort exactly once")
    total_bytes = 0
    if check_features:
        from core.training import read_feature_shape

        for path in feature_files:
            if not path.is_file():
                raise FileNotFoundError(path)
            _, width = read_feature_shape(path)
            if width != setup["input_dim"]:
                raise ValueError(f"Feature dimension mismatch in {path}: expected {setup['input_dim']}, got {width}")
            total_bytes += path.stat().st_size
    return {
        "patients": len(test_patients),
        "fold_counts": counts,
        "feature_files": len(feature_files),
        "feature_bytes": total_bytes,
    }


def check_environment(root: Path) -> None:
    snapshot = read_json(inside(root, "configs/environment-lock.json"))
    if f"{sys.version_info.major}.{sys.version_info.minor}" != snapshot["python"]:
        raise RuntimeError(f"Expected Python {snapshot['python']}")
    for name, version in snapshot["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Expected {name}=={version}")


def load_core() -> SimpleNamespace:
    directory = Path(__file__).resolve().parent / "core"
    sys.path.insert(0, str(directory.parent))
    modules = {}
    for name in ("mil_models", "training", "aggregation", "client_training", "experiment_utils"):
        module = importlib.import_module(f"core.{name}")
        if Path(module.__file__).resolve().parent != directory.resolve():
            raise RuntimeError(f"Refusing external project-code import: {module.__file__}")
        modules[name] = module
    return SimpleNamespace(**modules)


def training_args(root: Path, setup: dict, output_dir: Path, manifest_root: Path | None = None) -> SimpleNamespace:
    settings = dict(setup["settings"])
    if setup.get("variant") == "topk":
        options = setup["model_options"]
        settings.update(
            model_variant="topk",
            dtfd_top_k=options["top_k"],
            dtfd_pseudo_loss_weight=options["pseudo_loss_weight"],
            dtfd_eval_group_seed=options["eval_group_seed"],
        )
    manifest_root = (manifest_root or root / "private_data/manifests").resolve()
    settings.update(
        output_root=str(root),
        manifest_dir=str(manifest_root / setup["experiment"]),
        out_dir=str(inside(inside(root, "results"), output_dir)),
        force=False,
    )
    return SimpleNamespace(**settings)


def make_spec(core: SimpleNamespace, root: Path, setup: dict, output_dir: Path, manifest_root: Path | None = None):
    original = core.experiment_utils.MODEL_SPECS[f"{setup['experiment']}_dtfd"]
    manifest_root = (manifest_root or root / "private_data/manifests").resolve()
    is_topk = setup.get("variant") == "topk"
    return core.experiment_utils.ModelSpec(
        key=setup["model_key"],
        display_name=(f"{original.encoder} + Top-k two-tier MIL FedAvg epoch5" if is_topk else original.display_name),
        encoder=original.encoder,
        mil_label="Top-k two-tier MIL" if is_topk else original.mil_label,
        model_name=setup["model_name"],
        source_run_dir=manifest_root / setup["experiment"],
        out_dir=output_dir,
    )


def local_seed(setup: dict, fold: int, flare_round: int, site: str) -> int:
    return int(setup["settings"]["seed"]) + fold * 1000 + (flare_round + 1) * 100 + SITES.index(site)


def checked_feature_array(path, row_indices=None):
    from core.experiment_utils import safe_read_feature_array

    values = safe_read_feature_array(path, row_indices=row_indices)
    if row_indices is not None and values.shape[0] != len(row_indices):
        raise RuntimeError(
            f"Feature row count mismatch in {path}: requested {len(row_indices)}, "
            f"read {values.shape[0]}. Refusing silent row replacement or misalignment."
        )
    return values


def install_feature_reader(core: SimpleNamespace, log_path: Path) -> None:
    core.experiment_utils.install_hdf5_safe_reader(log_path)
    core.training.read_feature_array = checked_feature_array


def cpu_state(params: dict) -> dict:
    import torch

    if not params or any(not isinstance(value, torch.Tensor) for value in params.values()):
        raise TypeError("Expected full PyTorch tensor parameters")
    result = {key: value.detach().cpu().clone() for key, value in params.items()}
    if any(torch.is_floating_point(value) and not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite model parameters")
    return result


def state_digest(params: dict) -> str:
    digest = hashlib.sha256()
    for key, value in params.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
