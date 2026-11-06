#!/usr/bin/env python3
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

"""Fail-fast, read-only checks for a self-contained NVIDIA FLARE run."""

from __future__ import annotations

import ast
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

from fabric_common import EXPERIMENTS, check_environment, load_setup

REQUIRED_FILES = {
    "fabric_client.py",
    "fabric_aggregator.py",
    "fabric_common.py",
    "fabric_progress.py",
    "fabric_runtime.py",
    "fabric_resume.py",
    "run_fabric.py",
    "core/mil_models.py",
    "core/topk_mil.py",
    "core/aggregation.py",
    "core/training.py",
    "core/client_training.py",
    "core/experiment_utils.py",
}

CRITICAL_DEFINITIONS = {
    "mil_models.py": {"DTFDMIL", "MILModel", "build_mil_model"},
    "topk_mil.py": {"DTFDTopKMIL", "forward_topk_bags"},
    "aggregation.py": {"copy_state_dict", "fedavg_state_dict"},
    "training.py": {
        "PatientFeatureDataset",
        "build_model",
        "build_criterion",
        "compute_metrics",
        "evaluate_model",
        "forward_bags",
        "make_loader",
        "read_patient_bag",
    },
    "client_training.py": {"train_local_client_final"},
    "experiment_utils.py": {
        "condition_for",
        "install_hdf5_safe_reader",
        "normalized_result_row",
        "refresh_summaries",
        "safe_read_feature_array",
    },
}


def _class_constructor_args(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    return {arg.arg for arg in (*constructor.args.args, *constructor.args.kwonlyargs)}


def _check_nvflare_api() -> None:
    package_root = Path(importlib.metadata.distribution("nvflare").locate_file(""))
    checks = (
        (
            package_root / "nvflare/app_opt/pt/recipes/fedavg.py",
            "FedAvgRecipe",
            {
                "model",
                "initial_ckpt",
                "min_clients",
                "num_rounds",
                "train_script",
                "train_args",
                "aggregator",
                "launch_external_process",
                "launch_once",
                "command",
                "server_expected_format",
                "params_transfer_type",
                "per_site_config",
                "key_metric",
                "stop_cond",
                "patience",
            },
        ),
        (
            package_root / "nvflare/recipe/sim_env.py",
            "SimEnv",
            {"clients", "num_threads", "workspace_root"},
        ),
    )
    for path, class_name, required in checks:
        if not path.is_file():
            raise FileNotFoundError(f"Installed NVIDIA FLARE source not found: {path}")
        names = _class_constructor_args(path, class_name)
        if not required.issubset(names):
            raise ValueError(f"Installed {class_name} API lacks {sorted(required - names)}")
    aggregator = package_root / "nvflare/app_common/aggregators/model_aggregator.py"
    source = aggregator.read_text()
    for method in ("accept_model", "aggregate_model", "reset_stats"):
        if f"def {method}(" not in source:
            raise ValueError(f"Installed ModelAggregator lacks {method}")


def perform_checks(root: Path) -> dict:
    """Check code/config/API completeness without reading data or starting training."""
    root = root.resolve()
    code = root / "fabric"
    check_environment(root)
    for experiment in EXPERIMENTS:
        load_setup(root, experiment)

    missing = sorted(relative for relative in REQUIRED_FILES if not (code / relative).is_file())
    if missing:
        raise FileNotFoundError(f"Required runtime files missing: {missing}")
    python_files = sorted(code.rglob("*.py"))
    trees = {}
    for path in python_files:
        if path.is_symlink():
            raise ValueError(f"Runtime source must be an independent file, not a symlink: {path}")
        trees[path] = ast.parse(path.read_text(), filename=str(path))

    core_dir = code / "core"
    core_modules = {path.stem for path in core_dir.glob("*.py")}
    for path, tree in trees.items():
        if path.parent != core_dir:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                dependencies = [node.module] if node.module else [alias.name for alias in node.names]
                for module in dependencies:
                    if module not in core_modules:
                        raise ValueError(f"Missing runtime dependency: {path.name} -> {module}")

    for filename, required in CRITICAL_DEFINITIONS.items():
        tree = trees[core_dir / filename]
        names = {
            node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not required.issubset(names):
            raise ValueError(f"Missing required definitions in {filename}: {sorted(required - names)}")

    _check_nvflare_api()
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "static_checks_passed",
        "python_files_parsed": len(python_files),
        "required_runtime_files": len(REQUIRED_FILES),
        "nvflare_api": "2.7.2 source signatures checked",
    }
