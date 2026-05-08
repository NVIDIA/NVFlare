# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Medical VLM dataset bridge for the Auto-FL NVFlare harness.

The canonical loaders and Qwen3-VL collator live in the sibling
``VLM_Benchmark`` checkout.  This module keeps the NVFlare harness thin: each
simulated site maps to one registered VQA dataset, and validation uses the same
registry semantics as Phase 5.1.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


DEFAULT_VLM_REPO_ROOT = "/workspace/VLM_Benchmark"
DEFAULT_SITE_DATASETS = "vqa-rad,slake,path-vqa"


def _add_vlm_repo_to_path(vlm_repo_root: str | Path) -> Path:
    root = Path(vlm_repo_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"VLM repo root not found: {root}. Pass --vlm_repo_root or set VLM_BENCHMARK_ROOT."
        )
    for path in (root, root / "Phase_3.1"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root


def parse_site_datasets(site_datasets: str) -> list[str]:
    names = [item.strip() for item in site_datasets.split(",") if item.strip()]
    if not names:
        raise ValueError("--site_datasets must name at least one VLM dataset")
    return names


def dataset_for_site(site_name: str, site_datasets: str) -> str:
    names = parse_site_datasets(site_datasets)
    if not site_name.startswith("site-"):
        raise ValueError(f"Expected NVFlare site name like site-1, got {site_name!r}")
    try:
        site_index = int(site_name.rsplit("-", 1)[1]) - 1
    except ValueError as exc:
        raise ValueError(f"Could not parse site index from {site_name!r}") from exc
    if site_index < 0 or site_index >= len(names):
        raise ValueError(f"{site_name} is outside --site_datasets={site_datasets!r}")
    return names[site_index]


def _load_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, (str, Path)):
        return Image.open(image_obj).convert("RGB")
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    raise TypeError(f"Unsupported image object type: {type(image_obj)!r}")


class MedicalVQAValidationDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        *,
        vlm_repo_root: str | Path,
        hf_cache_dir: str | None,
        seed: int,
        max_samples: int = -1,
    ) -> None:
        _add_vlm_repo_to_path(vlm_repo_root)
        from src.common import get_dataset_config, load_registered_validation_dataset

        cfg = get_dataset_config(dataset_name)
        ds, _ = load_registered_validation_dataset(dataset_name, data_path=hf_cache_dir)
        keep = set(cfg["keep_columns"])
        ds = ds.remove_columns([col for col in ds.column_names if col not in keep])
        if max_samples > 0 and max_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_samples))

        self.cfg = cfg
        self.samples: list[dict[str, Any]] = []
        for i in range(len(ds)):
            ex = ds[i]
            answers = [a["answer"] for a in ex["answers"]]
            self.samples.append(
                {
                    "dataset_name": dataset_name,
                    "image": _load_image(ex["image"]),
                    "question": ex["question"].strip(),
                    "answers": answers,
                    "gt_primary": str(ex.get("multiple_choice_answer") or answers[0]).strip(),
                    "prompt_prefix": cfg.get("prompt_prefix", "").strip(),
                    "system_message": cfg.get("system_message", "").strip(),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def create_vlm_datasets(
    site_name: str,
    *,
    vlm_repo_root: str | Path,
    site_datasets: str,
    hf_cache_dir: str | None,
    seed: int,
    max_samples_per_site: int = -1,
    max_eval_samples: int = -1,
    reserve_validation_from_train: bool = True,
):
    _add_vlm_repo_to_path(vlm_repo_root)
    from finetune_qwen3vl_medvqa import MedicalVQABenchmarkDataset

    dataset_name = dataset_for_site(site_name, site_datasets)
    train_dataset = MedicalVQABenchmarkDataset(
        dataset_names=[dataset_name],
        cache_dir=hf_cache_dir,
        seed=seed,
        max_samples_per_dataset=max_samples_per_site,
        mix_strategy="concat",
        balanced_target_size=0,
        reserve_validation_from_train=reserve_validation_from_train,
    )
    valid_dataset = MedicalVQAValidationDataset(
        dataset_name,
        vlm_repo_root=vlm_repo_root,
        hf_cache_dir=hf_cache_dir,
        seed=seed,
        max_samples=max_eval_samples,
    )
    return train_dataset, valid_dataset, dataset_name


def create_vlm_train_collator(processor, *, vlm_repo_root: str | Path):
    _add_vlm_repo_to_path(vlm_repo_root)
    from finetune_qwen3vl_medvqa import QwenMedVQACollator

    return QwenMedVQACollator(processor=processor)


def shuffled_indices(length: int, seed: int) -> list[int]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices
