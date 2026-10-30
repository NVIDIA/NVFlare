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

"""YAML <-> RunConfig conversion with explicit validation.

This module owns the config validation rules:
  * Exactly one of `train` / `fl` is set, unless `eval.only=True`.
  * Settings the code accepts but does not apply are rejected (`aggregate_lora_only`,
    `client_state_keep` below 2, `attach_to` on the lerobot backend, `data.revision`).
  * Suite names valid for the chosen benchmark (deferred to benchmark
    `from_config`; we only validate the structural rules here).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml

from decent_vla.configs.schemas import (
    BackendConfig,
    BenchmarkConfig,
    DataConfig,
    EvalConfig,
    FLConfig,
    NVFlareConfig,
    OptimConfig,
    RunConfig,
    StrategyConfig,
    TrainConfig,
    WandbConfig,
)
from decent_vla.core.lora import LoRAConfig


class ConfigError(ValueError):
    """Raised when a config violates a structural rule."""


def _reject_unknown(d: dict, known: set[str], where: str) -> None:
    """Raise on keys `where` does not define.

    Every block is parsed field-by-field, so an unmatched key is silently
    dropped and is then indistinguishable from one that was never written.
    That is the same failure mode for `fl.save_every_n_round` (no trailing s
    → zero checkpoints for the whole run) as for `wandb.porject`, only more
    expensive. Blocks with a nested sub-block (`fl.strategy`, `fl.nvflare`,
    `backend.lora`, `backend.optim`, `eval.benchmark`) name it as a field of
    their own dataclass, so it is `known` here and validated by its own
    builder — never double-counted as unknown at the parent level.
    """
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {sorted(unknown)}; expected {sorted(known)}")


def _fields_of(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _normalize_raw(raw: dict) -> dict:
    """Spec-§9 sugar → flat dataclass fields. Mutates and returns `raw`.

    * `data.partitioning: {scheme, num_clients, held_out, seed}` → flat keys.
    * `fl.execution` → `fl.mode`; `fl.rounds` → `fl.num_rounds`.
    * `fl.batch_size` → `data.batch_size`.
    """
    data = raw.get("data") or {}
    part = data.get("partitioning")
    if isinstance(part, dict):
        _reject_unknown(part, {"scheme", "num_clients", "held_out", "seed"}, "data.partitioning")
        data["partitioning"] = part.get("scheme", "iid")
        for key in ("num_clients", "held_out", "seed"):
            if key in part:
                data[key] = part[key]
        raw["data"] = data
    fl = raw.get("fl")
    if isinstance(fl, dict):
        if "execution" in fl:
            fl["mode"] = fl.pop("execution")
        if "rounds" in fl:
            fl["num_rounds"] = fl.pop("rounds")
        if "batch_size" in fl:
            data["batch_size"] = fl.pop("batch_size")
            raw["data"] = data
    return raw


def _build_optim(d: dict | None) -> OptimConfig:
    if not d:
        return OptimConfig()
    return OptimConfig(**d)


def _build_lora(d: dict | None) -> LoRAConfig | None:
    if d is None:
        return None
    return LoRAConfig(**d)


def _build_backend(d: dict) -> BackendConfig:
    _reject_unknown(d, _fields_of(BackendConfig), "backend")
    return BackendConfig(**{**d, "lora": _build_lora(d.get("lora")), "optim": _build_optim(d.get("optim"))})


def _build_data(d: dict | None) -> DataConfig:
    if not d:
        return DataConfig()
    # explicit_repos: one client per repo list. Derive the count when it is not given, so a
    # second list is not silently dropped by the default num_clients of 1.
    if d.get("partitioning") == "explicit_repos" and d.get("client_repo_ids") and "num_clients" not in d:
        d["num_clients"] = len(d["client_repo_ids"])
    return DataConfig(**d)


def _build_strategy(d: dict | None) -> StrategyConfig:
    if not d:
        return StrategyConfig()
    if "server_betas" in d and isinstance(d["server_betas"], list):
        d["server_betas"] = tuple(d["server_betas"])
    return StrategyConfig(**d)


def _build_nvflare(d: dict | None) -> NVFlareConfig:
    if not d:
        return NVFlareConfig()
    _reject_unknown(d, _fields_of(NVFlareConfig), "fl.nvflare")
    return NVFlareConfig(**d)


def _build_fl(d: dict | None) -> FLConfig | None:
    if d is None:
        return None
    # `_normalize_raw` has already rewritten the `execution`/`rounds`/`batch_size`
    # sugar, so what reaches here must be spelled exactly as FLConfig declares it.
    _reject_unknown(d, _fields_of(FLConfig), "fl")
    if d.get("mode", "nvflare") not in ("nvflare", "local"):
        raise ConfigError("fl.mode must be nvflare (local is a migration alias for the FLARE simulator)")
    strategy = _build_strategy(d.get("strategy"))
    if strategy.name not in ("fedavg", "fedprox", "fedadam"):
        raise ConfigError(f"Unsupported native FLARE algorithm: {strategy.name}")
    if int(d.get("num_rounds", 10)) <= 0 or int(d.get("local_steps", 50)) <= 0:
        raise ConfigError("fl.num_rounds and fl.local_steps must be positive")
    # Fail fast on a bad wire_dtype: otherwise a typo surfaces only at the first
    # broadcast, minutes into a run that has already loaded a 14 GB model.
    from decent_vla.runtime import wire

    try:
        wire.resolve(d.get("wire_dtype", "fp16"))
    except ValueError as e:
        raise ConfigError(str(e)) from e
    return FLConfig(**{**d, "strategy": strategy, "mode": "nvflare", "nvflare": _build_nvflare(d.get("nvflare"))})


_WANDB_MODES = ("online", "offline", "disabled")


def _build_wandb(d: dict | None) -> WandbConfig:
    """Parse the optional top-level `wandb:` block.

    Unknown keys are an error rather than a silent no-op: a typo like
    `porject: decent-vla` would otherwise send a 12-hour formal run's curves
    nowhere, and you would only find out after it finished.
    """
    if not d:
        return WandbConfig()
    _reject_unknown(d, _fields_of(WandbConfig), "wandb")
    mode = str(d.get("mode", "online")).lower()
    if mode not in _WANDB_MODES:
        raise ConfigError(f"wandb.mode={mode!r} invalid; use one of {list(_WANDB_MODES)}")
    tags = d.get("tags") or []
    if not isinstance(tags, (list, tuple)):
        raise ConfigError(f"wandb.tags must be a list, got {type(tags).__name__}")
    return WandbConfig(**{**d, "mode": mode, "tags": [str(x) for x in tags]})


def _build_train(d: dict | None) -> TrainConfig | None:
    if d is None:
        return None
    return TrainConfig(**d)


def _build_benchmark(d: dict | None) -> BenchmarkConfig:
    if not d:
        return BenchmarkConfig()
    _reject_unknown(d, _fields_of(BenchmarkConfig), "eval.benchmark")
    return BenchmarkConfig(
        name=d.get("name", "mock"),
        suites=list(d.get("suites", ["default"])),
        extra=d.get("extra", {}),
    )


def _build_eval(d: dict | None) -> EvalConfig | None:
    if d is None:
        return None
    _reject_unknown(d, _fields_of(EvalConfig), "eval")
    return EvalConfig(**{**d, "benchmark": _build_benchmark(d.get("benchmark"))})


def validate_run_config(cfg: RunConfig) -> None:
    """Apply the config validation rules. Raises ConfigError on violation."""
    eval_only = cfg.eval is not None and cfg.eval.only
    has_train = cfg.train is not None
    has_fl = cfg.fl is not None

    if not eval_only:
        if has_train and has_fl:
            raise ConfigError("Set exactly one of `train` or `fl`, not both.")
        if not has_train and not has_fl:
            raise ConfigError("Set either `train`, `fl`, or `eval.only: true`.")

    if has_fl and cfg.fl.min_clients is not None and not 1 <= cfg.fl.min_clients <= cfg.data.num_clients:
        raise ConfigError("fl.min_clients must be between 1 and data.num_clients")
    if has_fl and cfg.fl.strategy.aggregate_lora_only:
        raise ConfigError(
            "fl.strategy.aggregate_lora_only is not supported: the strategies aggregate every shared "
            "parameter. Restrict the federated scope with backend.shared_params instead."
        )

    if has_fl and cfg.fl.client_state_keep is not None and cfg.fl.client_state_keep < 2:
        raise ConfigError(
            f"fl.client_state_keep={cfg.fl.client_state_keep} is too low: keep at least 2 client-state "
            "files, so the client state of the last round the server saved survives a crash."
        )

    if cfg.backend.name == "lerobot" and cfg.backend.lora is not None and cfg.backend.lora.attach_to:
        raise ConfigError(
            "backend.lora.attach_to is not supported by the lerobot backend, which chooses the LoRA "
            "subtree itself; remove it."
        )

    if cfg.data.revision is not None:
        raise ConfigError(
            "data.revision is not supported: no backend applies it to dataset loading. Remove it, "
            "and pin the dataset revision in the backend's own data config."
        )

    if cfg.data.partitioning == "explicit_repos" and cfg.data.client_repo_ids is not None:
        if len(cfg.data.client_repo_ids) != cfg.data.num_clients:
            raise ConfigError(
                f"data.client_repo_ids has {len(cfg.data.client_repo_ids)} repo lists but "
                f"data.num_clients is {cfg.data.num_clients}; explicit_repos needs one list per client."
            )

    if cfg.eval is not None and cfg.eval.every_n_rounds is not None:
        raise ConfigError("eval.every_n_rounds is deferred to v1.1+ — leave it unset for v0.1.")

    if has_fl and cfg.backend.shared_params is not None and not cfg.fl.persist_client_state:
        # Warm in-process state carries the personal params within one job, but
        # any process restart / chained resubmit silently resets them to
        # pretrained — a personalization run without persistence is usually a
        # mistake, not always (hence warn, not raise).
        logging.getLogger(__name__).warning(
            "backend.shared_params is set but fl.persist_client_state is false: "
            "personal (non-federated) params will NOT survive process restarts "
            "or chained-job resumes. Set fl.persist_client_state: true for "
            "personalization runs."
        )

    if cfg.eval is not None and not cfg.eval.eval_all_tasks:
        raise ConfigError("eval_all_tasks=false is not supported in v0.2; remove it or set true.")


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level config must be a YAML mapping.")
    if "name" not in raw:
        raise ConfigError(f"{path}: missing required field `name`.")
    if "backend" not in raw:
        raise ConfigError(f"{path}: missing required field `backend`.")
    raw = _normalize_raw(raw)

    cfg = RunConfig(
        name=raw["name"],
        backend=_build_backend(raw["backend"]),
        data=_build_data(raw.get("data")),
        train=_build_train(raw.get("train")),
        fl=_build_fl(raw.get("fl")),
        eval=_build_eval(raw.get("eval")),
        wandb=_build_wandb(raw.get("wandb")),
        seed=raw.get("seed", 42),
        device=raw.get("device", "cpu"),
    )
    validate_run_config(cfg)
    return cfg


def dump_run_config(cfg: RunConfig) -> dict[str, Any]:
    """Return a plain dict for hashing / logging."""
    return dataclasses.asdict(cfg)
