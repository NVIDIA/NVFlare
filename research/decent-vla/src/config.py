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

"""VLA run configuration, YAML loading and validation."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.support.lora import LoRAConfig


@dataclass
class OptimConfig:
    """Optimizer + LR schedule.

    `scheduler: "cosine"` means lerobot-recipe warmup-cosine: linear ramp
    over `warmup_steps` (0 = no ramp) to the peak `lr`, cosine decay down
    to `decay_lr` at `decay_steps`, then constant at `decay_lr`.
    `decay_lr: None` decays all the way to 0.
    See `support/optim.py::build_warmup_cosine`.
    """

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    scheduler: str | None = None
    warmup_steps: int = 0
    decay_steps: int = 30000
    decay_lr: float | None = None
    grad_clip_norm: float | None = None  # max grad-norm (None/0 = off; lerobot smolvla/pi05 default 10)


@dataclass
class BackendConfig:
    name: str  # "mock", "lerobot", "groot"
    pretrained_path: str | None = None
    lora: LoRAConfig | None = None
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainable_params: list[str] | None = None  # AXIS 1 (local training scope): fnmatch patterns —
    # matching params requires_grad=True, rest frozen.
    # Applied after model build + LoRA/native tune flags
    # (final override). None = policy defaults decide.
    shared_params: list[str] | None = None  # AXIS 2 (federated scope): fnmatch patterns over
    # TRAINABLE param names: only matching params are
    # federated ("shared" scope); the rest are personal
    # (client-local across rounds — pair with
    # fl.persist_client_state for cold runtimes).
    # None shares all trainable parameters.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset + partitioning spec."""

    repo_id: str | None = None
    batch_size: int = 8
    num_workers: int = 0
    partitioning: str = "iid"  # "iid" | "task" | "explicit_tasks" | "explicit_repos"
    num_clients: int = 1
    held_out: str | None = None  # path to held-out task spec
    seed: int = 42
    client_repo_ids: list[list[str]] | None = None  # explicit_repos: per-client repo lists
    client_task_indices: list[list[int]] | None = None  # explicit_tasks: per-client task indices
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    name: str = "fedavg"  # algorithm/benchmark name
    mu: float = 0.0  # FedProx coefficient (ignored if name != "fedprox")
    server_lr: float = 1.0  # FedAdam server-side LR
    server_betas: tuple[float, float] = (0.9, 0.999)
    server_eps: float = 1e-3
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NVFlareConfig:
    """Site/workflow settings; simulator resources are nvflare CLI options."""

    site_prefix: str = ""  # "" = sites named c0..cN-1 (what the simulator env uses);
    # "site-" maps a conventionally-provisioned site-{i+1} to c{i}
    launch_external_process: bool = False  # run the client script in a subprocess (CellPipe) instead of
    # in-process — for training loops needing their own process
    wait_time_after_min_received: int = 10  # with fl.min_clients below the clients selected: once that many
    # have answered, seconds to wait for the rest before the round
    # closes (FLARE's ScatterAndGather default). The round closes
    # sooner if every client answers.
    memory_gc_rounds: int = 1  # native workflow garbage-collection cadence (0 = never)


@dataclass
class FLConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    num_rounds: int = 10
    local_steps: int = 50
    nvflare: NVFlareConfig = field(default_factory=NVFlareConfig)
    min_clients: int | None = None  # min client updates per round (default: all)
    round_timeout_s: float | None = None  # per-round timeout
    save_every_n_rounds: int | None = None
    output_dir: str = "outputs/fl_run"
    persist_client_state: bool = False  # opt-in: save/restore per-client state for warm resume
    client_state_keep: int | None = None  # retain only the N highest-round client-state files per
    # client (None = keep all; minimum 2). Set to 2 for 3B-scale runs:
    # a personalization client blob is ~35 GB.
    # Shared wire.py resolves a scalar dtype or {broadcast: fp32, delta: fp16}.
    wire_dtype: str | dict[str, str] = "fp16"


@dataclass
class BenchmarkConfig:
    name: str = "libero"
    suites: list[str] = field(default_factory=lambda: ["libero_spatial"])
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    n_episodes_per_task: int = 5
    seed: int = 42
    output_dir: str | None = None
    personalized: bool = False  # partial-federation arms: evaluate each client's
    # model (server shared + its personal params) on its
    # OWN tasks; needs the training run's client_states
    # + partition manifest (fl.persist_client_state).
    n_action_steps: int | None = None  # rollout action horizon = the REPLANNING interval.
    # Chunked VLA policies predict chunk_size actions at
    # once and refill only when the queue empties, so this
    # is how often the policy looks at the world. SmolVLA
    # and pi0.5 both DEFAULT it to 50 (= the whole chunk,
    # fully open loop) while lerobot-eval passes 10. On one
    # identical checkpoint that was worth 33 points
    # (70.0% at 10 vs 37.0% at 50, 400 episodes). None
    # leaves the policy untouched.


@dataclass
class RunConfig:
    """Policy, data and native FLARE workflow configuration."""

    name: str
    backend: BackendConfig
    data: DataConfig = field(default_factory=DataConfig)
    fl: FLConfig | None = None
    eval: EvalConfig | None = None
    seed: int = 42
    device: str = "cpu"


class ConfigError(ValueError):
    """Raised when a config violates a structural rule."""


def _reject_unknown(d: dict, known: set[str], where: str) -> None:
    """Raise on keys `where` does not define.

    Every block is parsed field-by-field, so an unmatched key is silently
    dropped and is then indistinguishable from one that was never written.
    That catches costly no-ops such as `fl.save_every_n_round` (no trailing s)
    before training starts. Blocks with a nested sub-block (`fl.strategy`, `fl.nvflare`,
    `backend.lora`, `backend.optim`, `eval.benchmark`) name it as a field of
    their own dataclass, so it is `known` here and validated by its own
    builder — never double-counted as unknown at the parent level.
    """
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {sorted(unknown)}; expected {sorted(known)}")


def _fields_of(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


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
    if not isinstance(d.get("partitioning", "iid"), str):
        raise ConfigError("data.partitioning must be a scheme name; put its settings directly under data")
    if d.get("partitioning") == "explicit_repos" and isinstance(d.get("client_repo_ids"), list):
        d.setdefault("num_clients", len(d["client_repo_ids"]))
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
    _reject_unknown(d, _fields_of(FLConfig), "fl")
    strategy = _build_strategy(d.get("strategy"))
    if strategy.name not in ("fedavg", "fedprox", "fedadam"):
        raise ConfigError(f"Unsupported native FLARE algorithm: {strategy.name}")
    if int(d.get("num_rounds", 10)) <= 0 or int(d.get("local_steps", 50)) <= 0:
        raise ConfigError("fl.num_rounds and fl.local_steps must be positive")
    # Fail fast on a bad wire_dtype: otherwise a typo surfaces only at the first
    # broadcast, minutes into a run that has already loaded a 14 GB model.
    from src.support import wire

    try:
        wire.resolve(d.get("wire_dtype", "fp16"))
    except ValueError as e:
        raise ConfigError(str(e)) from e
    return FLConfig(**{**d, "strategy": strategy, "nvflare": _build_nvflare(d.get("nvflare"))})


def _build_benchmark(d: dict | None) -> BenchmarkConfig:
    if not d:
        return BenchmarkConfig()
    _reject_unknown(d, _fields_of(BenchmarkConfig), "eval.benchmark")
    return BenchmarkConfig(
        name=d.get("name", "libero"),
        suites=list(d.get("suites", ["libero_spatial"])),
        extra=d.get("extra", {}),
    )


def _build_eval(d: dict | None) -> EvalConfig | None:
    if d is None:
        return None
    _reject_unknown(d, _fields_of(EvalConfig), "eval")
    return EvalConfig(**{**d, "benchmark": _build_benchmark(d.get("benchmark"))})


def validate_run_config(cfg: RunConfig) -> None:
    """Apply the config validation rules. Raises ConfigError on violation."""
    if cfg.fl is None:
        raise ConfigError("A native FLARE `fl` block is required.")

    if cfg.fl.min_clients is not None and not 1 <= cfg.fl.min_clients <= cfg.data.num_clients:
        raise ConfigError("fl.min_clients must be between 1 and data.num_clients")
    if cfg.fl.client_state_keep is not None and cfg.fl.client_state_keep < 2:
        raise ConfigError(
            f"fl.client_state_keep={cfg.fl.client_state_keep} is too low: keep at least 2 client-state "
            "files, so the client state of the last round the server saved survives a crash."
        )

    if cfg.backend.name == "lerobot" and cfg.backend.lora is not None and cfg.backend.lora.attach_to:
        raise ConfigError(
            "backend.lora.attach_to is not supported by the lerobot backend, which chooses the LoRA "
            "subtree itself; remove it."
        )

    if cfg.data.partitioning == "explicit_repos":
        repos = cfg.data.client_repo_ids
        if (
            not isinstance(repos, list)
            or not repos
            or any(
                not isinstance(group, list)
                or not group
                or any(not isinstance(repo, str) or not repo.strip() for repo in group)
                for group in repos
            )
        ):
            raise ConfigError(
                "data.partitioning=explicit_repos requires data.client_repo_ids as a non-empty list "
                "of non-empty repository-ID lists, one per client; for example "
                "[[org/client0-data], [org/client1-data]]. See README.md: Explicit repository assignment."
            )
        if cfg.backend.name != "lerobot":
            raise ConfigError("data.partitioning=explicit_repos requires backend.name=lerobot")
        if cfg.eval is not None and cfg.eval.personalized:
            raise ConfigError(
                "eval.personalized=true is unsupported with data.partitioning=explicit_repos: "
                "its manifest records repository IDs rather than client task names. Set "
                "eval.personalized=false to evaluate the global checkpoint, or use task_round_robin "
                "or explicit_tasks partitioning for personalized evaluation. See README.md: "
                "Explicit repository assignment."
            )
        if len(cfg.data.client_repo_ids) != cfg.data.num_clients:
            raise ConfigError(
                f"data.client_repo_ids has {len(cfg.data.client_repo_ids)} repo lists but "
                f"data.num_clients is {cfg.data.num_clients}; explicit_repos needs one list per client."
            )

    if cfg.backend.shared_params is not None and not cfg.fl.persist_client_state:
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
    _reject_unknown(raw, _fields_of(RunConfig), "run")

    cfg = RunConfig(
        name=raw["name"],
        backend=_build_backend(raw["backend"]),
        data=_build_data(raw.get("data")),
        fl=_build_fl(raw.get("fl")),
        eval=_build_eval(raw.get("eval")),
        seed=raw.get("seed", 42),
        device=raw.get("device", "cpu"),
    )
    validate_run_config(cfg)
    return cfg
