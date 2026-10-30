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

"""Config dataclasses. `configs/loader.py` validates the rules these encode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decent_vla.core.lora import LoRAConfig


@dataclass
class OptimConfig:
    """Optimizer + LR schedule.

    `scheduler: "cosine"` means lerobot-recipe warmup-cosine: linear ramp
    over `warmup_steps` (0 = no ramp) to the peak `lr`, cosine decay down
    to `decay_lr` at `decay_steps`, then constant at `decay_lr`.
    `decay_lr: None` decays all the way to 0.
    See `core/optim.py::build_warmup_cosine`.
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
    model: dict[str, Any] = field(default_factory=dict)
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
    # None = historical behavior, all trainable shared.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset + partitioning spec."""

    name: str = "mock"
    repo_id: str | None = None
    revision: str | None = None  # not supported (the loader rejects it): no backend applies it
    batch_size: int = 8
    num_workers: int = 0
    partitioning: str = "iid"  # "iid" | "task" | "explicit_tasks" | "explicit_repos"
    num_clients: int = 1
    held_out: str | None = None  # path to held-out task spec (spec §9)
    task_key: str | None = None  # column name when partitioning="task"
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
    aggregate_lora_only: bool = False  # not supported (the loader rejects true)
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
    # Precision on the wire, in BOTH directions, on nvflare —
    # resolved through runtime/wire.py. Either a
    # scalar, "fp16" (default; halves every host-side copy of the payload) or
    # "fp32" (lossless — what the local<->nvflare equivalence test pins), or a
    # mapping for an asymmetric wire: {broadcast: fp32, delta: fp16}.
    # Measured fp16 cost on real pi05_base weights: max 4.9e-4 relative error,
    # zero overflow, ~20x finer than the bf16 autocast arithmetic downstream.
    # bf16 is ALWAYS downcast to fp16 regardless — numpy cannot carry bfloat16.
    wire_dtype: str | dict[str, str] = "fp16"


@dataclass
class BenchmarkConfig:
    name: str = "mock"  # algorithm/benchmark name
    suites: list[str] = field(default_factory=lambda: ["default"])
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    n_episodes_per_task: int = 5
    seed: int = 42
    eval_all_tasks: bool = True  # spec §9: eval held-out + seen tasks
    output_dir: str | None = None
    save_episodes: bool = False
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
class WandbConfig:
    """Weights & Biases logging, declared in the run config.

    Lives in the YAML rather than the launcher so a run's experiment-tracking
    identity travels WITH the hyperparameters it describes: the same file that
    fixes lr/rounds/strategy also fixes where the curves land. Launchers set
    env vars at best inconsistently, and a shell-only setting is invisible to
    anyone reading the config later.

    Precedence is config > environment > default, so the legacy env-var route
    (`WANDB_PROJECT`/`WANDB_NAME`) still works
    for configs that declare no `wandb:` block.
    """

    enabled: bool | None = None  # None = infer (on iff a project is resolvable).
    # Set false to force off for smokes.
    project: str | None = None  # e.g. "decent-vla" — the standing project.
    entity: str | None = None  # team/user; None = wandb default for the login.
    name: str | None = None  # run name; None = the config file's basename.
    group: str | None = None  # None = the config basename (groups resumed
    # chunks and the arms of one experiment).
    job_type: str | None = None
    mode: str = "online"  # "online" | "offline" | "disabled". Standing
    # policy: smokes offline, formal runs online.
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class RunConfig:
    """Policy, data and native FLARE workflow configuration."""

    name: str
    backend: BackendConfig
    data: DataConfig = field(default_factory=DataConfig)
    fl: FLConfig | None = None
    eval: EvalConfig | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    device: str = "cpu"
