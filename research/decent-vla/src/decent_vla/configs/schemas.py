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
    grad_clip_norm: float | None = None         # max grad-norm (None/0 = off; lerobot smolvla/pi05 default 10)


@dataclass
class BackendConfig:
    name: str                                   # entry-point key: "mock", "lerobot", "groot"
    model: dict[str, Any] = field(default_factory=dict)
    pretrained_path: str | None = None
    lora: LoRAConfig | None = None
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainable_params: list[str] | None = None   # AXIS 1 (local training scope): fnmatch patterns —
                                                 # matching params requires_grad=True, rest frozen.
                                                 # Applied after model build + LoRA/native tune flags
                                                 # (final override). None = policy defaults decide.
    shared_params: list[str] | None = None      # AXIS 2 (federated scope): fnmatch patterns over
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
    revision: str | None = None                 # not supported (the loader rejects it): no backend applies it
    batch_size: int = 8
    num_workers: int = 0
    partitioning: str = "iid"                   # "iid" | "task" | "explicit_tasks" | "explicit_repos"
    num_clients: int = 1
    held_out: str | None = None              # path to held-out task spec (spec §9)
    task_key: str | None = None              # column name when partitioning="task"
    seed: int = 42
    client_repo_ids: list[list[str]] | None = None   # explicit_repos: per-client repo lists
    client_task_indices: list[list[int]] | None = None  # explicit_tasks: per-client task indices
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    name: str = "fedavg"                        # entry-point key
    mu: float = 0.0                             # FedProx coefficient (ignored if name != "fedprox")
    server_lr: float = 1.0                      # FedAdam server-side LR
    server_betas: tuple[float, float] = (0.9, 0.999)
    server_eps: float = 1e-3
    aggregate_lora_only: bool = False           # not supported (the loader rejects true)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NVFlareConfig:
    """Runtime knobs for `fl.mode: nvflare` (NVIDIA FLARE >= 2.8).

    Generic FL knobs (num_rounds, local_steps, min_clients, round_timeout_s,
    save_every_n_rounds, wire_dtype, persist_client_state, output_dir) live on
    FLConfig and are shared across runtimes; only nvflare-execution specifics
    belong here.
    """

    env: str = "simulator"                      # "simulator" (in-process, blocking) | "export" (write a
                                                 # job folder for POC/provisioned submission and exit)
    workspace: str | None = None                # simulator workspace; default <fl.output_dir>/nvflare_workspace
    threads: int | None = None                  # simulator client concurrency; default num_clients — fewer
                                                 # threads make the simulator swap client processes, which
                                                 # drops warm in-memory client state between rounds
    gpu: str | None = None                      # simulator client->GPU map, ONE ENTRY PER CLIENT: 8 clients
                                                 # on 2 cards is "0,1,0,1,0,1,0,1", not "0,1". NVFlare runs one
                                                 # client per GPU group at a time, so a short map makes the
                                                 # simulator swap client processes (policy reload + cold Adam
                                                 # every round). Validated by runtime/nvflare/job.check_gpu_map.
    site_prefix: str = ""                       # "" = sites named c0..cN-1 (what the simulator env uses);
                                                 # "site-" maps a conventionally-provisioned site-{i+1} to c{i}
    launch_external_process: bool = False       # run the client script in a subprocess (CellPipe) instead of
                                                 # in-process — for training loops needing their own process
    wait_time_after_min_received: int = 10      # with fl.min_clients below the clients selected: once that many
                                                 # have answered, seconds to wait for the rest before the round
                                                 # closes (FLARE's ScatterAndGather default). The round closes
                                                 # sooner if every client answers.
    memory_gc_rounds: int = 1                   # run gc.collect()+malloc_trim() every N server rounds
                                                 # (0 = never). NVFlare puts this cleanup on its CONCRETE
                                                 # workflows (BaseFedAvg._maybe_cleanup_memory); a custom
                                                 # ModelController subclass like ours silently opts out.
                                                 # That matters because the normal task-exit path
                                                 # (wf_comm_server._do_check_tasks) never nulls task.data,
                                                 # and Task<->ClientTask is a reference CYCLE, so refcounting
                                                 # alone can never free a finished round's payload — only a
                                                 # gen-2 collection can. MEASURED consequence at pi0.5 scale:
                                                 # the server banked ~6.7 GiB per round and was OOM-killed at
                                                 # round ~8 of 50.
    job_dir: str | None = None                  # export env: where to write the job folder;
                                                 # default <fl.output_dir>/nvflare_job


@dataclass
class FLConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    num_rounds: int = 10
    local_steps: int = 50
    mode: str = "local"                         # "local" (single-process) | "nvflare"
    nvflare: NVFlareConfig = field(default_factory=NVFlareConfig)
    min_clients: int | None = None              # min client updates per round (default: all)
    round_timeout_s: float | None = None        # per-round timeout
    save_every_n_rounds: int | None = None
    output_dir: str = "outputs/fl_run"
    persist_client_state: bool = False          # opt-in: save/restore per-client state for warm resume
    client_state_keep: int | None = None        # retain only the N highest-round client-state files per
                                                 # client (None = keep all; minimum 2). Set to 2 for 3B-scale runs:
                                                 # a personalization client blob is ~35 GB.
    client_gpu_spread: bool = False             # local: one dedicated CUDA device per client (no single-GPU
                                                 # juggling). Engages only when visible GPUs >= num_clients,
                                                 # else falls back to juggling. For full-model FL where one
                                                 # client already fills a GPU (pi0.5/smolvla full-FT).
    # Precision on the wire, in BOTH directions, on nvflare —
    # resolved through runtime/wire.py. Either a
    # scalar, "fp16" (default; halves every host-side copy of the payload) or
    # "fp32" (lossless — what the local<->nvflare equivalence test pins), or a
    # mapping for an asymmetric wire: {broadcast: fp32, delta: fp16}.
    # Measured fp16 cost on real pi05_base weights: max 4.9e-4 relative error,
    # zero overflow, ~20x finer than the bf16 autocast arithmetic downstream.
    # bf16 is ALWAYS downcast to fp16 regardless — numpy cannot carry bfloat16.
    wire_dtype: str | dict[str, str] = "fp16"
    # Mutate server-owned buffers instead of duplicating the fp32 master:
    # in-place aggregation plus draining the bootstrap state while upcasting.
    # Numerically identical to the copying path; saves ~2 full payloads, which
    # at pi0.5's 3.62B params is ~27 GB. Set false to restore the historical
    # copying behaviour (needed only if you hold your own reference to the
    # state dict you handed the runner).
    lean_host_memory: bool = True


@dataclass
class TrainConfig:
    """Centralized (non-FL) training. Mutually exclusive with FLConfig."""

    num_steps: int = 1000
    save_every: int = 500
    output_dir: str = "outputs/centralized_run"


@dataclass
class BenchmarkConfig:
    name: str = "mock"                          # entry-point key
    suites: list[str] = field(default_factory=lambda: ["default"])
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    n_episodes_per_task: int = 5
    seed: int = 42
    after_training: bool = True
    every_n_rounds: int | None = None        # v1.1+; ignored in v0.1
    eval_all_tasks: bool = True              # spec §9: eval held-out + seen tasks
    only: bool = False
    output_dir: str | None = None
    save_episodes: bool = False
    personalized: bool = False               # partial-federation arms: evaluate each client's
                                              # model (server shared + its personal params) on its
                                              # OWN tasks; needs the training run's client_states
                                              # + partition manifest (fl.persist_client_state).
    n_action_steps: int | None = None        # rollout action horizon = the REPLANNING interval.
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

    enabled: bool | None = None   # None = infer (on iff a project is resolvable).
                                  # Set false to force off for smokes.
    project: str | None = None    # e.g. "decent-vla" — the standing project.
    entity: str | None = None     # team/user; None = wandb default for the login.
    name: str | None = None       # run name; None = the config file's basename.
    group: str | None = None      # None = the config basename (groups resumed
                                  # chunks and the arms of one experiment).
    job_type: str | None = None
    mode: str = "online"          # "online" | "offline" | "disabled". Standing
                                  # policy: smokes offline, formal runs online.
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class RunConfig:
    """Top-level config — one of `train` or `fl` must be set (unless eval.only)."""

    name: str
    backend: BackendConfig
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig | None = None
    fl: FLConfig | None = None
    eval: EvalConfig | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    device: str = "cpu"
