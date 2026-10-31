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

"""LeRobotBackend — Backend protocol implementation for HuggingFace lerobot policies.

Design note: rather than re-expressing every lerobot config field in
decent-vla's schema, this adapter accepts a path to an existing lerobot
`TrainPipelineConfig` YAML in `backend.extra.train_config_path`. The
adapter constructs the lerobot policy + dataset through lerobot's own
factories (`make_policy`, `make_dataset`), so any improvements upstream
flow through automatically.

This means a decent-vla config for a real lerobot run looks like:

    backend:
      name: lerobot
      extra:
        train_config_path: configs/smolvla_libero.yaml
      lora:
        r: 16
        alpha: 32
        target_modules: [q_proj, k_proj, v_proj, o_proj]

The `train_config_path` YAML is the same lerobot training config you'd
pass to `lerobot-train` — `policy.type`, `dataset.repo_id`, etc.

Partitioning is episode-level (splitting an episode across clients
breaks action-chunk policies): the adapter exposes the optional
`list_episodes(data_cfg)` Backend method (episode -> task metadata from
`LeRobotDatasetMetadata`), and the generic
`core/partitioner.py::partition_episodes` does the actual split.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import torch

from decent_vla.configs.schemas import BackendConfig, DataConfig, OptimConfig
from decent_vla.core.lora import apply_peft
from decent_vla.core.optim import build_warmup_cosine
from decent_vla.core.partitioner import EpisodeMeta, partition_episodes
from decent_vla.core.types import Batch, CheckpointSpec, LossDict, ParamScope

log = logging.getLogger(__name__)


# Backbone attribute priority for LoRA target resolution:
#   paligemma_with_expert — pi0 / pi0.5 PaliGemmaWithExpertModel
#   vlm_with_expert       — SmolVLA
#   vlm                   — anything else with a vlm attribute (fallback)
_LORA_ATTR_PRIORITY = ("paligemma_with_expert", "vlm_with_expert", "vlm")


class LeRobotBackend:
    """lerobot-policy backend. Satisfies `decent_vla.core.backend.Backend`."""

    def __init__(self, cfg: BackendConfig, device: torch.device, data_cfg: DataConfig | None = None):
        self.cfg = cfg
        self.device = device
        self._data_cfg = data_cfg              # optional DataConfig (explicit_repos stats)
        self._train_pipeline_cfg = None       # lerobot TrainPipelineConfig
        self._policy = None                    # lerobot PreTrainedPolicy
        self._dataset_meta = None              # LeRobotDatasetMetadata
        self._preprocessor = None
        self._postprocessor = None
        self._partition_episodes: list[int] | None = None  # set by build_dataloader
        self._autocast_dtype: torch.dtype | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, cfg: BackendConfig, device: torch.device, data_cfg: DataConfig | None = None
    ) -> LeRobotBackend:
        """Build policy + dataset metadata. Heavy lifting deferred to lerobot factories."""
        self = cls(cfg, device, data_cfg=data_cfg)
        self._train_pipeline_cfg = _load_train_pipeline_config(cfg)
        self._dataset_meta = _get_dataset_meta(self._train_pipeline_cfg)
        # explicit_repos FL: both clients must normalize with the SAME stats
        # (aggregated over the union of all client repos), so FedAvg never
        # averages divergent per-client normalization. Guarded: no-op for the
        # single-dataset path (client_repo_ids unset).
        if getattr(data_cfg, "client_repo_ids", None):
            self._dataset_meta.stats = aggregated_dataset_stats(data_cfg.client_repo_ids)
        self._policy = _build_policy(self._train_pipeline_cfg, self._dataset_meta, device)

        # Pre/post processor pipelines (tokenization, observation rename,
        # normalization, device placement). lerobot policies require the
        # preprocessor to run on every batch before `forward` — mirrors
        # lerobot_train.py. Built before the LoRA wrap for clarity (the
        # processors never touch parameters).
        # Offline gated-tokenizer fix: pi0/pi05 build their tokenizer by repo id
        # (e.g. the gated google/paligemma-3b-pt-224). Under HF_HUB_OFFLINE that
        # load probes the repo's config.json on the Hub and fails even though the
        # tokenizer files are cached. Resolving the repo to its local snapshot dir
        # makes transformers load straight from disk. Opt-in via
        # backend.extra.tokenizer_local_repo; no-op for other policies.
        tokenizer_override = None
        tok_repo = (cfg.extra or {}).get("tokenizer_local_repo")
        if tok_repo:
            snap = _local_hub_snapshot(tok_repo)
            if snap is not None:
                log.info("Redirecting tokenizer %s -> local snapshot %s", tok_repo, snap)
                tokenizer_override = {"tokenizer_processor": {"tokenizer_name": snap}}
            else:
                log.warning("tokenizer_local_repo=%s not found in HF cache; using as-is", tok_repo)
        self._preprocessor, self._postprocessor = _build_processors(
            self._train_pipeline_cfg, self._dataset_meta, self._policy, device,
            tokenizer_override=tokenizer_override,
        )

        # Optional LoRA wrapping.
        if cfg.lora is not None:
            log.info("Applying LoRA: r=%d alpha=%d targets=%s",
                     cfg.lora.r, cfg.lora.alpha, cfg.lora.target_modules)
            inner = getattr(self._policy, "model", self._policy)
            apply_peft(inner, cfg.lora, attr_priority=_LORA_ATTR_PRIORITY)

        # Axis 1 (backend.trainable_params): generic pattern freeze — the
        # final word on requires_grad, after the policy's native flags and
        # LoRA. Closes the gap for policies without freeze flags (PI05Config).
        if cfg.trainable_params is not None:
            from decent_vla.core.scope import apply_trainable_patterns

            apply_trainable_patterns(self._policy, cfg.trainable_params)

        # Optional compute dtype (backend.extra.dtype: bfloat16) — implemented
        # as bf16 AUTOCAST mixed precision (weights stay fp32), matching
        # lerobot's training recipe (accelerator.autocast() around forward;
        # smolvla's sample_noise is hard-coded fp32 and breaks under a pure
        # weight cast). Server master state stays fp32 end to end.
        dtype = (cfg.extra or {}).get("dtype")
        if dtype == "bfloat16":
            self._autocast_dtype = torch.bfloat16
        elif dtype in (None, "float32", "fp32"):
            self._autocast_dtype = None
        else:
            raise ValueError(f"Unsupported backend.extra.dtype: {dtype!r}")

        return self

    # ------------------------------------------------------------------
    # training loop primitives
    # ------------------------------------------------------------------

    def build_dataloader(
        self, data_cfg: DataConfig, partition_indices: list[int] | None
    ) -> Iterator[Batch]:
        """Build a cycling iterator over training batches.

        `partition_indices` here are interpreted as **episode indices**
        (lerobot is episodic) when the FL outer driver chose
        episode-level partitioning. If it's None we use the full dataset.
        """
        if getattr(data_cfg, "partitioning", None) == "explicit_repos":
            return self._build_concat_dataloader(data_cfg, partition_indices)

        from copy import deepcopy

        from lerobot.datasets.factory import make_dataset
        from torch.utils.data import DataLoader

        tp_cfg = deepcopy(self._train_pipeline_cfg)
        # Do NOT set `tp_cfg.dataset.episodes`: lerobot's dataset loader does
        # not row-filter on it (0.4.1 loads ALL parquet; 0.5.2 re-indexes the
        # hf_dataset from 0, which would desync the GLOBAL frame indices the
        # sampler needs). Relying on it silently trained every FL client on the
        # full dataset — the void-partition bug. Always build the full,
        # globally-indexed dataset and restrict to the partition via an
        # EpisodeAwareSampler keyed on the partition's episode indices.
        dataset = make_dataset(tp_cfg)
        _apply_dataset_tolerance(dataset, data_cfg)

        # Restrict to the client's partition via an EpisodeAwareSampler keyed
        # on its episode indices (NOT cfg.dataset.episodes — see above).
        sampler = None
        shuffle = True
        if partition_indices is not None:
            from lerobot.datasets.sampler import EpisodeAwareSampler

            self._partition_episodes = list(partition_indices)
            drop = getattr(getattr(tp_cfg, "policy", None), "drop_n_last_frames", 0) or 0
            sampler = EpisodeAwareSampler(
                dataset.meta.episodes["dataset_from_index"],
                dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=list(partition_indices),
                drop_n_last_frames=drop,
                shuffle=True,
            )
            shuffle = False  # sampler and shuffle are mutually exclusive
            n_frames = len(sampler)
            # Hard guard: a void/empty partition can never pass silently again.
            if n_frames == 0:
                raise ValueError(
                    f"partition episodes {list(partition_indices)!r} resolved to "
                    f"0 sampled frames — empty/invalid partition (the per-client "
                    f"shard would be void; check episode indices vs dataset)"
                )
            log.info(
                "lerobot client dataloader: %d episodes -> %d frames "
                "(partition-restricted via EpisodeAwareSampler)",
                len(partition_indices),
                n_frames,
            )

        # Worker subprocesses are forked AFTER the policy is on CUDA; with a
        # video-decoding dataset (LIBERO's AV1 via pyav) that fork-after-CUDA
        # path can segfault the workers when several clients share a node. `dataloader_mp_context` (e.g. "spawn"
        # or "forkserver") starts workers without inheriting the parent's CUDA
        # / libav state. `prefetch_factor` (e.g. 1) caps shared-memory
        # pressure. Both are opt-in via
        # data.extra so CPU/mock paths keep torch defaults.
        extra = data_cfg.extra or {}
        kwargs: dict = dict(
            batch_size=data_cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=data_cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        if data_cfg.num_workers > 0:
            mp_context = extra.get("dataloader_mp_context")
            if mp_context:
                kwargs["multiprocessing_context"] = mp_context
            kwargs["prefetch_factor"] = int(extra.get("prefetch_factor", 1))
            kwargs["persistent_workers"] = bool(
                extra.get("persistent_workers", False)
            )
        loader = DataLoader(dataset, **kwargs)
        return _cycle(loader)

    def _build_concat_dataloader(self, data_cfg, partition_indices):
        """explicit_repos: client trains on the FULL data of every repo in its
        list, concatenated. `partition_indices == [client_index]` (a sentinel,
        not episode indices — see runtime.builders.partition_for). Each
        sub-dataset keeps its own EpisodeAwareSampler (preserving
        drop_n_last_frames); indices are offset into the ConcatDataset and
        shuffled each epoch."""
        from copy import deepcopy

        from lerobot.datasets.factory import make_dataset
        from lerobot.datasets.sampler import EpisodeAwareSampler
        from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler

        client_idx = int(partition_indices[0])
        repo_ids = data_cfg.client_repo_ids[client_idx]
        drop = getattr(getattr(self._train_pipeline_cfg, "policy", None), "drop_n_last_frames", 0) or 0

        datasets, combined_idx, offset = [], [], 0
        for rid in repo_ids:
            tp = deepcopy(self._train_pipeline_cfg)
            tp.dataset.repo_id = rid
            tp.dataset.root = None
            tp.dataset.episodes = None  # full dataset per repo; never a subset
            ds = make_dataset(tp)
            _apply_dataset_tolerance(ds, data_cfg)
            samp = EpisodeAwareSampler(
                ds.meta.episodes["dataset_from_index"],
                ds.meta.episodes["dataset_to_index"],
                episode_indices_to_use=list(range(ds.meta.total_episodes)),
                drop_n_last_frames=drop,
                shuffle=False,
            )
            combined_idx.extend(offset + i for i in samp)
            offset += len(ds)
            datasets.append(ds)

        if not combined_idx:
            raise ValueError(f"explicit_repos client {client_idx} ({repo_ids}) sampled 0 frames")

        self._explicit_combined_idx = combined_idx
        self._explicit_frame_count = len(combined_idx)
        log.info("lerobot explicit_repos client %d: %d repos -> %d frames",
                 client_idx, len(repo_ids), len(combined_idx))

        concat = ConcatDataset(datasets)
        extra = data_cfg.extra or {}
        kwargs: dict = dict(
            batch_size=data_cfg.batch_size,
            sampler=SubsetRandomSampler(combined_idx),
            num_workers=data_cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        if data_cfg.num_workers > 0:
            mp_context = extra.get("dataloader_mp_context")
            if mp_context:
                kwargs["multiprocessing_context"] = mp_context
            kwargs["prefetch_factor"] = int(extra.get("prefetch_factor", 1))
            kwargs["persistent_workers"] = bool(extra.get("persistent_workers", False))
        return _cycle(DataLoader(concat, **kwargs))

    def num_train_frames(self) -> int | None:
        """Frames the current client's dataloader samples per epoch — the
        FedAvg weight under explicit_repos. None outside explicit_repos."""
        return getattr(self, "_explicit_frame_count", None)

    @staticmethod
    def list_episodes(data_cfg: DataConfig) -> list[EpisodeMeta]:
        """Optional Backend method (see core/backend.py): episode -> task metadata.

        Reads only `LeRobotDatasetMetadata` (no full-data download), so this
        is cheap. Static because it needs nothing but the dataset identity:
        `data_cfg.repo_id` (or `data_cfg.extra["repo_id"]`), optionally
        `data_cfg.extra["root"]`.
        """
        repo_id = data_cfg.repo_id or (data_cfg.extra or {}).get("repo_id")
        if not repo_id:
            raise ValueError(
                "lerobot list_episodes requires `data.repo_id` "
                "(e.g. HuggingFaceVLA/libero_spatial_no_noops)."
            )
        root = _resolve_local_root(repo_id, (data_cfg.extra or {}).get("root"))
        return _list_lerobot_episodes(repo_id, root=root)

    def compute_loss(self, batch: Batch) -> LossDict:
        """Forward pass. lerobot returns `(loss, output_dict)`; we wrap to LossDict."""
        self._policy.train()
        # The preprocessor pipeline handles tokenization, observation rename,
        # normalization, and device placement (mirrors lerobot_train.py's
        # `batch = preprocessor(batch)` before `policy.forward`).
        batch = self._preprocessor(batch)
        if self._autocast_dtype is not None:
            with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype):
                loss, output_dict = self._policy.forward(batch)
        else:
            loss, output_dict = self._policy.forward(batch)
        result: LossDict = {"loss": loss}
        if isinstance(output_dict, dict):
            for k, v in output_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    result[k] = v.detach()
        return result

    def build_optimizer(
        self, optim_cfg: OptimConfig
    ) -> tuple[torch.optim.Optimizer, object | None]:
        """Build optimizer over trainable params (LoRA-filtered if LoRA is on)."""
        params = [p for p in self._policy.parameters() if p.requires_grad]
        opt_name = (optim_cfg.name or "adamw").lower()
        if opt_name == "adamw":
            opt = torch.optim.AdamW(
                params,
                lr=optim_cfg.lr,
                betas=tuple(optim_cfg.betas),
                eps=optim_cfg.eps,
                weight_decay=optim_cfg.weight_decay,
            )
        elif opt_name == "adam":
            opt = torch.optim.Adam(
                params,
                lr=optim_cfg.lr,
                betas=tuple(optim_cfg.betas),
                eps=optim_cfg.eps,
                weight_decay=optim_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer name: {optim_cfg.name!r}")

        scheduler = None
        if optim_cfg.scheduler == "cosine":
            # lerobot recipe parity: linear warmup to peak lr, cosine decay
            # to decay_lr over decay_steps, constant floor after.
            scheduler = build_warmup_cosine(
                opt,
                lr=optim_cfg.lr,
                warmup_steps=optim_cfg.warmup_steps,
                decay_steps=optim_cfg.decay_steps,
                decay_lr=optim_cfg.decay_lr,
            )

        return opt, scheduler

    # ------------------------------------------------------------------
    # parameter exchange (the FL boundary)
    # ------------------------------------------------------------------

    def get_state(self, scope: ParamScope = "shared") -> dict[str, torch.Tensor]:
        """Return all TRAINABLE parameters (exactly what the optimizer trains).

        With LoRA (PEFT wraps only the VLM backbone) this is the LoRA
        adapters *plus* everything left trainable outside the wrapped
        submodule — action head / expert projections. Filtering by a
        ``lora_`` key substring instead would silently drop the trained
        action head from aggregation and broadcast (the documented
        action-head failure mode: server checkpoint evals ~0%).
        Without LoRA it is all trainable params (full FT).

        Detached + cloned. Buffers are excluded (named_parameters only).
        With `backend.shared_params` set, "shared" is the matching trainable
        subset and "personal" its complement (the personalization seam);
        otherwise shared = all trainable and personal is empty. "all" is
        always the full trainable set — checkpoints/restore use it.
        """
        from decent_vla.core.scope import trainable_state_keys

        keys = trainable_state_keys(self._policy)
        trainable = {
            k: v.detach().clone()
            for k, v in self._policy.state_dict().items()
            if k in keys
        }
        if scope == "all":
            return trainable
        from decent_vla.core.scope import resolve_shared_names

        shared = resolve_shared_names(trainable, self.cfg.shared_params)
        if scope == "personal":
            return {k: v for k, v in trainable.items() if k not in shared}
        return {k: v for k, v in trainable.items() if k in shared}

    def set_state(
        self, state: dict[str, torch.Tensor], scope: ParamScope = "shared"
    ) -> None:
        if scope == "personal":
            return
        own = self._policy.state_dict()
        for k, v in state.items():
            if k in own:
                own[k] = v.to(own[k].dtype).to(own[k].device)
        self._policy.load_state_dict(own, strict=False)

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, spec: CheckpointSpec) -> None:
        """Save in lerobot's native format so upstream tooling can still read it."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Use lerobot's PreTrainedPolicy.save_pretrained when available (gives
        # us a HF-style directory). For a raw .pt path, fall back to torch.save.
        if hasattr(self._policy, "save_pretrained") and p.suffix == "":
            self._policy.save_pretrained(p)
        else:
            torch.save(
                {
                    "state_dict": self._policy.state_dict(),
                    "include": spec.include,
                    "extra": spec.extra_state,
                },
                p,
            )

    def load_checkpoint(self, path: str) -> None:
        p = Path(path)
        if p.is_dir() and hasattr(self._policy, "from_pretrained"):
            # Reload from HF-style dir into the existing policy (preserves device + LoRA wrap).
            other = type(self._policy).from_pretrained(p)
            self._checked_load_state_dict(other.state_dict(), str(p))
            del other
        else:
            blob = torch.load(p, map_location=self.device, weights_only=False)
            sd = blob
            if isinstance(blob, dict):
                if "state_dict" in blob:
                    sd = blob["state_dict"]
                elif "state" in blob and isinstance(blob["state"], dict):
                    # round_loop server checkpoint: {"state": shared_state, "round": N}
                    sd = blob["state"]
            self._checked_load_state_dict(sd, str(p))

    def _checked_load_state_dict(self, sd: dict[str, torch.Tensor], source: str) -> None:
        """Load ``sd`` non-strict, but refuse the two silent-drop failure modes.

        ``load_state_dict(strict=False)`` silently ignores (a) a checkpoint that
        shares NO keys — eval then runs the base model (~0%), and (b) a checkpoint
        whose keys are ABSENT from the policy, e.g. a LoRA checkpoint (``lora_A``/
        ``lora_B`` keys) loaded into a full-FT-configured policy: the shared
        action-head keys overlap so a zero-overlap check passes while every LoRA
        weight is dropped. Both mean the eval config's arm doesn't match the
        checkpoint. Fail loudly. A legitimate partial load (the server's
        ``{"state": shared_state}`` blob) is a strict subset of the policy keys,
        so it has no absent keys and passes — but not *silently*: see the
        coverage warning below.
        """
        from decent_vla.core.scope import partial_load_warning, trainable_state_keys

        own = set(self._policy.state_dict())
        if not own & set(sd):
            raise ValueError(
                f"checkpoint {source} shares no keys with the policy state_dict; "
                "refusing to silently load nothing (arm/config mismatch?)"
            )
        absent = set(sd) - own
        if absent:
            raise ValueError(
                f"checkpoint {source} has {len(absent)} key(s) the policy cannot "
                f"receive (e.g. {sorted(absent)[:3]}); this usually means the eval "
                "config's arm (LoRA vs full fine-tune) does not match the checkpoint. "
                "Refusing to silently drop them — load with the matching config."
            )
        # Legal but never silent — see core.scope.partial_load_warning.
        msg = partial_load_warning(trainable_state_keys(self._policy), sd, source)
        if msg:
            log.warning("%s", msg)
        self._policy.load_state_dict(sd, strict=False)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def build_policy(self):
        from decent_vla.adapters.lerobot.policy import LeRobotPolicy

        return LeRobotPolicy(
            self._policy,
            self.device,
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            autocast_dtype=self._autocast_dtype,
        )

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    @property
    def model(self) -> torch.nn.Module:
        return self._policy

    @property
    def name(self) -> str:
        policy_type = getattr(getattr(self._train_pipeline_cfg, "policy", None), "type", "unknown")
        return f"lerobot:{policy_type}"

    @property
    def supports_personal_params(self) -> bool:
        return False


# ----------------------------------------------------------------------
# helpers (private; isolated for testability with mocked lerobot)
# ----------------------------------------------------------------------


def _load_train_pipeline_config(cfg: BackendConfig):
    """Load the lerobot `TrainPipelineConfig` YAML pointed at by cfg.extra.

    Resolution order:
      1) cfg.extra["train_config_path"]      — path to a lerobot YAML
      2) cfg.extra["train_pipeline_config"]  — an already-built TrainPipelineConfig
    """
    if cfg.extra:
        if "train_pipeline_config" in cfg.extra:
            return cfg.extra["train_pipeline_config"]
        tp_path = cfg.extra.get("train_config_path")
        if tp_path:
            import draccus
            import lerobot.policies  # noqa: F401 — registers policy config

            # choice classes (smolvla/pi05/…) with draccus before parsing
            from lerobot.configs.train import TrainPipelineConfig

            # draccus.parse takes a config *path* (it opens the file itself).
            # args=[] is load-bearing: draccus otherwise consumes the host
            # process's sys.argv as CLI overrides, which crashes inside
            # any launcher whose own flags are not TrainPipelineConfig fields.
            tp_cfg = draccus.parse(TrainPipelineConfig, tp_path, args=[])
            tp_cfg = _resolve_pretrained_policy(tp_cfg, tp_path)
            # Offline default: point the dataset at its local HF_LEROBOT_HOME
            # copy so lerobot 0.5.2 skips the online version check (which breaks
            # under HF_HUB_OFFLINE=1, required for the gated paligemma tokenizer).
            if getattr(tp_cfg.dataset, "root", None) is None:
                local_root = _resolve_local_root(getattr(tp_cfg.dataset, "repo_id", None))
                if local_root is not None:
                    tp_cfg.dataset.root = local_root
            return tp_cfg
    raise ValueError(
        "lerobot backend requires either `backend.extra.train_config_path` "
        "(path to a lerobot TrainPipelineConfig YAML) or "
        "`backend.extra.train_pipeline_config` (a built TrainPipelineConfig instance)."
    )


def _resolve_pretrained_policy(tp_cfg, yaml_path):
    """Mirror `TrainPipelineConfig.validate()`'s `--policy.path` branch.

    When `policy.pretrained_path` is set, lerobot-train loads the policy
    config FROM THE CHECKPOINT (`PreTrainedConfig.from_pretrained`) so
    `input_features` / `output_features` / normalization come from the
    pretrained model, with any other CLI policy fields applied as
    overrides. Parsing only the YAML leaves `input_features` empty, which
    makes `make_policy` fill them from the *dataset* feature names — that
    inverts the rename_map direction and crashes `prepare_images` ("All
    image features are missing from the batch").

    The YAML's `policy:` section (minus `type` / `pretrained_path`) plays
    the role of the CLI `--policy.*` overrides.
    """
    pretrained_path = getattr(tp_cfg.policy, "pretrained_path", None)
    if not pretrained_path:
        return tp_cfg

    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(
        str(pretrained_path),
        cli_overrides=_policy_cli_overrides(yaml_path),
    )
    policy_cfg.pretrained_path = Path(str(pretrained_path))
    tp_cfg.policy = policy_cfg
    return tp_cfg


def _policy_cli_overrides(yaml_path) -> list[str]:
    """YAML `policy:` section -> draccus CLI override strings.

    Equivalent to `parser.get_cli_overrides("policy")` in lerobot's
    validate(): de-nested `--{key}={value}` strings relative to the policy
    config root. `type` and `pretrained_path` are the path/choice keys and
    are excluded (like `--policy.path=` / `--policy.type=` upstream).
    """
    import json

    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    policy_section = raw.get("policy") or {}

    overrides: list[str] = []

    for key, value in policy_section.items():
        if key in ("type", "pretrained_path") or value is None:
            continue
        if isinstance(value, dict):
            # One json override per dict field (draccus decodes it). Dotted
            # expansion would break on feature keys that contain dots
            # ('observation.state').
            overrides.append(f"--{key}={json.dumps(value)}")
        elif isinstance(value, (list, tuple)):
            overrides.append(f"--{key}={json.dumps(list(value))}")
        elif isinstance(value, bool):
            # draccus only accepts lowercase true/false for bools.
            overrides.append(f"--{key}={json.dumps(value)}")
        else:
            overrides.append(f"--{key}={value}")
    return overrides


def _apply_dataset_tolerance(ds, data_cfg) -> None:
    """Opt-in override of LeRobotDataset's default 1e-4 s timestamp tolerance.

    Real-robot videos can have ~1-frame (33 ms @ 30 fps) timing jitter at
    episode boundaries, so the strict default raises inside __getitem__ ("query
    timestamps unexpectedly violate the tolerance"). Set `data.extra.tolerance_s`
    to relax it; when unset the lerobot default is left as-is, so LIBERO and the
    other arms are untouched. tolerance_s is read live at query time, so setting
    it on the already-built dataset takes effect for every subsequent item."""
    extra = getattr(data_cfg, "extra", None) or {}
    tol = extra.get("tolerance_s") if isinstance(extra, dict) else None
    if tol is not None:
        ds.tolerance_s = float(tol)


def _get_dataset_meta(tp_cfg):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(tp_cfg.dataset.repo_id, root=tp_cfg.dataset.root)


def aggregated_dataset_stats(client_repo_ids: list[list[str]], root: str | None = None) -> dict:
    """Count-weighted aggregate of per-repo stats over the UNION of all repos
    across all clients — the shared normalization for explicit_repos FL so
    FedAvg never averages divergent per-client normalization."""
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    seen, stats_list = set(), []
    for repos in client_repo_ids:
        for rid in repos:
            if rid in seen:
                continue
            seen.add(rid)
            stats_list.append(LeRobotDatasetMetadata(rid, root=root).stats)
    return aggregate_stats(stats_list)


def _build_policy(tp_cfg, ds_meta, device: torch.device):
    from lerobot.policies.factory import make_policy

    # lerobot builds the policy on its config's device (pi0.5 moves itself there
    # in __init__), so a CPU build would otherwise still allocate the whole model
    # on the GPU first. The .to() below keeps a device index such as cuda:1.
    tp_cfg.policy.device = device.type
    policy = make_policy(
        cfg=tp_cfg.policy,
        ds_meta=ds_meta,
        rename_map=getattr(tp_cfg, "rename_map", None),
    )
    policy.to(device)
    return policy


def _build_processors(tp_cfg, ds_meta, policy, device: torch.device, tokenizer_override=None):
    """Build the pre/post processor pipelines exactly like lerobot_train.py.

    The preprocessor handles tokenization, observation rename, normalization
    and device placement; the postprocessor unnormalizes actions. When the
    policy is pretrained, the saved processor configs are loaded from the
    pretrained path with overrides re-pointing them at the current dataset
    stats / device / rename_map (mirrors the fork's training script).

    `tokenizer_override` (optional) merges into the preprocessor overrides —
    used to redirect a gated tokenizer repo id to its local snapshot dir for
    offline runs (see from_config).
    """
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_path = getattr(tp_cfg.policy, "pretrained_path", None)
    kwargs: dict = {"dataset_stats": ds_meta.stats}
    if pretrained_path is not None:
        kwargs["preprocessor_overrides"] = {
            # str(device) keeps the CUDA index (e.g. "cuda:1"); device.type
            # would drop it to "cuda", placing input tokens on the current
            # device (cuda:0) while a client-per-GPU model sits on cuda:1.
            "device_processor": {"device": str(device)},
            "normalizer_processor": {
                "stats": ds_meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {
                "rename_map": getattr(tp_cfg, "rename_map", None)
            },
        }
        if tokenizer_override:
            kwargs["preprocessor_overrides"].update(tokenizer_override)
        kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": ds_meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    return make_pre_post_processors(
        policy_cfg=tp_cfg.policy,
        pretrained_path=pretrained_path,
        **kwargs,
    )


def _cycle(iterable):
    """Infinite iterator over a DataLoader (re-iterates on exhaustion).

    Inlined to avoid depending on lerobot's internal `cycle` helper, which moved
    out of `lerobot.datasets.utils` in 0.5.2.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def _resolve_local_root(repo_id: str | None, explicit_root: str | None = None) -> str | None:
    """Default the dataset root to the local HF_LEROBOT_HOME copy when present.

    lerobot 0.5.2's `LeRobotDatasetMetadata` runs an online version check
    (`get_safe_version`) whenever `root` is None — which raises under
    `HF_HUB_OFFLINE=1` (needed here for the gated paligemma tokenizer). Passing a
    root that already holds `meta/` makes the load fully offline + deterministic.
    No-op when a root is given explicitly or no local copy exists (then the
    caller keeps lerobot's default online behaviour).
    """
    if explicit_root:
        return explicit_root
    if not repo_id:
        return None
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
    except Exception:  # pragma: no cover - import path varies across lerobot versions
        try:
            from lerobot.constants import HF_LEROBOT_HOME  # older lerobot
        except Exception:
            return None
    cand = Path(HF_LEROBOT_HOME) / repo_id
    return str(cand) if (cand / "meta").is_dir() else None


def _local_hub_snapshot(repo_id: str) -> str | None:
    """Return the local HF-hub snapshot dir for a repo if its tokenizer is cached.

    Used to redirect a gated tokenizer (e.g. google/paligemma-3b-pt-224) to disk
    so it loads under HF_HUB_OFFLINE without the Hub config.json probe. Returns
    None when not cached (caller keeps the original repo id).
    """
    if not repo_id or Path(repo_id).is_dir():
        return repo_id if repo_id and Path(repo_id).is_dir() else None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None
    base = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    snaps = base / "snapshots"
    refs = base / "refs" / "main"
    candidates = []
    if refs.is_file():
        candidates.append(snaps / refs.read_text().strip())
    if snaps.is_dir():
        candidates.extend(sorted(snaps.iterdir(), reverse=True))
    for d in candidates:
        if (d / "tokenizer_config.json").is_file():
            return str(d)
    return None


def _list_lerobot_episodes(repo_id: str, root: str | None = None) -> list[EpisodeMeta]:
    """Load `LeRobotDatasetMetadata` and flatten it to one EpisodeMeta per episode.

    meta.episodes is a dict[episode_index] -> {tasks: [str], ...} in lerobot
    v2.x, and a columnar HF `datasets.Dataset` (with `episode_index` and
    `tasks` columns) in v3.x. Either way, an episode's task is the first
    entry of its `tasks` list (lerobot convention).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    episodes = meta.episodes
    if hasattr(episodes, "items"):  # v2: dict[episode_index] -> info dict
        ep_task_pairs = (
            (ep_idx, ep_info.get("tasks") if isinstance(ep_info, dict) else None)
            for ep_idx, ep_info in episodes.items()
        )
    elif hasattr(episodes, "column_names"):  # v3: columnar datasets.Dataset
        ep_task_pairs = zip(episodes["episode_index"], episodes["tasks"], strict=True)
    else:
        raise TypeError(
            "Unexpected `meta.episodes` type for lerobot dataset "
            f"{repo_id!r}: {type(episodes).__name__}. Expected a dict "
            "(lerobot v2.x) or a columnar datasets.Dataset (lerobot v3.x)."
        )
    episode_to_task: dict[int, str] = {}
    for ep_idx, tasks in ep_task_pairs:
        task_name = (
            tasks[0] if isinstance(tasks, list) and tasks
            else ("unknown" if tasks is None else str(tasks))
        )
        episode_to_task[int(ep_idx)] = task_name
    name_to_index = _task_name_to_index(meta, episode_to_task.values())
    # Assign new indices (after max existing) for any unmapped names, in sorted order.
    unmapped = sorted(n for n in episode_to_task.values() if n not in name_to_index)
    if unmapped:
        next_idx = (max(name_to_index.values()) + 1) if name_to_index else 0
        for name in unmapped:
            if name not in name_to_index:
                name_to_index[name] = next_idx
                next_idx += 1
    return [
        EpisodeMeta(episode_index=ep, task_index=name_to_index[name], task_name=name)
        for ep, name in sorted(episode_to_task.items())
    ]


def _task_name_to_index(meta, task_names: Iterable[str]) -> dict[str, int]:
    """Map task name -> dataset task_index, preferring the dataset's own mapping.

    `meta.tasks` is `{task_index: task_name}` in lerobot v2.x and a pandas
    DataFrame (task-name index, `task_index` column) in v3.x. If neither is
    present, fall back to enumerating sorted unique task names — stable, but
    only guaranteed to match `HeldOutSpec.dataset_task_index` when the
    dataset's own indices follow the same order.
    """
    tasks = getattr(meta, "tasks", None)
    if tasks is not None:
        columns = getattr(tasks, "columns", None)
        if columns is not None and "task_index" in columns:  # v3: pandas DataFrame
            return {str(name): int(idx) for name, idx in tasks["task_index"].items()}
        if isinstance(tasks, Mapping) and tasks:
            first_key = next(iter(tasks))
            if isinstance(first_key, int):  # v2: {task_index: task_name}
                return {str(name): int(idx) for idx, name in tasks.items()}
            return {str(name): int(idx) for name, idx in tasks.items()}
    return {name: i for i, name in enumerate(sorted(set(task_names)))}


# Episode-partition convenience (called by the runner / CLI when the user
# requests FL partitioning="task" or "iid_episode" for a lerobot dataset).
def partition_lerobot_dataset(
    repo_id: str,
    num_clients: int,
    partitioning: str = "task",
    seed: int = 42,
) -> list[list[int]]:
    """Return one list of episode indices per client.

    Thin wrapper over `list_episodes` + the generic
    `core/partitioner.py::partition_episodes` ("task" and "iid"/"iid_episode"
    resolve through its scheme aliases). Uses the dataset metadata only
    (no full-data load), so this is cheap.
    """
    episodes = _list_lerobot_episodes(repo_id)
    chunks, _manifest = partition_episodes(
        episodes,
        scheme=partitioning,
        num_clients=num_clients,
        seed=seed,
        dataset_repo_id=repo_id,
    )
    return chunks
