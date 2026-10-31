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

"""GrootBackend — Backend protocol implementation for NVIDIA Isaac-GR00T N1.7.

Drives the ``Gr00tN1d7`` model (Cosmos-Reason2-2B / Qwen3-VL backbone) for
federated **LoRA** finetuning of the foundation model on LIBERO. The recipe:

* Start from the BASE checkpoint ``nvidia/GR00T-N1.7-3B`` (federated finetuning
  of the foundation model), embodiment ``LIBERO_PANDA`` (value ``"libero_sim"``,
  projector slot index 2 — already present in the base model's
  ``max_num_embodiments``-sized embodiment-conditioned MLP, so no projector
  surgery is needed when loading from base; the embodiment is fed per-batch as
  ``embodiment_id``).
* Tune flags ``tune_llm=False, tune_visual=False, tune_projector=True,
  tune_diffusion_model=True, tune_vlln=True`` → the whole action head is
  trainable, nothing in the backbone.
* LoRA (wired here — Isaac-GR00T ships no PEFT) on the backbone language model
  (``model.backbone.model.language_model``) → adds trainable LoRA adapters.
* FL shared scope = ``{n: p for n, p in model.named_parameters()
  if p.requires_grad}`` = **LoRA adapters + action head**.

This adapter deliberately does NOT reuse ``Gr00tN1d7Pipeline._create_model``:
that path has a ``mask_token`` branch that AttributeErrors on N1.7 and raises on
any missing/unexpected key (so it rejects custom LoRA keys). We write our own
``AutoModel.from_pretrained`` load (mirroring ``setup.py:82-95``) after
``import gr00t.model`` (which registers ``Gr00tN1d7`` with ``AutoModel`` /
``AutoProcessor`` / ``AutoConfig``).

Config surface (decent-vla ``BackendConfig``):

    backend:
      name: groot
      pretrained_path: nvidia/GR00T-N1.7-3B   # or a local snapshot dir
      lora:
        r: 16
        alpha: 32
        attach_to: backbone.model.language_model   # default; or "action_head"
        target_modules: [q_proj, k_proj, v_proj, o_proj,
                         gate_proj, up_proj, down_proj]   # Qwen3-VL names.
                         # For attach_to: action_head use the diffusers names
                         # [to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2].
      extra:
        embodiment_tag: LIBERO_PANDA          # default
        state_dropout_prob: 0.2
        dtype: bfloat16                        # bf16 autocast (weights stay fp32)
        gradient_checkpointing: true
    data:
      repo_id: /path/to/libero_spatial_no_noops_1.0.0_lerobot   # a single LIBERO suite dir
      batch_size: 16
      num_workers: 4
      extra:
        video_backend: torchcodec             # or "pyav" / "decord"

This module is written against the Isaac-GR00T API (``third_party/Isaac-GR00T``)
and imports cleanly without gr00t installed (all heavy imports are
function-local). Points flagged ``VERIFY ON GPU`` mark the load-bearing
assumptions about that API.

All heavy imports (``gr00t``, ``torch`` beyond the protocol type hints,
``peft``) are function-local so the module lands in ``registry.errors`` rather
than crashing the CLI when Isaac-GR00T is absent (the decent-vla plugin
pattern).
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from decent_vla.core.optim import build_warmup_cosine
from decent_vla.core.partitioner import EpisodeMeta
from decent_vla.core.types import Batch, CheckpointSpec, LossDict, ParamScope

if TYPE_CHECKING:
    from decent_vla.configs.schemas import BackendConfig, DataConfig, OptimConfig
    from decent_vla.core.policy import Policy

log = logging.getLogger(__name__)

# Default LoRA targets for the Qwen3-VL language model — the attention + MLP
# Linear names verified against transformers 4.57.3's Qwen3-VL.
#
# These are WRONG for `attach_to: action_head`. That subtree is diffusers-shaped
# (gr00t/model/modules/dit.py builds diffusers' Attention + FeedForward), so its
# Linear leaves are `to_q`/`to_k`/`to_v`/`to_out.0`, `ff.net.0.proj`, `ff.net.2`,
# `norm1.linear`, `proj_out_1`, `proj_out_2` — enumerated on the real N1.7
# checkpoint, not guessed. An arm targeting the action head must spell them out.
# `_assert_targets_match` turns a wrong name into an error at build time rather
# than a silently smaller adapter set.
_DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Where LoRA goes when `backend.lora.attach_to` is not given: the historical
# arm, adapters on the VLM with the action head fully trainable.
_DEFAULT_LORA_ATTACH_TO = "backbone.model.language_model"

# Default base checkpoint and embodiment for the federated LIBERO recipe.
_DEFAULT_BASE_CKPT = "nvidia/GR00T-N1.7-3B"
_DEFAULT_EMBODIMENT = "LIBERO_PANDA"  # -> value "libero_sim", projector idx 2


def _resolve_module_path(model, path: str):
    """Walk a dotted attribute path; return (parent, final_attr, module).

    Returning the parent as well as the module is what lets the caller rebind
    the PEFT wrapper at an arbitrary depth.
    """
    parts = path.split(".")
    obj = model
    for i, part in enumerate(parts[:-1]):
        nxt = getattr(obj, part, None)
        if nxt is None:
            raise AttributeError(
                f"Could not resolve LoRA attach_to={path!r}: {'.'.join(parts[:i + 1])!r} "
                f"does not exist. Available at that level: "
                f"{[n for n, _ in obj.named_children()] if hasattr(obj, 'named_children') else '?'}"
            )
        obj = nxt
    leaf = getattr(obj, parts[-1], None)
    if leaf is None:
        raise AttributeError(
            f"Could not resolve LoRA attach_to={path!r}: {parts[-1]!r} does not exist on "
            f"{'.'.join(parts[:-1]) or 'the model root'}. Available: "
            f"{[n for n, _ in obj.named_children()] if hasattr(obj, 'named_children') else '?'}"
        )
    return obj, parts[-1], leaf


def _assert_targets_match(module, targets, path: str) -> None:
    """Every requested target name must hit at least one Linear.

    PEFT raises only when *all* targets miss; one bad name among several is
    silently dropped, which quietly shrinks the adapted set — and with it the
    trainable/uplink parameter counts an arm is measured by. Cheap to check
    here, expensive to notice 50 rounds later.
    """
    def matches(key: str, t: str) -> bool:      # peft's check_target_module_exists
        return key == t or key.endswith("." + t)

    if isinstance(targets, str):                # regex form: PEFT's business, not ours
        return
    names = [n for n, m in module.named_modules() if isinstance(m, torch.nn.Linear)]
    missed = [t for t in targets if not any(matches(n, t) for n in names)]
    if missed:
        leaves = sorted({n.rsplit(".", 1)[-1] for n in names})
        raise ValueError(
            f"LoRA target_modules {missed} match no Linear under {path!r}. "
            f"Linear leaf names available there: {leaves}"
        )


class GrootBackend:
    """Isaac-GR00T N1.7 backend. Satisfies ``decent_vla.core.backend.Backend``."""

    def __init__(self, cfg: BackendConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._model = None                       # Gr00tN1d7
        self._processor = None                   # Gr00tN1d7Processor (statistics set per dataset)
        self._embodiment_value: str = "libero_sim"
        self._modality_configs: dict[str, Any] | None = None  # MODALITY_CONFIGS[emb]
        self._autocast_dtype: torch.dtype | None = None
        self._stats_initialized = False          # processor.set_statistics done?

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: BackendConfig, device: torch.device) -> GrootBackend:
        """Load the base GR00T model, wire LoRA, build the processor.

        Mirrors ``gr00t/model/gr00t_n1d7/setup.py:82-95`` for the model load
        but bypasses the buggy ``_create_model`` (see module docstring).
        """
        # Heavy imports kept local to the constructor so the module imports
        # without Isaac-GR00T installed.
        import gr00t.model  # noqa: F401 — registers Gr00tN1d7 for AutoModel/AutoProcessor
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.embodiment_tags import EmbodimentTag
        from transformers import AutoModel

        self = cls(cfg, device)
        extra = cfg.extra or {}

        # ---- resolve embodiment ----
        emb_tag = EmbodimentTag.resolve(extra.get("embodiment_tag", _DEFAULT_EMBODIMENT))
        self._embodiment_value = emb_tag.value  # "libero_sim"
        if self._embodiment_value not in MODALITY_CONFIGS:
            raise ValueError(
                f"No registered modality config for embodiment {self._embodiment_value!r}. "
                f"Known: {sorted(MODALITY_CONFIGS)}. (LIBERO is pre-registered as "
                f"'libero_sim' in gr00t/configs/data/embodiment_configs.py.)"
            )
        self._modality_configs = MODALITY_CONFIGS[self._embodiment_value]

        # ---- model load (our own, not Pipeline._create_model) ----
        base_ckpt = cfg.pretrained_path or _DEFAULT_BASE_CKPT
        transformers_loading_kwargs = {
            "trust_remote_code": True,
            "local_files_only": bool(extra.get("local_files_only", False)),
        }
        log.info("Loading GR00T base model from %s (embodiment=%s)", base_ckpt, self._embodiment_value)
        model = AutoModel.from_pretrained(
            base_ckpt,
            tune_llm=bool(extra.get("tune_llm", False)),
            tune_visual=bool(extra.get("tune_visual", False)),
            tune_projector=bool(extra.get("tune_projector", True)),
            tune_diffusion_model=bool(extra.get("tune_diffusion_model", True)),
            tune_vlln=bool(extra.get("tune_vlln", True)),
            state_dropout_prob=float(extra.get("state_dropout_prob", 0.2)),
            backbone_trainable_params_fp32=True,
            load_bf16=bool(extra.get("load_bf16", False)),
            transformers_loading_kwargs=transformers_loading_kwargs,
            **transformers_loading_kwargs,
        )
        model.to(device)
        self._model = model

        # ---- optional gradient checkpointing (supported per the API map) ----
        if bool(extra.get("gradient_checkpointing", False)):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            else:  # pragma: no cover - depends on transformers version
                log.warning("gradient_checkpointing requested but model has no enable() method")

        # ---- LoRA on the backbone language model (we wire PEFT ourselves) ----
        if cfg.lora is not None:
            self._apply_lora(model, cfg.lora)

        # ---- Axis 1 (backend.trainable_params): generic pattern freeze ----
        # Final override on requires_grad, composing with (after) the native
        # tune_* flags and LoRA above.
        if cfg.trainable_params is not None:
            from decent_vla.core.scope import apply_trainable_patterns

            apply_trainable_patterns(model, cfg.trainable_params)

        # ---- processor (state/action normalization, tokenization, embodiment id) ----
        self._processor = self._build_processor(base_ckpt, extra, transformers_loading_kwargs)

        # ---- compute dtype (bf16 autocast; weights stay fp32, matching lerobot adapter) ----
        dtype = extra.get("dtype")
        if dtype == "bfloat16":
            self._autocast_dtype = torch.bfloat16
        elif dtype in (None, "float32", "fp32"):
            self._autocast_dtype = None
        else:
            raise ValueError(f"Unsupported backend.extra.dtype: {dtype!r}")

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log.info(
            "GR00T trainable params (LoRA + action head): %d / %d (%.2f%%)",
            n_train, n_total, 100.0 * n_train / max(n_total, 1),
        )
        return self

    def _apply_lora(self, model, lora_cfg) -> None:
        """Attach LoRA adapters to the submodule named by ``lora.attach_to``.

        Isaac-GR00T has no PEFT integration, so we call ``get_peft_model``
        ourselves and rebind the wrapper in place, leaving every other part of
        the network referencing the same module objects.

        Two attachment points are in use:

        * ``backbone.model.language_model`` (the default) — the Qwen3-VL decoder
          (``gr00t/model/modules/qwen3_backbone.py:80-88``). Adapts the VLM,
          action head stays fully trainable from the ``tune_*`` flags.
        * ``action_head`` — the inverse arm: backbone frozen, the 1.62B action
          head adapted instead. Its Linear leaves are diffusers-shaped
          (``to_q``/``to_k``/``to_v``/``to_out.0``, ``ff.net.0.proj``,
          ``ff.net.2``), NOT the Qwen names — see ``_DEFAULT_LORA_TARGETS``.

        We do NOT use ``core/lora.py::apply_peft``: that helper picks a
        submodule by attr-name priority and re-assigns on the model root, which
        cannot express a path three levels deep.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "GR00T LoRA requested but `peft` is not installed. "
                "Install with: pip install -e '.[peft]'"
            ) from exc

        path = getattr(lora_cfg, "attach_to", None) or _DEFAULT_LORA_ATTACH_TO
        parent, attr, target = _resolve_module_path(model, path)

        targets = lora_cfg.target_modules
        if not targets:
            targets = list(_DEFAULT_LORA_TARGETS)
        _assert_targets_match(target, targets, path)

        log.info("Applying LoRA to %s: r=%d alpha=%d targets=%s",
                 path, lora_cfg.r, lora_cfg.alpha, targets)
        peft_cfg = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=list(targets),
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,  # left None for a VLA backbone
        )
        # get_peft_model mutates `target` and returns a PeftModel wrapper;
        # rebind it on its parent so the path now carries the adapters.
        setattr(parent, attr, get_peft_model(target, peft_cfg))

    def _build_processor(self, base_ckpt: str, extra: dict, transformers_loading_kwargs: dict):
        """Build the GR00T processor with the LIBERO modality config injected.

        Mirrors ``setup.py:155`` (``AutoProcessor.from_pretrained(<ckpt>,
        modality_configs=..., state_dropout_prob=..., ...)``). The processor
        owns state/action normalization, VLM tokenization, image transforms and
        the ``embodiment_id`` injection. Its ``.collator`` (a
        ``Gr00tN1d7DataCollator``) is what we use as the DataLoader collate_fn.

        IMPORTANT: the base checkpoint's ``statistics.json`` does NOT contain
        ``libero_sim`` stats (LIBERO is a posttrain embodiment). The processor
        therefore cannot normalize LIBERO state/action until
        ``set_statistics`` is called with stats computed from the actual
        dataset. ``build_dataloader`` does that on first call (via the upstream
        ``ShardedMixtureDataset.merge_statistics`` path) — see
        ``_ensure_statistics``.
        """
        from transformers import AutoProcessor

        modality_configs = {self._embodiment_value: self._modality_configs}
        processor = AutoProcessor.from_pretrained(
            base_ckpt,
            modality_configs=modality_configs,
            state_dropout_prob=float(extra.get("state_dropout_prob", 0.2)),
            transformers_loading_kwargs=transformers_loading_kwargs,
            **transformers_loading_kwargs,
        )
        return processor

    # ------------------------------------------------------------------
    # training loop primitives
    # ------------------------------------------------------------------

    def build_dataloader(
        self, data_cfg: DataConfig, partition_indices: list[int] | None
    ) -> Iterator[Batch]:
        """Build a cycling iterator over collated GR00T training batches.

        ``partition_indices`` are **episode indices** (GR00T data is episodic).
        ``None`` -> use every episode.

        GR00T's ``ShardedSingleStepDataset`` is shard-based (not map-style), and
        its constituent timesteps are only processed once a ``processor`` with
        dataset statistics is attached. We:

          1. Build a per-client ``ShardedSingleStepDataset`` restricted to the
             partition's episodes (``build_partitioned_dataset``).
          2. Wire statistics + processor into it exactly the way upstream does
             — by constructing a single-dataset ``ShardedMixtureDataset``,
             which runs ``merge_statistics`` (sets ``processor.set_statistics``)
             and ``ds.set_processor`` (so ``get_shard`` works). We reuse this
             only for its setup side effects; we do NOT iterate the streaming
             mixture (its IterableDataset semantics are awkward for
             deterministic per-round FL batches).
          3. Flatten the sharded dataset into a torch **map-style** dataset
             (``_FlattenedShardDataset``) of individual processed timesteps and
             drive it with a plain ``torch.utils.data.DataLoader`` whose
             ``collate_fn`` is ``processor.collator``. The collator yields a
             ``BatchFeature({"inputs": {...}})``.

        VERIFY ON GPU:
          * ``processor.collator`` is the right collate_fn and the resulting
            ``BatchFeature`` carries everything ``model.forward`` needs.
          * memory: ``_FlattenedShardDataset`` materializes shard *index*
            metadata eagerly but loads each shard's timesteps lazily per
            ``__getitem__``; for big partitions consider ``num_workers``.
        """
        from torch.utils.data import ConcatDataset, DataLoader

        extra = data_cfg.extra or {}
        video_backend = extra.get("video_backend", "torchcodec")

        # Cross-suite support: GR00T's LIBERO is 4 separate suite datasets
        # (spatial/object/goal/10). `list_episodes` assigns GLOBAL episode indices
        # across the suites in path order; here we map the client's global episode
        # partition back to per-suite LOCAL episode indices using the same offsets.
        paths = self._resolve_dataset_paths(data_cfg)
        offsets = _suite_offsets(paths)  # [(path, global_start, n_episodes), ...]
        per_suite: list[tuple[str, list[int] | None]] = []
        for path, start, n in offsets:
            if partition_indices is None:
                per_suite.append((path, None))
            else:
                local = sorted(gi - start for gi in partition_indices if start <= gi < start + n)
                if local:
                    per_suite.append((path, local))
        if not per_suite:
            raise ValueError(
                f"GR00T partition {partition_indices!r} matched no episodes across {paths}"
            )

        # Build one filtered ShardedSingleStepDataset per suite, wire stats across
        # ALL of them in a single mixture (merges statistics), then flatten + concat.
        sharded_list = self._build_and_wire_datasets(
            per_suite,
            video_backend=video_backend,
            video_backend_kwargs=extra.get("video_backend_kwargs"),
            shard_size=int(extra.get("shard_size", 2**10)),
            episode_sampling_rate=float(extra.get("episode_sampling_rate", 1.0)),
            seed=data_cfg.seed,
            allow_padding=bool(extra.get("allow_padding", False)),
        )
        lazy_dp = bool(extra.get("lazy_episode_datapoints", False))
        flats = [
            _FlattenedShardDataset(
                s,
                lazy_datapoints=lazy_dp,
                episode_cache_size=int(extra.get("episode_cache_size", 0)),
            )
            for s in sharded_list
        ]
        if lazy_dp:
            log.info(
                "GR00T client dataloader: lazy-datapoint mode ON "
                "(global shuffle preserved; raw episodes cached per worker, "
                "cache_size=%s)",
                extra.get("episode_cache_size", 0) or "unbounded",
            )
        combined = flats[0] if len(flats) == 1 else ConcatDataset(flats)
        if len(combined) == 0:
            raise ValueError(
                f"GR00T partition {partition_indices!r} produced 0 timesteps "
                f"across {[p for p, _ in per_suite]} — empty/invalid partition."
            )
        log.info(
            "GR00T client dataloader: %d suite(s), %s episodes -> %d shards -> %d timesteps",
            len(flats),
            "all" if partition_indices is None else len(partition_indices),
            sum(len(s) for s in sharded_list),
            len(combined),
        )

        kwargs: dict = dict(
            num_workers=data_cfg.num_workers,
            collate_fn=self._processor.collator,
            pin_memory=(self.device.type == "cuda"),
        )
        if bool(extra.get("shard_grouped_batches", False)):
            # See _ShardGroupedBatchSampler: a global shuffle over ~30 shards
            # re-decodes a whole ~1000-timestep shard for every single sample.
            groups = _shard_groups(combined)
            sampler = _ShardGroupedBatchSampler(
                groups,
                batch_size=data_cfg.batch_size,
                seed=data_cfg.seed,
                drop_last=bool(extra.get("drop_last", False)),
            )
            kwargs["batch_sampler"] = sampler
            log.info(
                "GR00T client dataloader: shard-grouped batches ON "
                "(%d shard groups, %d batches/epoch, ~%.0f samples per shard load)",
                len(groups),
                len(sampler),
                (len(combined) / len(groups)) if groups else 0.0,
            )
        else:
            kwargs["batch_size"] = data_cfg.batch_size
            kwargs["shuffle"] = True
            kwargs["drop_last"] = False
        if data_cfg.num_workers > 0:
            mp_context = extra.get("dataloader_mp_context")
            if mp_context:
                kwargs["multiprocessing_context"] = mp_context
            kwargs["prefetch_factor"] = int(extra.get("prefetch_factor", 1))
            kwargs["persistent_workers"] = bool(extra.get("persistent_workers", False))
        loader = DataLoader(combined, **kwargs)
        return _cycle(loader)

    def _build_and_wire_datasets(
        self,
        per_suite: list[tuple[str, list[int] | None]],
        *,
        video_backend: str,
        video_backend_kwargs: dict | None,
        shard_size: int,
        episode_sampling_rate: float,
        seed: int,
        allow_padding: bool,
    ) -> list:
        """Build one filtered ShardedSingleStepDataset per suite, wire stats once.

        Passing ALL suites' datasets to a single ``ShardedMixtureDataset`` runs
        upstream's ``merge_statistics`` across them (so normalization stats are
        computed over the client's full cross-suite data) and calls
        ``set_processor`` on each. We discard the mixture and drive our own
        flatteners (one per suite, concatenated). ``override_pretraining_statistics``
        because the base processor has no ``libero_sim`` stats to begin with.
        """
        from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset

        from decent_vla.adapters.groot._dataset import build_partitioned_dataset

        sharded_list = [
            build_partitioned_dataset(
                dataset_path=path,
                embodiment_tag_value=self._embodiment_value,
                modality_configs=self._modality_configs,
                keep_episode_indices=local_eps,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
                shard_size=shard_size,
                episode_sampling_rate=episode_sampling_rate,
                seed=seed,
                allow_padding=allow_padding,
            )
            for path, local_eps in per_suite
        ]
        ShardedMixtureDataset(
            datasets=sharded_list,
            weights=[1.0] * len(sharded_list),
            processor=self._processor,
            seed=seed,
            training=True,
            override_pretraining_statistics=True,
        )
        self._stats_initialized = True
        for s in sharded_list:
            if getattr(s, "processor", None) is None:
                s.set_processor(self._processor)
        return sharded_list

    def compute_loss(self, batch: Batch) -> LossDict:
        """Forward pass. ``Gr00tN1d7.forward(inputs_dict)`` returns a dict-like
        ``BatchFeature`` with ``"loss"`` (flow-matching MSE).

        The collator wraps the batch as ``BatchFeature({"inputs": {...}})``, so
        the model expects the inner dict positionally: ``model(batch["inputs"])``
        (equivalent to ``model(**collated)`` -> ``forward(inputs=...)``).
        """
        self._model.train()
        # The collator returns a transformers BatchFeature, which is a UserDict
        # (NOT a dict subclass) — an isinstance(batch, dict) check fails and would
        # pass the doubly-wrapped {"inputs": {...}} straight to the model, so the
        # backbone can't find input_ids at the top level. Check membership directly.
        try:
            inner = batch["inputs"] if "inputs" in batch else batch
        except (TypeError, KeyError):
            inner = batch
        if self._autocast_dtype is not None:
            with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype):
                out = self._model(inner)
        else:
            out = self._model(inner)

        loss = out["loss"]
        result: LossDict = {"loss": loss}
        # Surface scalar extras (e.g. "action_loss") as detached metrics, like
        # the lerobot adapter does for lerobot's output_dict.
        try:
            items = out.items()
        except AttributeError:  # pragma: no cover - non-mapping output
            items = []
        for k, v in items:
            if k == "loss":
                continue
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                result[k] = v.detach()
        return result

    def build_optimizer(
        self, optim_cfg: OptimConfig
    ) -> tuple[torch.optim.Optimizer, object | None]:
        """AdamW over trainable params (LoRA + action head) + warmup-cosine LR."""
        params = [p for p in self._model.parameters() if p.requires_grad]
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
        """Return all TRAINABLE parameters — exactly LoRA adapters + action head.

        Mirrors the lerobot adapter: take the names of every parameter with
        ``requires_grad`` and slice them (plus their sub-keys) out of the full
        ``state_dict``, detached + cloned. Filtering on a ``lora_`` substring
        would silently drop the trained action head from aggregation.

        With ``backend.shared_params`` set, "shared" is the matching trainable
        subset and "personal" its complement (the personalization seam);
        otherwise shared = all trainable and personal is empty. "all" is
        always the full trainable set — checkpoints/restore use it.
        """
        from decent_vla.core.scope import trainable_state_keys

        keys = trainable_state_keys(self._model)
        trainable = {
            k: v.detach().clone()
            for k, v in self._model.state_dict().items()
            if k in keys
        }
        if scope == "all":
            return trainable
        from decent_vla.core.scope import resolve_shared_names

        shared = resolve_shared_names(trainable, self.cfg.shared_params)
        if scope == "personal":
            return {k: v for k, v in trainable.items() if k not in shared}
        return {k: v for k, v in trainable.items() if k in shared}

    def set_state(self, state: dict[str, torch.Tensor], scope: ParamScope = "shared") -> None:
        """Load shared params with dtype/device casting; non-strict."""
        if scope == "personal":
            return
        own = self._model.state_dict()
        for k, v in state.items():
            if k in own:
                own[k] = v.to(own[k].dtype).to(own[k].device)
        self._model.load_state_dict(own, strict=False)

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, spec: CheckpointSpec) -> None:
        """Save the trainable state + metadata via torch.save.

        Kept deliberately simple and robust: we serialize the FULL trainable
        sub-state (``get_state("all")`` — under ``backend.shared_params`` the
        "shared" scope is only the federated subset, and narrowing to it here
        would silently drop trained personal params), plus provenance. A full
        HF-style ``save_pretrained`` would also persist the
        frozen 3B backbone (huge, redundant across rounds) — out of scope for
        the FL round checkpoints.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.get_state("all"),
                "include": spec.include,
                "extra": spec.extra_state,
                "embodiment": self._embodiment_value,
                "backend": self.name,
            },
            p,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load a trainable-state checkpoint written by ``save_checkpoint``.

        Also accepts a round_loop server checkpoint (``{"state": shared_state,
        "round": N}``). Fails loudly if no keys overlap, so we never silently
        run the un-finetuned base model (the documented action-head failure
        mode).
        """
        p = Path(path)
        blob = torch.load(p, map_location=self.device, weights_only=False)
        sd = blob
        if isinstance(blob, dict):
            if "state_dict" in blob:
                sd = blob["state_dict"]
            elif "state" in blob and isinstance(blob["state"], dict):
                sd = blob["state"]
        own = set(self._model.state_dict())
        if not own & set(sd):
            raise ValueError(
                f"checkpoint {p} shares no keys with the GR00T model state_dict; "
                "refusing to silently load nothing (arm/config mismatch?)"
            )
        absent = set(sd) - own
        if absent:
            # e.g. a LoRA checkpoint loaded into a non-LoRA (action-head-only)
            # model: the shared keys overlap but every lora_* weight would be
            # silently dropped. Arm mismatch — fail loudly. A server
            # {"state": shared_state} blob is a subset, so it has no absent keys.
            raise ValueError(
                f"checkpoint {p} has {len(absent)} key(s) the GR00T model cannot "
                f"receive (e.g. {sorted(absent)[:3]}); the eval config's arm "
                "(LoRA vs full fine-tune) likely does not match the checkpoint. "
                "Refusing to silently drop them — load with the matching config."
            )
        # Legal but never silent — see core.scope.partial_load_warning. Matters
        # most here: a GR00T backbone-share server checkpoint is 1.52B of a
        # 3.14B model, so exporting it alone leaves the whole action head at its
        # pretrained values.
        from decent_vla.core.scope import partial_load_warning, trainable_state_keys

        msg = partial_load_warning(trainable_state_keys(self._model), sd, str(p))
        if msg:
            log.warning("%s", msg)
        self.set_state(sd, scope="shared")

    # ------------------------------------------------------------------
    # inference (eval path — out of scope this pass)
    # ------------------------------------------------------------------

    def build_policy(self) -> Policy:
        """Return a GrootPolicy over the (FL-aggregated) in-memory model+processor.

        The processor carries NO LIBERO normalization statistics until they are
        wired from the dataset. ``build_dataloader`` does that during training,
        but at eval it never runs, so without wiring them here the model's
        actions can't be un-normalized and eval is garbage. We wire them
        (idempotent) from ``backend.extra.eval_dataset_paths`` (falls back to
        ``dataset_paths``) and load the per-key state/action index ranges from
        the dataset's ``meta/modality.json``.
        """
        from decent_vla.adapters.groot.policy import GrootPolicy

        extra = self.cfg.extra or {}
        paths = extra.get("eval_dataset_paths") or extra.get("dataset_paths")
        if not paths:
            raise ValueError(
                "GR00T build_policy needs dataset paths to wire normalization "
                "statistics + modality ranges at eval time. Set "
                "backend.extra.eval_dataset_paths (or dataset_paths) in the eval config."
            )
        if isinstance(paths, str):
            paths = [paths]

        # 1) wire LIBERO normalization stats into the processor (idempotent),
        #    reusing the training stats-merge path over every episode per suite.
        if not self._stats_initialized:
            log.info("eval: wiring GR00T normalization statistics from %d suite(s)", len(paths))
            self._build_and_wire_datasets(
                [(p, None) for p in paths],
                video_backend=extra.get("eval_video_backend", extra.get("video_backend", "pyav")),
                video_backend_kwargs=extra.get("video_backend_kwargs"),
                shard_size=int(extra.get("shard_size", 2 ** 10)),
                episode_sampling_rate=1.0,
                seed=0,
                allow_padding=False,
            )

        # 2) per-key index ranges from the dataset modality.json
        state_ranges, action_ranges = _load_modality_ranges(paths[0])

        video_key_map = extra.get("eval_video_key_map") or {"image": "image", "wrist_image": "image2"}
        return GrootPolicy(
            self._model,
            self._processor,
            embodiment_value=self._embodiment_value,
            modality_configs=self._modality_configs,
            device=self.device,
            state_index_ranges=state_ranges,
            action_index_ranges=action_ranges,
            video_key_map=video_key_map,
            autocast_dtype=self._autocast_dtype,
            replan_steps=extra.get("eval_replan_steps"),
            gripper_transform=extra.get("eval_gripper_transform", "normalize_invert"),
        )

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def name(self) -> str:
        return f"groot:n1d7:{self._embodiment_value}"

    @property
    def supports_personal_params(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # optional Backend method: episode metadata for the generic partitioner
    # ------------------------------------------------------------------

    @staticmethod
    def list_episodes(data_cfg: DataConfig) -> list[EpisodeMeta]:
        """Episode -> task metadata for the generic partitioner, across 1+ suites.

        GR00T's LIBERO is 4 separate suite datasets. We assign GLOBAL episode
        indices cumulatively in dataset-path order (suite k: ``offset_k +
        local_episode_index``), matching the offsets ``build_dataloader`` uses to
        map a client's global episode partition back to per-suite LOCAL indices.
        Task indices are unified across suites by task NAME, so
        ``task_round_robin`` partitions over ALL tasks (e.g. all 40 LIBERO tasks
        across the 4 suites). Reads only ``meta/episodes.jsonl`` per suite (cheap,
        callable before any backend instance exists).

        Dataset paths: ``data.repo_id`` (str or list) /
        ``data.extra['dataset_paths']`` / ``['dataset_path']`` / ``['root']``.
        """
        import json as _json

        paths = GrootBackend._resolve_dataset_paths(data_cfg)
        out: list[EpisodeMeta] = []
        global_task_index: dict[str, int] = {}
        offset = 0
        for path in paths:
            episodes_path = Path(path) / "meta" / "episodes.jsonl"
            if not episodes_path.is_file():
                raise FileNotFoundError(
                    f"GR00T list_episodes: {episodes_path} not found. data paths "
                    "must point at LeRobot v2.x dirs with meta/episodes.jsonl."
                )
            n = 0
            with episodes_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ep = _json.loads(line)
                    local_idx = int(ep["episode_index"])
                    # An episode's task is the first entry of its "tasks" list
                    # (LeRobot convention). Unify task ids across suites by name.
                    tasks = ep.get("tasks")
                    if isinstance(tasks, list) and tasks:
                        task_name = str(tasks[0])
                    elif tasks is None:
                        task_name = "unknown"
                    else:
                        task_name = str(tasks)
                    tidx = global_task_index.setdefault(task_name, len(global_task_index))
                    out.append(
                        EpisodeMeta(
                            episode_index=offset + local_idx,
                            task_index=tidx,
                            task_name=task_name,
                        )
                    )
                    n = max(n, local_idx + 1)
            offset += n
        out.sort(key=lambda m: m.episode_index)
        return out

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dataset_paths(data_cfg: DataConfig) -> list[str]:
        """Resolve one or more LeRobot v2.x suite dirs from the data config.

        Accepts a list (cross-suite: all 4 LIBERO suites) or a single path.
        Order: ``data.extra['dataset_paths']`` (list) -> ``data.repo_id`` (str or
        list) -> ``data.extra['dataset_path']`` -> ``data.extra['root']``. Values
        are local paths (GR00T reads files from disk, not the HF hub).
        """
        extra = data_cfg.extra or {}
        paths = extra.get("dataset_paths")
        if not paths:
            single = data_cfg.repo_id or extra.get("dataset_path") or extra.get("root")
            if isinstance(single, (list, tuple)):
                paths = list(single)
            elif single:
                paths = [single]
        if not paths:
            raise ValueError(
                "GR00T backend requires dataset path(s) via data.repo_id "
                "(str or list), data.extra['dataset_paths'] (list), or "
                "data.extra['dataset_path']/['root'] pointing at LeRobot v2.x "
                "LIBERO suite dir(s)."
            )
        return [str(p) for p in paths]


def _load_modality_ranges(dataset_path):
    """Load per-key state/action index ranges from a dataset's meta/modality.json.

    Returns ({state_key: (start, end)}, {action_key: (start, end)}) — the slices
    of the flat observation.state / action vectors each modality key occupies
    (LIBERO: state x:(0,1)..yaw:(5,6),gripper:(6,8); action gripper:(6,7)).
    """
    import json as _json

    mj = Path(dataset_path) / "meta" / "modality.json"
    with mj.open() as f:
        m = _json.load(f)
    state = {k: (int(v["start"]), int(v["end"])) for k, v in m["state"].items()}
    action = {k: (int(v["start"]), int(v["end"])) for k, v in m["action"].items()}
    return state, action


def _count_episodes(dataset_path: str) -> int:
    """Number of episodes in a LeRobot v2.x suite (lines of meta/episodes.jsonl)."""
    p = Path(dataset_path) / "meta" / "episodes.jsonl"
    with p.open() as f:
        return sum(1 for line in f if line.strip())


def _suite_offsets(paths: list[str]) -> list[tuple[str, int, int]]:
    """Per-suite ``(path, global_start, n_episodes)`` in path order.

    The cumulative ``global_start`` is the SAME offset ``list_episodes`` uses to
    assign global episode indices, so ``global_index - global_start`` recovers the
    per-suite LOCAL episode index (which ``_FilteredEpisodeLoader`` filters on).
    LeRobot episode indices are contiguous 0..n-1, so the line count == n.
    """
    out: list[tuple[str, int, int]] = []
    start = 0
    for p in paths:
        n = _count_episodes(p)
        out.append((p, start, n))
        start += n
    return out


class _FlattenedShardDataset:
    """Map-style view over a shard-based ``ShardedSingleStepDataset``.

    GR00T's sharded dataset exposes ``__len__`` = number of *shards* and
    ``get_shard(i)`` = a list of fully-processed timesteps (each already passed
    through ``processor`` via ``get_datapoint``). To drive a deterministic
    ``torch.utils.data.DataLoader`` we present a flat index over *timesteps*:
    item ``i`` maps to ``(shard_idx, offset_in_shard)`` and ``__getitem__``
    loads that shard and returns the one timestep.

    A simple per-instance one-shard cache avoids re-decoding the same shard for
    consecutive indices (the DataLoader's sampler shuffles globally, so the
    cache hit rate depends on shuffle order; with ``num_workers>0`` each worker
    holds its own cache). This trades a little CPU for avoiding loading all
    shards into RAM at once.

    VERIFY ON GPU: ``get_shard`` decodes video for every timestep in the shard,
    so a cache miss is expensive. If throughput is poor, the cleaner option is
    to iterate the upstream ``ShardedMixtureDataset`` (IterableDataset) directly
    and wrap *that* in a DataLoader with ``batch_size`` set; this map-style
    flattener was chosen for deterministic, restartable per-round FL batches.
    """

    def __init__(self, sharded, lazy_datapoints: bool = False, episode_cache_size: int = 0):
        self._sharded = sharded
        # Eagerly build the (shard_idx, offset) index from per-shard lengths —
        # cheap (no data load), uses ShardedDataset.get_shard_length.
        self._index: list[tuple[int, int]] = []
        for shard_idx in range(len(sharded)):
            shard_len = int(sharded.get_shard_length(shard_idx))
            for offset in range(shard_len):
                self._index.append((shard_idx, offset))
        self._cache_shard_idx: int | None = None
        self._cache_shard: list | None = None

        # --- lazy-datapoint mode -------------------------------------------
        # get_shard() builds a shard's datapoints by walking its episodes in
        # order and, within each, its assigned step indices. Flattening that
        # same walk gives (episode, step) for every global index, matching
        # self._index position-for-position — so the dataset's ordering, and
        # therefore what `shuffle=True` draws, is byte-for-byte unchanged.
        self._lazy_datapoints = bool(lazy_datapoints)
        self._episode_cache_size = int(episode_cache_size)
        self._flat: list[tuple[int, int]] | None = None
        self._ep_cache: OrderedDict = OrderedDict()
        if self._lazy_datapoints:
            flat: list[tuple[int, int]] = []
            for shard_idx in range(len(sharded)):
                for ep_idx, step_indices in sharded.sharded_episodes[shard_idx]:
                    for step in step_indices:
                        flat.append((int(ep_idx), int(step)))
            if len(flat) != len(self._index):
                raise RuntimeError(
                    "lazy flat index disagrees with the shard index "
                    f"({len(flat)} vs {len(self._index)}) — refusing to train on a "
                    "dataset whose ordering would differ from the sharded path."
                )
            self._flat = flat

    def __getstate__(self):
        # Workers are spawned, so the dataset is pickled to each one. Never ship
        # a warm episode cache through that pipe (43 MB/episode).
        state = dict(self.__dict__)
        state["_ep_cache"] = OrderedDict()
        state["_cache_shard"] = None
        state["_cache_shard_idx"] = None
        return state

    def _episode(self, ep_idx: int):
        """Raw episode DataFrame for ``ep_idx``, cached per worker."""
        ep = self._ep_cache.get(ep_idx)
        if ep is not None:
            self._ep_cache.move_to_end(ep_idx)
            return ep
        ep = self._sharded.episode_loader[ep_idx]
        self._ep_cache[ep_idx] = ep
        if self._episode_cache_size and len(self._ep_cache) > self._episode_cache_size:
            self._ep_cache.popitem(last=False)
        return ep

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        if self._lazy_datapoints:
            assert self._flat is not None
            ep_idx, step = self._flat[i]
            return self._sharded.get_datapoint(self._episode(ep_idx), step)
        shard_idx, offset = self._index[i]
        if shard_idx != self._cache_shard_idx:
            # Load (and process) the whole shard, then cache it.
            self._cache_shard = self._sharded.get_shard(shard_idx)
            self._cache_shard_idx = shard_idx
        assert self._cache_shard is not None
        return self._cache_shard[offset]


def _shard_groups(combined) -> list[list[int]]:
    """Group ``combined``'s global indices by the shard each one comes from.

    ``combined`` is either a single ``_FlattenedShardDataset`` or a
    ``ConcatDataset`` of one per LIBERO suite; ``sub._index[i]`` is that
    sub-dataset's ``(shard_idx, offset)`` for local index ``i``.
    """
    from torch.utils.data import ConcatDataset

    subs = combined.datasets if isinstance(combined, ConcatDataset) else [combined]
    groups: list[list[int]] = []
    base = 0
    for sub in subs:
        by_shard: dict[int, list[int]] = {}
        for i, (shard_idx, _offset) in enumerate(sub._index):
            by_shard.setdefault(int(shard_idx), []).append(base + i)
        groups.extend(by_shard[k] for k in sorted(by_shard))
        base += len(sub)
    return groups


class _ShardGroupedBatchSampler:
    """Batch sampler that keeps every batch inside a single shard.

    ``_FlattenedShardDataset.__getitem__`` loads *and processes* an entire
    shard to return one timestep, keeping only that one shard cached. A LIBERO
    client partition is ~29 000 timesteps in ~30 shards of ~980, so under
    ``shuffle=True`` roughly 29 draws in 30 are cache misses and the full shard
    is decoded + processed once **per sample**: measured ~7.5 s/sample on the
    GR00T arms, with the GPU pinned at 0% util while the decode workers stayed
    CPU-bound. Worker count only ever bought a linear slice of that waste.

    Grouping each batch inside one shard, and emitting a shard's batches
    consecutively, makes one shard load serve ``shard_len / batch_size``
    batches (~30) instead of one sample. Shard order and within-shard order are
    redrawn every epoch from ``seed``, so batches stay deterministic and the
    run stays restartable.

    Trade-off: a batch's samples come from one shard rather than from the whole
    partition, so batches are no longer globally IID. A LIBERO shard is ~9-10
    whole episodes, so a batch still mixes episodes. Opt in per run with
    ``data.extra.shard_grouped_batches: true``; the default keeps the old
    globally-shuffled behaviour so in-flight arms are unaffected.
    """

    def __init__(self, groups, batch_size: int, seed: int = 0, drop_last: bool = False):
        self._groups = [list(g) for g in groups if g]
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self._epoch = 0
        if self._batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        self._len = sum(
            (len(g) // self._batch_size)
            if self._drop_last
            else -(-len(g) // self._batch_size)
            for g in self._groups
        )

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        import numpy as np

        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        for group_idx in rng.permutation(len(self._groups)):
            idx = np.asarray(self._groups[group_idx], dtype=np.int64)
            rng.shuffle(idx)
            for start in range(0, len(idx), self._batch_size):
                batch = idx[start : start + self._batch_size].tolist()
                if self._drop_last and len(batch) < self._batch_size:
                    continue
                yield batch


def _cycle(iterable):
    """Infinite iterator over a DataLoader (re-iterates on exhaustion).

    Inlined (mirrors the lerobot adapter) so the round loop can pull
    ``local_steps`` batches without worrying about epoch boundaries.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
