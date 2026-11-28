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

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..common import MASMessage
from .memco_backend import MemCoOnlineEpisodeBuilder, build_memco_prompt_payload, rank_messages_for_query
from .memory_base import MASMemoryBase


@dataclass
class MemCoBase(MASMemoryBase):
    """Minimal MemCo backend integrated into the current MAS memory lifecycle.

    Phase 1 goals:
    - dynamic per-episode overlay via move_memory_state()
    - per-task committed memory via save_task_context()/add_memory()
    - structured prompt payload for MAS workflows
    """

    committed_messages: list[MASMessage] = field(default_factory=list, init=False)
    episode_builder: MemCoOnlineEpisodeBuilder | None = field(default=None, init=False)
    insight_bank: list[str] = field(default_factory=list, init=False)
    refresh_each_step: bool = field(default=False, init=False)
    _external_adapter: Any = field(default=None, init=False, repr=False)
    _external_adapters: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _external_retriever: Any = field(default=None, init=False, repr=False)
    _external_local_memories: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _external_global_memory: Any = field(default=None, init=False, repr=False)
    _external_empty_local: Any = field(default=None, init=False, repr=False)
    _external_empty_global: Any = field(default=None, init=False, repr=False)
    _external_memory_query_type: Any = field(default=None, init=False, repr=False)
    _external_candidate_type: Any = field(default=None, init=False, repr=False)
    _external_builder: Any = field(default=None, init=False, repr=False)
    _external_maintainer: Any = field(default=None, init=False, repr=False)
    _external_promoter: Any = field(default=None, init=False, repr=False)
    _external_local_to_dict: Any = field(default=None, init=False, repr=False)
    _external_global_to_dict: Any = field(default=None, init=False, repr=False)
    _external_load_local_memory: Any = field(default=None, init=False, repr=False)
    _external_load_global_memory: Any = field(default=None, init=False, repr=False)
    _external_artifact_dir: Path | None = field(default=None, init=False, repr=False)
    _external_shared_global_dir: Path | None = field(default=None, init=False, repr=False)
    _external_retrieval_mode: str = field(default="lightweight", init=False)
    _dynamic_graph_enabled: bool = field(default=False, init=False)
    _external_enabled: bool = field(default=False, init=False)
    _external_error: str = field(default="", init=False)
    _memco_core_episode_global_skeleton: list[str] = field(default_factory=list, init=False, repr=False)
    _memco_core_episode_global_skeleton_key: str = field(default="", init=False, repr=False)
    _memco_core_feedback_path: Path | None = field(default=None, init=False, repr=False)
    _memco_core_feedback_stats: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _memco_core_episode_feedback_items: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _memco_core_quality_last_action_candidates: list[str] = field(default_factory=list, init=False, repr=False)
    _memco_core_quality_source_prior_candidates: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _memco_core_graph_policy_source_key: str = field(default="", init=False, repr=False)
    _memco_core_graph_policy_source_queue: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _memco_core_graph_policy_next_source_action: str = field(default="", init=False, repr=False)
    _memco_core_debug_trace_path: Path | None = field(default=None, init=False, repr=False)
    _memco_core_debug_current_task: str = field(default="", init=False, repr=False)
    _memco_core_debug_last_step: int = field(default=0, init=False, repr=False)
    _memco_core_debug_env_step: int = field(default=0, init=False, repr=False)
    _graph_config_warnings: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.namespace = self.namespace or "memco"
        self._records_path = os.path.join(self.persist_dir, "episodes.jsonl")
        self._insights_path = os.path.join(self.persist_dir, "insights.json")
        self.freeze_memory: bool = bool(self._graph_config_value("freeze_memory", False))
        self.enable_overlay: bool = bool(self._graph_config_value("enable_overlay", True))
        self.reset_memory: bool = bool(self._graph_config_value("reset_memory", False))
        self.promotion_threshold: float = float(self._graph_config_value("promotion_threshold", 0.35))
        self.promotion_policy: str = (
            str(self._graph_config_value("promotion_policy", "legacy") or "legacy").strip().lower()
        )
        self.wilson_alpha: float = float(self._graph_config_value("wilson_alpha", 0.05))
        self.wilson_threshold: float = float(self._graph_config_value("wilson_threshold", 0.35))
        if self.reset_memory:
            self._overwrite_lightweight_memory_files()
        self.committed_messages = self._load_messages()
        self.insight_bank = self._load_insights()
        self._init_external_graph_memory()

    def _graph_config_prefix(self) -> str:
        return "memco"

    def _warn_legacy_graph_config(self, legacy_key: str, new_key: str) -> None:
        if legacy_key in self._graph_config_warnings:
            return
        self._graph_config_warnings.add(legacy_key)
        print(
            f"[{self.namespace}] legacy config `{legacy_key}` detected; prefer `{new_key}`.",
            flush=True,
        )

    def _graph_config_value(self, suffix: str, default: Any = None) -> Any:
        prefix = self._graph_config_prefix()
        primary_key = f"{prefix}_{suffix}"
        if primary_key in self.global_config:
            return self.global_config.get(primary_key)
        return default

    def init_task_context(self, task_main: str, task_description: str = None) -> MASMessage:
        message = super().init_task_context(task_main, task_description)
        self.episode_builder = MemCoOnlineEpisodeBuilder.from_task(task_main, task_description or "")
        self._memco_core_episode_global_skeleton = []
        self._memco_core_episode_global_skeleton_key = ""
        self._memco_core_episode_feedback_items = {}
        self._memco_core_quality_last_action_candidates = []
        self._memco_core_graph_policy_source_key = ""
        self._memco_core_graph_policy_source_queue = []
        self._memco_core_graph_policy_next_source_action = ""
        self._memco_core_quality_source_prior_candidates = []
        self._memco_core_debug_current_task = str(task_main or "")
        self._memco_core_debug_last_step = 0
        self._memco_core_debug_env_step = 0
        return message

    def move_memory_state(self, action: str, observation: str, **kargs) -> None:
        super().move_memory_state(action, observation, **kargs)
        self._memco_core_debug_env_step += 1
        self._memco_core_debug_append(
            "env_feedback",
            step_index=self._memco_core_debug_env_step,
            payload={
                "action": str(action or ""),
                "reward": kargs.get("reward"),
                "observation": self._memco_core_debug_text(str(observation or ""), limit=1200),
            },
        )
        if self.enable_overlay and self.episode_builder is not None:
            self.episode_builder.update(action, observation)

    def retrieve_memory(self, **kargs) -> tuple[list[MASMessage], list[MASMessage], list[str]]:
        query_task = str(kargs.get("query_task") or "")
        successful_topk = int(kargs.get("successful_topk", 1) or 0)
        failed_topk = int(kargs.get("failed_topk", 0) or 0)
        insight_topk = int(kargs.get("insight_topk", kargs.get("insights_topk", 3)) or 0)
        successful = rank_messages_for_query(
            query_task,
            self.committed_messages,
            topk=successful_topk,
            label=True,
        )
        failed = rank_messages_for_query(
            query_task,
            self.committed_messages,
            topk=failed_topk,
            label=False,
        )
        insights = list(self.insight_bank[: max(insight_topk, 0)])
        return successful, failed, insights

    def retrieve_prompt_payload(self, **kargs) -> dict[str, list[str]]:
        if self._external_enabled:
            return self._retrieve_external_prompt_payload(**kargs)
        successful, failed, insights = self.retrieve_memory(**kargs)
        payload = build_memco_prompt_payload(
            successful_messages=successful,
            failed_messages=failed,
            overlay_builder=self.episode_builder if self.enable_overlay else None,
            stored_insights=insights,
        )
        return payload.to_dict()

    def _init_external_graph_memory(self) -> None:
        memory_dir = str(self._graph_config_value("memory_dir", "") or "").strip()
        dynamic_graph = bool(self._graph_config_value("dynamic_graph", False))
        if not memory_dir and not dynamic_graph:
            return
        memory_path = Path(memory_dir).expanduser() if memory_dir else Path(self.persist_dir)
        local_artifact_scene = self._memco_core_local_artifact_scene(memory_path)
        if memory_dir and not memory_path.exists():
            self._external_error = f"external {self._graph_config_prefix()}_memory_dir not found: {memory_path}"
            return

        try:
            from .memco_backend.alfworld_adapter import ALFWorldAdapter
            from .memco_backend.build_memory_graph import _global_to_dict, _local_to_dict
            from .memco_backend.construction_graph import EpisodeGraphBuilder, GlobalPromoter, LocalGraphMaintainer
            from .memco_backend.fever_adapter import FeverAdapter
            from .memco_backend.graph_types import CandidateType, GlobalGraphMemory, LocalGraphMemory, MemoryQuery
            from .memco_backend.pddl_adapter import PDDLAdapter
            from .memco_backend.retrieval_graph import QueryBasedRetriever
            from .memco_backend.serialization import empty_global, empty_local, load_global_memory, load_local_memory
        except Exception as exc:
            self._external_error = (
                "failed to import vendored memco_backend graph modules. " f"{type(exc).__name__}: {exc}"
            )
            return

        mode = str(self._graph_config_value("retrieval_mode", "lightweight") or "lightweight")
        self._external_retrieval_mode = mode
        self._memco_core_feedback_path = Path(self.persist_dir) / "feedback_stats.json"
        if mode in {"graph_policy_feedback", "graph_policy_quality"}:
            if self.reset_memory:
                self._memco_core_feedback_stats = {"version": 1, "items": {}}
                self._save_memco_core_feedback_stats()
            else:
                self._memco_core_feedback_stats = self._load_memco_core_feedback_stats()
        self._external_retriever = QueryBasedRetriever(top_k=5)

        self._external_adapters = {
            "alfworld": ALFWorldAdapter(),
            "pddl": PDDLAdapter(),
            "fever": FeverAdapter(),
        }
        self._external_adapter = self._external_adapters.get(
            self._infer_external_domain(), self._external_adapters["alfworld"]
        )
        self._external_empty_local = empty_local
        self._external_empty_global = empty_global
        self._external_memory_query_type = MemoryQuery
        self._external_candidate_type = CandidateType
        self._external_builder = EpisodeGraphBuilder()
        self._external_maintainer = LocalGraphMaintainer(
            record_wilson_evidence=self.promotion_policy in {"shadow", "wilson"}
        )
        self._external_promoter = GlobalPromoter(
            score_threshold=self.promotion_threshold,
            policy=self.promotion_policy,
            wilson_alpha=self.wilson_alpha,
            wilson_threshold=self.wilson_threshold,
        )
        self._external_local_to_dict = _local_to_dict
        self._external_global_to_dict = _global_to_dict
        self._external_load_local_memory = load_local_memory
        self._external_load_global_memory = load_global_memory
        self._external_artifact_dir = Path(self.persist_dir)
        self._memco_core_debug_trace_path = Path(self.persist_dir) / "memco_debug_trace.jsonl"
        self._dynamic_graph_enabled = dynamic_graph
        shared_global_dir = str(self._graph_config_value("shared_global_dir", "") or "").strip()
        if shared_global_dir:
            shared_path = Path(shared_global_dir).expanduser()
            if shared_path.name != self.namespace:
                shared_path = shared_path / self.namespace
            shared_path.mkdir(parents=True, exist_ok=True)
            self._external_shared_global_dir = shared_path

        def _memory_search_roots(path: Path) -> list[Path]:
            roots: list[Path] = []

            def add(candidate: Path) -> None:
                if candidate not in roots:
                    roots.append(candidate)

            add(path)
            for candidate in (path, *path.parents):
                if candidate.name == self.namespace:
                    add(candidate)
                    # Full nvdamas runs store MemCo artifacts as:
                    #   .../memory/memco/local/<scene>/memco/local_<scene>.json
                    #   .../memory/memco/global/memco/global_memory.json
                    if candidate.parent.name in {"local", "global"}:
                        add(candidate.parent.parent)
            return roots

        def _iter_local_memory_files(path: Path) -> list[Path]:
            files: dict[str, Path] = {}
            for root in _memory_search_roots(path):
                for local_file in sorted(root.glob("local_*.json")):
                    files.setdefault(str(local_file), local_file)
                for local_file in sorted(root.glob(f"local/*/{self.namespace}/local_*.json")):
                    files.setdefault(str(local_file), local_file)
            return list(files.values())

        def _find_global_memory_file(path: Path) -> Path | None:
            for root in _memory_search_roots(path):
                direct = root / "global_memory.json"
                if direct.exists():
                    return direct
                nested = root / "global" / self.namespace / "global_memory.json"
                if nested.exists():
                    return nested
            return None

        if dynamic_graph:
            if not self.reset_memory:
                for local_file in _iter_local_memory_files(memory_path):
                    scene = local_file.stem[len("local_") :]
                    if local_artifact_scene and scene != local_artifact_scene:
                        continue
                    try:
                        self._external_local_memories[scene] = load_local_memory(str(local_file))
                    except Exception:
                        continue
                global_file = _find_global_memory_file(memory_path)
                if global_file is not None and global_file.exists():
                    try:
                        self._external_global_memory = load_global_memory(str(global_file))
                    except Exception:
                        self._external_global_memory = GlobalGraphMemory()
                else:
                    self._external_global_memory = GlobalGraphMemory()
            else:
                self._external_global_memory = GlobalGraphMemory()
            self._refresh_shared_global_memory()
            owner_scene = str(self._graph_config_value("owner_scene", "") or "").strip()
            if owner_scene and owner_scene not in self._external_local_memories:
                self._external_local_memories[owner_scene] = LocalGraphMemory(agent_id=f"agent_{owner_scene}")
            if self.reset_memory:
                self._persist_dynamic_graph_memory()
        else:
            for local_file in _iter_local_memory_files(memory_path):
                scene = local_file.stem[len("local_") :]
                if local_artifact_scene and scene != local_artifact_scene:
                    continue
                try:
                    self._external_local_memories[scene] = load_local_memory(str(local_file))
                except Exception:
                    continue

            global_file = _find_global_memory_file(memory_path)
            if global_file is not None and global_file.exists():
                try:
                    self._external_global_memory = load_global_memory(str(global_file))
                except Exception:
                    self._external_global_memory = _empty_global()
            else:
                self._external_global_memory = _empty_global()

        self._external_enabled = True
        self.refresh_each_step = True

    def _memco_core_local_artifact_scene(self, path: Path) -> str:
        """Return the scene owned by a local graph-memory artifact directory.

        Full collab runs store per-scene memory under:
            .../<memory_name>/local/<scene>/<memory_name>/

        Workers should not load or persist other scenes' local_*.json files inside
        this directory. Older runs may contain such stale cross-scene copies, and
        loading them by scene name can overwrite the real local graph.
        """
        try:
            if path.name == self.namespace and path.parent.parent.name == "local":
                return path.parent.name
        except Exception:
            return ""
        return ""

    def _refresh_shared_global_memory(self) -> None:
        if self._external_shared_global_dir is None:
            return
        global_file = self._external_shared_global_dir / "global_memory.json"
        if global_file.exists() and callable(self._external_load_global_memory):
            try:
                self._external_global_memory = self._external_load_global_memory(str(global_file))
                return
            except Exception:
                pass
        try:
            from .memco_backend.graph_types import GlobalGraphMemory

            self._external_global_memory = GlobalGraphMemory()
        except Exception:
            self._external_global_memory = self._external_empty_global()

    def _wilson_promotion_locals(self, scene: str, current_memory: Any) -> list[Any]:
        """Return one completed local snapshot per available source scope.

        Wilson promotion is recomputed from local evidence, rather than
        incrementally scoring only the worker that happened to finish last.
        This is intentionally isolated from legacy and shadow execution.
        """

        memories: dict[str, Any] = dict(self._external_local_memories)
        if self._external_shared_global_dir is not None and callable(self._external_load_local_memory):
            memory_root = self._external_shared_global_dir.parent.parent
            pattern = f"local/*/{self.namespace}/local_*.json"
            for local_file in sorted(memory_root.glob(pattern)):
                source_scene = local_file.stem[len("local_") :]
                if not source_scene or source_scene == scene:
                    continue
                try:
                    memories[source_scene] = self._external_load_local_memory(str(local_file))
                except Exception:
                    # A concurrently published local file can be temporarily
                    # unavailable; the next episode and the final rebuild retry.
                    continue
        memories[scene] = current_memory
        return [memories[key] for key in sorted(memories)]

    def _persist_shared_global_memory(self) -> None:
        if self._external_shared_global_dir is None:
            return
        self._external_shared_global_dir.mkdir(parents=True, exist_ok=True)
        promoted_batches = self._dedupe_promoted_batches(
            list(getattr(self._external_global_memory, "promoted_batches", []) or [])
        )
        try:
            self._external_global_memory.promoted_batches = promoted_batches
        except Exception:
            pass
        global_path = self._external_shared_global_dir / "global_memory.json"
        self._write_committed_json(
            global_path,
            self._external_global_to_dict(self._external_global_memory),
        )
        self._persist_wilson_promotion_outputs(self._external_shared_global_dir)
        summary = {
            "mode": "strict_online_shared_global",
            "global": {
                "candidate_count": len(getattr(self._external_global_memory, "candidates", {})),
                "rule_count": len(getattr(self._external_global_memory, "rules_by_id", {})),
                "artifact_count": len(getattr(self._external_global_memory, "artifacts_by_id", {})),
                "promoted_batches": promoted_batches,
            },
        }
        with (self._external_shared_global_dir / "summary.json").open("w", encoding="utf-8") as writer:
            json.dump(summary, writer, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_committed_json(path: Path, payload: Any) -> None:
        """Publish a complete JSON snapshot without exposing a partial write."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as writer:
                temporary_path = Path(writer.name)
                json.dump(payload, writer, ensure_ascii=False, indent=2)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _persist_wilson_promotion_outputs(self, output_dir: Path) -> None:
        promoter = self._external_promoter
        report = getattr(promoter, "last_promotion_report", None)
        if not report:
            return
        with (output_dir / "promotion_wilson.json").open("w", encoding="utf-8") as writer:
            json.dump(report, writer, ensure_ascii=False, indent=2)
        if self.promotion_policy != "shadow":
            return
        shadow_memory = getattr(promoter, "last_wilson_memory", None)
        if shadow_memory is None:
            return
        with (output_dir / "global_memory_wilson.json").open("w", encoding="utf-8") as writer:
            json.dump(
                self._external_global_to_dict(shadow_memory),
                writer,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _dedupe_promoted_batches(batches: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for batch in batches:
            key = str(batch or "")
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _memco_core_debug_enabled(self) -> bool:
        return bool(
            self._external_enabled
            and self._memco_core_debug_trace_path is not None
            and self._external_retrieval_mode
            in {
                "graph_policy",
                "graph_policy_rerank",
                "graph_policy_feedback",
                "graph_policy_candidate",
                "graph_policy_quality",
            }
        )

    def _memco_core_debug_text(self, text: str, *, limit: int = 800) -> str:
        value = str(text or "")
        if limit <= 0 or len(value) <= limit:
            return value
        return value[:limit] + f"...[truncated {len(value) - limit}]"

    def _memco_core_debug_jsonable(self, value: Any, *, depth: int = 0) -> Any:
        if depth > 4:
            return self._memco_core_debug_text(str(value), limit=300)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= 40:
                    out["..."] = f"{len(value) - idx} more"
                    break
                out[str(key)] = self._memco_core_debug_jsonable(item, depth=depth + 1)
            return out
        if isinstance(value, (list, tuple, set)):
            seq = list(value)
            out = [self._memco_core_debug_jsonable(item, depth=depth + 1) for item in seq[:40]]
            if len(seq) > 40:
                out.append(f"... {len(seq) - 40} more")
            return out
        return self._memco_core_debug_text(str(value), limit=500)

    def _memco_core_debug_query_snapshot(self, query: Any) -> dict[str, Any]:
        if query is None:
            return {}
        return {
            "goal": str(getattr(query, "goal", "") or ""),
            "task_family": str(getattr(query, "task_family", "") or ""),
            "scene_id": str(getattr(query, "scene_id", "") or ""),
            "location": str(getattr(query, "location", "") or ""),
            "current_stage": str(getattr(query, "current_stage", "") or ""),
            "progress_state": str(getattr(query, "progress_state", "") or ""),
            "progress_hint": str(getattr(query, "progress_hint", "") or ""),
            "required_count": getattr(query, "required_count", None),
            "held_relevant_count": getattr(query, "held_relevant_count", None),
            "placed_relevant_count": getattr(query, "placed_relevant_count", None),
            "remaining_relevant_count": getattr(query, "remaining_relevant_count", None),
            "goal_roles": self._memco_core_debug_jsonable(getattr(query, "goal_roles", {}) or {}),
            "dynamic_context": self._memco_core_debug_jsonable(getattr(query, "dynamic_context", {}) or {}),
        }

    def _memco_core_debug_item_snapshot(self, item: Any) -> dict[str, Any]:
        dynamic = getattr(item, "dynamic", {}) or {}
        if not isinstance(dynamic, dict):
            dynamic = {"value": str(dynamic)}
        payload = dynamic.get("payload", {}) if isinstance(dynamic.get("payload", {}), dict) else {}
        dynamic_keys = (
            "relation_kind",
            "artifact_kind",
            "object_role",
            "source_instance",
            "source_type",
            "location",
            "target",
            "container",
        )
        compact_dynamic = {
            key: dynamic.get(key) for key in dynamic_keys if key in dynamic and dynamic.get(key) not in (None, "")
        }
        for key in dynamic_keys:
            if key in payload and payload.get(key) not in (None, ""):
                compact_dynamic[f"payload.{key}"] = payload.get(key)
        graph_refs = payload.get("graph_refs", [])
        if graph_refs:
            compact_dynamic["payload.graph_refs"] = list(graph_refs)[:5]
        action_patterns = [
            str(pattern) for pattern in (getattr(item, "action_patterns", ()) or ()) if str(pattern).strip()
        ]
        candidate_type = getattr(item, "candidate_type", "")
        candidate_type_text = getattr(candidate_type, "value", candidate_type)
        return {
            "source": str(getattr(item, "source", "") or ""),
            "candidate_id": str(getattr(item, "candidate_id", "") or ""),
            "candidate_type": str(candidate_type_text or ""),
            "pattern_kind": str(getattr(item, "pattern_kind", "") or ""),
            "branch_tag": str(getattr(item, "branch_tag", "") or ""),
            "score": float(getattr(item, "score", 0.0) or 0.0),
            "task_relevance": float(getattr(item, "task_relevance", 0.0) or 0.0),
            "goal_relevance": float(getattr(item, "goal_relevance", 0.0) or 0.0),
            "state_relevance": float(getattr(item, "state_relevance", 0.0) or 0.0),
            "positive": int(getattr(item, "positive", 0) or 0),
            "negative": int(getattr(item, "negative", 0) or 0),
            "summary": self._memco_core_debug_text(str(getattr(item, "summary", "") or ""), limit=700),
            "action_patterns": action_patterns[:5],
            "dynamic": self._memco_core_debug_jsonable(compact_dynamic),
        }

    def _memco_core_debug_items(self, bundle: Any, field_name: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return [
            self._memco_core_debug_item_snapshot(item) for item in list(getattr(bundle, field_name, []) or [])[:limit]
        ]

    def _memco_core_debug_bundle_snapshot(self, bundle: Any) -> dict[str, Any]:
        if bundle is None:
            return {}
        return {
            "routing_weights": self._memco_core_debug_jsonable(getattr(bundle, "routing_weights", {}) or {}),
            "routing_decisions": self._memco_core_debug_jsonable(getattr(bundle, "routing_decisions", {}) or {}),
            "local": {
                "local_items": self._memco_core_debug_items(bundle, "local_items"),
                "fact_items": self._memco_core_debug_items(bundle, "fact_items"),
                "relation_items": self._memco_core_debug_items(bundle, "relation_items"),
                "local_graph_contribution": self._memco_core_debug_items(bundle, "local_graph_contribution"),
                "local_promoted_contribution": self._memco_core_debug_items(bundle, "local_promoted_contribution"),
                "plan_items": self._memco_core_debug_items(bundle, "plan_items"),
                "workflow_items": [
                    item
                    for item in self._memco_core_debug_items(bundle, "workflow_items")
                    if not str(item.get("source", "")).startswith("global")
                ],
            },
            "global": {
                "global_items": self._memco_core_debug_items(bundle, "global_items"),
                "global_task_plan_items": self._memco_core_debug_items(bundle, "global_task_plan_items"),
                "global_promoted_contribution": self._memco_core_debug_items(bundle, "global_promoted_contribution"),
                "workflow_items": [
                    item
                    for item in self._memco_core_debug_items(bundle, "workflow_items")
                    if str(item.get("source", "")).startswith("global")
                ],
                "precondition_items": [
                    item
                    for item in self._memco_core_debug_items(bundle, "precondition_items")
                    if str(item.get("source", "")).startswith("global")
                ],
            },
            "policy_outputs": {
                "suggested_actions": self._memco_core_debug_jsonable(getattr(bundle, "suggested_actions", []) or []),
                "blocked_actions": self._memco_core_debug_jsonable(getattr(bundle, "blocked_actions", []) or []),
                "warnings": self._memco_core_debug_jsonable(getattr(bundle, "warnings", []) or []),
                "workflow_hints": self._memco_core_debug_jsonable(getattr(bundle, "workflow_hints", []) or []),
            },
        }

    def _memco_core_debug_memory_counts(self, memory: Any) -> dict[str, int]:
        if memory is None:
            return {}
        return {
            "episodes": len(getattr(memory, "episode_ids", []) or []),
            "candidates": len(getattr(memory, "candidates", {}) or {}),
            "rules": len(getattr(memory, "rules_by_id", {}) or {}),
            "artifacts": len(getattr(memory, "artifacts_by_id", {}) or {}),
            "nodes": len(getattr(memory, "nodes_by_signature", {}) or {}),
            "edges": len(getattr(memory, "edges_by_signature", {}) or {}),
        }

    def _memco_core_debug_append(
        self,
        event: str,
        *,
        step_index: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._memco_core_debug_enabled():
            return
        try:
            path = self._memco_core_debug_trace_path
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "event": str(event),
                "step_index": int(step_index or 0),
                "task": self._memco_core_debug_current_task,
                "retrieval_mode": self._external_retrieval_mode,
                "freeze_memory": bool(getattr(self, "freeze_memory", False)),
                "payload": self._memco_core_debug_jsonable(payload or {}),
            }
            with path.open("a", encoding="utf-8") as writer:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _retrieve_external_prompt_payload(self, **kargs) -> dict[str, list[str]]:
        if not self._external_enabled:
            return {"reference_cases": [], "execution_patterns": [], "insights": []}
        query = self._build_external_query(**kargs)
        step_index = int(kargs.get("step_index", 0) or 0)
        self._memco_core_debug_last_step = step_index
        setting = str(self._graph_config_value("settings", "local_only") or "local_only")
        trajectory_payload = None
        if setting not in {"base", "global_only"}:
            successful, failed, insights = self.retrieve_memory(**kargs)
            trajectory_payload = build_memco_prompt_payload(
                successful_messages=successful,
                failed_messages=failed,
                overlay_builder=None,
                stored_insights=insights,
            )
        if query is None:
            note = self._external_error or "external MemCo query unavailable for this task state."
            self._memco_core_debug_append(
                "retrieve_unavailable",
                step_index=step_index,
                payload={"note": note, "setting": setting},
            )
            return {
                "reference_cases": [],
                "execution_patterns": list(trajectory_payload.execution_patterns) if trajectory_payload else [],
                "insights": list(trajectory_payload.insights) if trajectory_payload else [],
                "planner_notes": [f"[MemCoExternal] {note}"],
                "action_constraints": [],
                "repair_hints": list(trajectory_payload.repair_hints) if trajectory_payload else [],
            }

        owner_scene = self._resolve_external_owner_scene(kargs.get("task_config"), kargs.get("env_ref"))
        local_memory = (
            self._external_empty_local(owner_scene)
            if setting in {"base", "global_only"}
            else self._external_local_memories.get(owner_scene, self._external_empty_local(owner_scene))
        )
        if setting not in {"base", "local_only"}:
            self._refresh_shared_global_memory()
        suppress_global_for_second_search = self._memco_core_is_two_object_second_search(
            query
        ) and self._external_retrieval_mode not in {"graph_policy", "graph_policy_rerank", "graph_policy_quality"}
        global_memory = (
            self._external_empty_global()
            if setting in {"base", "local_only"} or suppress_global_for_second_search
            else self._external_global_memory
        )

        bundle = self._external_retriever.retrieve(query, local_memory, global_memory)
        state = query.dynamic_context or {}
        if self._external_retrieval_mode == "lightweight":
            support_text = self._render_graph_aware_lightweight_evidence(
                query=query,
                bundle=bundle,
                env_ref=kargs.get("env_ref"),
            ).strip()
        elif self._external_retrieval_mode == "graph_policy_quality":
            # graph_policy_quality renders memory evidence inside _phasee_policy_prompt_payload.
            # Keeping support_text empty avoids duplicating routed evidence in the final prompt.
            support_text = ""
        elif self._external_retrieval_mode in {
            "graph_policy",
            "graph_policy_rerank",
            "graph_policy_feedback",
            "graph_policy_candidate",
        }:
            support_text = self._render_graph_policy_evidence(
                query=query,
                bundle=bundle,
                env_ref=kargs.get("env_ref"),
                local_memory=local_memory,
                global_memory=global_memory,
            ).strip()
        else:
            support_text = bundle.to_planner_text(
                max_items=4,
                current_location=str(query.location or ""),
                exhausted_locations=tuple(state.get("exhausted_locations", ()) or ()),
                visible_objects=tuple(state.get("visible_objects", ()) or ()),
                held_objects=tuple(state.get("held_objects", ()) or ()),
                task_family=str(query.task_family or ""),
            ).strip()
        phasee_payload = self._phasee_policy_prompt_payload(
            query=query,
            bundle=bundle,
            env_ref=kargs.get("env_ref"),
        )
        planner_notes: list[str] = []
        if support_text:
            planner_notes.append(
                "### MemCo MEMORY EVIDENCE\n"
                "Treat this as supplementary evidence from the MemCo retriever. "
                "Current observation and admissible actions have priority.\n" + support_text
            )
        planner_notes.extend(phasee_payload.get("planner_notes", []))
        action_constraints = phasee_payload.get("action_constraints", [])
        repair_hints = phasee_payload.get("repair_hints", [])
        execution_patterns = list(trajectory_payload.execution_patterns) if trajectory_payload else []
        insights = list(trajectory_payload.insights) if trajectory_payload else []
        repair_hints = (list(trajectory_payload.repair_hints) if trajectory_payload else []) + list(repair_hints)
        if (
            not execution_patterns
            and not insights
            and not planner_notes
            and not action_constraints
            and not repair_hints
        ):
            self._memco_core_debug_append(
                "retrieve",
                step_index=step_index,
                payload={
                    "query": self._memco_core_debug_query_snapshot(query),
                    "setting": setting,
                    "owner_scene": owner_scene,
                    "local_memory_counts": self._memco_core_debug_memory_counts(local_memory),
                    "global_memory_counts": self._memco_core_debug_memory_counts(global_memory),
                    "bundle": self._memco_core_debug_bundle_snapshot(bundle),
                    "rendered_prompt_sections": {},
                    "source_priority_candidates": self._memco_core_debug_jsonable(
                        self._memco_core_quality_source_prior_candidates
                    ),
                    "source_priority_queue": self._memco_core_debug_jsonable(
                        self._memco_core_graph_policy_source_queue
                    ),
                    "next_source_action": self._memco_core_graph_policy_next_source_action,
                },
            )
            return {
                "reference_cases": [],
                "execution_patterns": [],
                "insights": [],
                "planner_notes": [],
                "action_constraints": [],
                "repair_hints": [],
            }
        payload = {
            "reference_cases": [],
            "execution_patterns": execution_patterns,
            "insights": insights,
            "planner_notes": planner_notes,
            "action_constraints": action_constraints,
            "repair_hints": repair_hints,
        }
        self._memco_core_debug_append(
            "retrieve",
            step_index=step_index,
            payload={
                "query": self._memco_core_debug_query_snapshot(query),
                "setting": setting,
                "owner_scene": owner_scene,
                "local_memory_counts": self._memco_core_debug_memory_counts(local_memory),
                "global_memory_counts": self._memco_core_debug_memory_counts(global_memory),
                "bundle": self._memco_core_debug_bundle_snapshot(bundle),
                "rendered_prompt_sections": {
                    "support_text": self._memco_core_debug_text(support_text, limit=3000),
                    "planner_notes": [self._memco_core_debug_text(item, limit=2500) for item in planner_notes[:5]],
                    "action_constraints": [
                        self._memco_core_debug_text(item, limit=1000) for item in action_constraints[:8]
                    ],
                    "repair_hints": [self._memco_core_debug_text(item, limit=1000) for item in repair_hints[:8]],
                    "execution_patterns": [
                        self._memco_core_debug_text(item, limit=1000) for item in execution_patterns[:5]
                    ],
                    "insights": [self._memco_core_debug_text(item, limit=1000) for item in insights[:5]],
                    "counts": {
                        "planner_notes": len(planner_notes),
                        "action_constraints": len(action_constraints),
                        "repair_hints": len(repair_hints),
                        "execution_patterns": len(execution_patterns),
                        "insights": len(insights),
                    },
                },
                "source_priority_candidates": self._memco_core_debug_jsonable(
                    self._memco_core_quality_source_prior_candidates
                ),
                "source_priority_queue": self._memco_core_debug_jsonable(self._memco_core_graph_policy_source_queue),
                "next_source_action": self._memco_core_graph_policy_next_source_action,
            },
        )
        return payload

    def _phasee_policy_prompt_payload(self, *, query: Any, bundle: Any, env_ref: Any) -> dict[str, list[str]]:
        """Render graph_policy_quality memory evidence as prompt-only guidance.

        Only graph_policy_quality is handled here (memory evidence without a
        state-policy overlay). All other phasee modes return empty results.
        """
        if self._external_retrieval_mode != "graph_policy_quality" or env_ref is None:
            return {"planner_notes": [], "action_constraints": [], "repair_hints": []}
        try:
            context = query.dynamic_context or {}
            memory_support = bundle.to_planner_text(
                max_items=4,
                current_location=str(query.location or ""),
                exhausted_locations=tuple(context.get("exhausted_locations", ()) or ()),
                visible_objects=tuple(context.get("visible_objects", ()) or ()),
                held_objects=tuple(context.get("held_objects", ()) or ()),
                task_family=str(query.task_family or ""),
            ).strip()
            self._record_phasee_quality_feedback_items(bundle)
            sections: list[str] = []
            if memory_support:
                sections.append(
                    "### MemCo STATIC MEMORY EVIDENCE\n"
                    "Use this as supplementary graph memory evidence, not as a direct command. "
                    "Prefer evidence that matches the current goal roles, held objects, visible objects, "
                    "and admissible actions.\n" + memory_support
                )
            sections.append(
                "### MemCo POLICY ROUTING\n"
                "Use local graph evidence for current-state grounding. Use global evidence only as "
                "abstract task workflow. Ignore scene-specific locations unless they are visible or "
                "admissible now. Failed-memory evidence is cautionary, not an instruction to repeat it."
            )
            planner_notes: list[str] = []
            if sections:
                planner_notes.append(
                    "### MemCo GRAPHPOLICY QUALITY VIEW"
                    "\nPrompt-only graph memory policy view. It does not replace the nvdamas AutoGen workflow.\n"
                    + "\n\n".join(sections)
                )
            return {
                "planner_notes": planner_notes,
                "action_constraints": [],
                "repair_hints": [],
            }
        except Exception as exc:
            self._external_error = f"graph_policy_quality prompt payload skipped: {type(exc).__name__}: {exc}"
            return {"planner_notes": [], "action_constraints": [], "repair_hints": []}

    def repair_action(
        self,
        *,
        raw_response: str,
        processed_action: str,
        env_ref: Any,
        task_config: dict | None = None,
        step_index: int = 0,
    ) -> str:
        """Optionally repair invalid ALFWorld actions before env.step().

        This is intentionally enabled only for explicit action/repair modes.
        Existing lightweight / hybrid_policy runs remain prompt-only.
        """
        action_modes = {
            "hybrid_repair",
            "lightweight_repair",
            "graph_policy",
            "graph_policy_rerank",
            "graph_policy_feedback",
            "graph_policy_candidate",
            "graph_policy_quality",
        }
        if self._external_retrieval_mode not in action_modes or not self._external_enabled:
            return processed_action
        admissible = [str(cmd) for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()]
        if not admissible:
            return processed_action
        normalized_admissible = {self._normalize_action_text(cmd): cmd for cmd in admissible}
        normalized_processed = self._normalize_action_text(processed_action)
        processed_admissible = normalized_admissible.get(normalized_processed)

        def _debug_return(selected_action: str, reason: str) -> str:
            self._memco_core_debug_append(
                "repair_action",
                step_index=step_index,
                payload={
                    "reason": reason,
                    "raw_response": self._memco_core_debug_text(str(raw_response or ""), limit=1200),
                    "processed_action": str(processed_action or ""),
                    "processed_admissible": bool(processed_admissible),
                    "final_action": str(selected_action or ""),
                    "admissible_sample": admissible[:30],
                    "next_source_action": self._memco_core_graph_policy_next_source_action,
                    "source_priority_queue": self._memco_core_debug_jsonable(
                        self._memco_core_graph_policy_source_queue
                    ),
                    "last_action_candidates": self._memco_core_debug_jsonable(
                        self._memco_core_quality_last_action_candidates
                    ),
                },
            )
            return selected_action

        # Official ALFWorld exposes final placement as `move X to Y`. When the
        # solver is already holding the target at the destination it can still
        # repeat a legacy preparation action such as `open safe 1`. Prefer the
        # exact admissible delivery action in that narrow state. This is a
        # deterministic syntax/state repair, not a policy rerank.
        delivery_repair = self._deterministic_delivery_repair(
            processed_action=processed_action,
            env_ref=env_ref,
            task_config=task_config or {},
            step_index=step_index,
            admissible_actions=admissible,
        )
        if delivery_repair:
            self._external_error = (
                f"last {self._external_retrieval_mode}=delivery_repair; " f"selected={delivery_repair}"
            )
            return _debug_return(delivery_repair, "delivery_repair")

        if self._external_retrieval_mode in {
            "graph_policy",
            "graph_policy_rerank",
            "graph_policy_feedback",
            "graph_policy_candidate",
            "graph_policy_quality",
        } and self._is_concrete_alfworld_action(processed_action):
            object_guard = self._deterministic_object_guard_repair(
                processed_action=processed_action,
                admissible_actions=admissible,
                env_ref=env_ref,
                task_config=task_config or {},
                step_index=step_index,
            )
            if object_guard:
                self._external_error = (
                    f"last {self._external_retrieval_mode}=object_guard_repair; " f"selected={object_guard}"
                )
                return _debug_return(object_guard, "object_guard_repair")
            # Preserve the solver/env workflow for concrete ALFWorld commands.
            # The env adapter still handles official syntax normalization
            # such as put -> move. Graph policy is only a fallback for thoughts
            # or unparseable model output.
            self._external_error = f"last {self._external_retrieval_mode}=concrete_action_preserved"
            return _debug_return(
                processed_admissible or processed_action,
                "concrete_action_preserved",
            )

        if self._external_retrieval_mode == "lightweight_repair":
            # Lightweight repair is deliberately narrow. It never builds a
            # state policy and never overrides an already admissible solver
            # action. It only projects explicit textual intent onto the current
            # admissible command list, plus the official put->move syntax
            # bridge handled by _extract_admissible_action.
            if processed_admissible:
                self._external_error = "last lightweight_repair=already_admissible"
                return processed_admissible
            proposed_action = self._extract_admissible_action(
                raw_response=str(raw_response or ""),
                processed_action=processed_action,
                admissible_actions=admissible,
            )
            if proposed_action is None:
                proposed_action = self._deterministic_delivery_repair(
                    processed_action=processed_action,
                    env_ref=env_ref,
                    task_config=task_config or {},
                    step_index=step_index,
                    admissible_actions=admissible,
                )
            if proposed_action:
                self._external_error = "last lightweight_repair=deterministic_repair; " f"selected={proposed_action}"
                return proposed_action
            self._external_error = "last lightweight_repair=no_deterministic_repair"
            return processed_action

        if self._external_retrieval_mode in {"graph_policy", "graph_policy_rerank", "graph_policy_candidate"}:
            if processed_admissible:
                self._external_error = f"last {self._external_retrieval_mode}=already_admissible"
                return _debug_return(processed_admissible, "already_admissible")
            if self._external_retrieval_mode == "graph_policy":
                phase_action = self._memco_core_graph_policy_phase_fallback_action(
                    env_ref=env_ref,
                    task_config=task_config or {},
                    step_index=step_index,
                    admissible_actions=admissible,
                )
                if phase_action:
                    self._external_error = "last graph_policy=phase_fallback_action; " f"selected={phase_action}"
                    return _debug_return(phase_action, "phase_fallback_action")
            freeform_text = self._normalize_search_text_for_projection(
                " ".join(str(part or "") for part in (raw_response, processed_action))
            )
            queued_search = self._memco_core_graph_policy_queued_search_from_text(
                raw_response=str(raw_response or ""),
                processed_action=processed_action,
                admissible_actions=admissible,
            )
            if queued_search:
                self._external_error = (
                    f"last {self._external_retrieval_mode}=queued_search_from_text; " f"selected={queued_search}"
                )
                return _debug_return(queued_search, "queued_search_from_text")
            safe_search_intent = self._memco_core_extract_safe_search_intent(
                raw_response=str(raw_response or ""),
                processed_action=processed_action,
                admissible_actions=admissible,
            )
            if self._external_retrieval_mode == "graph_policy_rerank":
                reranked_search = self._memco_core_graph_policy_rerank_freeform_search(
                    raw_response=str(raw_response or ""),
                    processed_action=processed_action,
                    explicit_search_action=safe_search_intent,
                    admissible_actions=admissible,
                )
                if reranked_search:
                    self._external_error = (
                        "last graph_policy_rerank=episode_source_rerank; " f"selected={reranked_search}"
                    )
                    return _debug_return(reranked_search, "episode_source_rerank")
            if safe_search_intent:
                self._external_error = (
                    f"last {self._external_retrieval_mode}=explicit_search_intent; " f"selected={safe_search_intent}"
                )
                return _debug_return(safe_search_intent, "explicit_search_intent")
            if self._memco_core_has_explicit_search_target_text(freeform_text):
                self._external_error = (
                    f"last {self._external_retrieval_mode}=explicit_search_unmatched_no_memory_override"
                )
                return _debug_return(processed_action, "explicit_search_unmatched_no_memory_override")
            direct_scene_source = self._memco_core_graph_policy_direct_scene_source_action(
                processed_action=processed_action,
                admissible_actions=admissible,
                env_ref=env_ref,
                task_config=task_config or {},
                step_index=step_index,
                broad_only=False,
            )
            if direct_scene_source:
                self._external_error = (
                    f"last {self._external_retrieval_mode}=direct_scene_source_from_text; "
                    f"selected={direct_scene_source}"
                )
                return _debug_return(direct_scene_source, "direct_scene_source_from_text")
            # GraphPolicy uses graph memory during prompt routing. The action
            # hook remains deterministic and should not run a second retrieve
            # just to reject free-form thoughts; delivery/object guards above
            # are the only action-level interventions for these modes.
            self._external_error = f"last {self._external_retrieval_mode}=no_freeform_action_projection"
            return _debug_return(processed_action, "no_freeform_action_projection")

        query = self._build_external_query(
            env_ref=env_ref,
            task_config=task_config or {},
            step_index=step_index,
        )
        if query is None:
            return processed_action
        setting = str(self._graph_config_value("settings", "local_only") or "local_only")
        owner_scene = self._resolve_external_owner_scene(task_config or {}, env_ref)
        local_memory = (
            self._external_empty_local(owner_scene)
            if setting in {"base", "global_only"}
            else self._external_local_memories.get(owner_scene, self._external_empty_local(owner_scene))
        )
        global_memory = (
            self._external_empty_global()
            if setting in {"base", "local_only"} or self._memco_core_is_two_object_second_search(query)
            else self._external_global_memory
        )
        bundle = self._external_retriever.retrieve(query, local_memory, global_memory)

        # hybrid_repair falls back to deterministic extraction only.
        if processed_admissible:
            self._external_error = "last hybrid_repair=already_admissible"
            return processed_admissible
        proposed_action = self._extract_admissible_action(
            raw_response=str(raw_response or ""),
            processed_action=processed_action,
            admissible_actions=admissible,
        )
        if proposed_action:
            self._external_error = f"last hybrid_repair=deterministic_repair; selected={proposed_action}"
            return proposed_action
        self._external_error = "last hybrid_repair=no_deterministic_repair"
        return processed_action

    def _memco_core_graph_policy_rerank_freeform_search(
        self,
        *,
        raw_response: str,
        processed_action: str,
        explicit_search_action: str | None,
        admissible_actions: list[str],
    ) -> str | None:
        """Use the graph-policy source queue only to break repeated search sweeps.

        This is intentionally narrower than general action repair. It only runs
        for graph_policy_rerank, only when the solver produced a free-form
        search/thought, and only when the queued graph-supported action points
        to a different source type than the current explicit search intent.
        Concrete solver actions are preserved before this helper is reached.
        """
        queued = str(self._memco_core_graph_policy_next_source_action or "").strip()
        if not queued:
            return None
        normalized = {self._normalize_action_text(cmd): cmd for cmd in admissible_actions}
        queued_admissible = normalized.get(self._normalize_action_text(queued))
        if not queued_admissible:
            return None

        text = self._normalize_search_text_for_projection(
            " ".join(str(part or "") for part in (raw_response, processed_action))
        )
        if not text or not any(
            marker in text for marker in ("think", "search", "check", "find", "next likely", "start")
        ):
            return None
        if any(marker in text for marker in ("take ", "clean ", "heat ", "cool ", "move ", "put ")):
            return None

        queued_base = self._memco_core_search_action_base(queued_admissible)
        if not queued_base:
            return None

        explicit_base = self._memco_core_search_action_base(explicit_search_action or "")
        if explicit_base and explicit_base == queued_base:
            return None

        # If the model is explicitly continuing a broad repeated sweep, let
        # rerank switch to the next graph-supported source type. This is the
        # failure pattern in living unseen: drawer/shelf/cabinet sequences keep
        # consuming the 30-step budget after local evidence has gone stale.
        if explicit_base:
            stalled_bases = self._memco_core_episode_searched_source_bases()
            if explicit_base in stalled_bases:
                return queued_admissible
            if explicit_base in {"cabinet", "drawer", "shelf"} and any(
                base in stalled_bases for base in {"cabinet", "drawer", "shelf"}
            ):
                return queued_admissible
            return None

        # Generic thoughts like "I should check another likely location" do not
        # bind a concrete target; using the queued graph action is safe here.
        return queued_admissible

    def _memco_core_search_action_base(self, action: str) -> str:
        text = self._normalize_search_text_for_projection(str(action or ""))
        for prefix in ("go to ", "open ", "examine "):
            if text.startswith(prefix):
                target = text[len(prefix) :].strip()
                return re.sub(r"\s+\d+$", "", target).strip()
        return ""

    def _memco_core_episode_searched_source_bases(self) -> set[str]:
        bases: set[str] = set()
        builder = self.episode_builder
        if builder is None:
            return bases
        state = getattr(builder, "state", None)
        exhausted = list(getattr(state, "exhausted_locations", []) or [])
        searched = list(getattr(state, "searched_locations", []) or [])
        for item in exhausted + searched:
            text = self._normalize_search_text_for_projection(str(item or ""))
            text = re.sub(r"\s+\d+$", "", text).strip()
            if text:
                bases.add(text)
        return bases

    def _extract_admissible_action(
        self,
        *,
        raw_response: str,
        processed_action: str,
        admissible_actions: list[str],
    ) -> str | None:
        import re

        normalized = {self._normalize_action_text(cmd): cmd for cmd in admissible_actions}
        for candidate in [processed_action, raw_response]:
            key = self._normalize_action_text(candidate)
            if key in normalized:
                return normalized[key]
            match = re.match(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$", str(candidate or ""), flags=re.IGNORECASE)
            if match:
                move_key = self._normalize_action_text(f"move {match.group(1)} to {match.group(2)}")
                if move_key in normalized:
                    return normalized[move_key]
        for line in str(raw_response or "").splitlines():
            line = line.strip().lstrip(">").strip()
            if ":" in line and not line.lower().startswith(
                (
                    "think:",
                    "go to ",
                    "take ",
                    "open ",
                    "close ",
                    "examine ",
                    "move ",
                    "put ",
                    "clean ",
                    "cool ",
                    "heat ",
                    "use ",
                )
            ):
                line = line.split(":", 1)[-1].strip()
            key = self._normalize_action_text(line)
            if key in normalized:
                return normalized[key]
            match = re.match(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$", line, flags=re.IGNORECASE)
            if match:
                move_key = self._normalize_action_text(f"move {match.group(1)} to {match.group(2)}")
                if move_key in normalized:
                    return normalized[move_key]
        processed_canonical = self._external_adapter.canonicalize_action(str(processed_action or ""))
        if processed_canonical.verb and processed_canonical.verb != "other":
            for command in admissible_actions:
                action = self._external_adapter.canonicalize_action(command)
                if action.verb == processed_canonical.verb and action.slots == processed_canonical.slots:
                    return command
        embedded = self._extract_embedded_admissible_intent(
            raw_response=raw_response,
            processed_action=processed_action,
            admissible_actions=admissible_actions,
        )
        if embedded:
            return embedded
        return None

    def _extract_embedded_admissible_intent(
        self,
        *,
        raw_response: str,
        processed_action: str,
        admissible_actions: list[str],
    ) -> str | None:
        """Project an embedded natural-language next-action intent to an admissible command.

        The nvdamas solver often emits `think:` turns such as "I will start by
        checking drawer 3" even when the intended environment action is
        currently admissible. MemCo avoids this by parsing and
        repairing against the admissible list. Keep that behavior conservative:
        only return a command when the raw text explicitly points to the same
        object/location as an admissible command.
        """
        import re

        def _intent_norm(value: str) -> str:
            text = str(value or "").lower()
            text = re.sub(r"\((\d+)\)", r" \1", text)
            text = text.replace("_", " ").replace("-", " ")
            text = re.sub(r"[^a-z0-9 ]+", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        text = _intent_norm(f"{processed_action}\n{raw_response}")
        if not text or not text.startswith("think"):
            return None

        def _base(value: str) -> str:
            token = _intent_norm(value)
            for prefix in ("a ", "an ", "the ", "some "):
                if token.startswith(prefix):
                    token = token[len(prefix) :]
            return token

        def _contains_any(patterns: list[str]) -> bool:
            return any(pattern and pattern in text for pattern in patterns)

        scored: list[tuple[int, int, str]] = []
        for index, command in enumerate(admissible_actions):
            action = self._external_adapter.canonicalize_action(command)
            verb = str(action.verb or "").lower()
            slots = {key: _intent_norm(value) for key, value in (action.slots or {}).items()}
            command_norm = _intent_norm(command)
            score = 0

            if command_norm and command_norm in text:
                score = max(score, 100)

            if verb == "go":
                target = slots.get("target", "")
                target_base = _base(target)
                if target:
                    high = [
                        f"go to {target}",
                        f"head to {target}",
                        f"move to {target}",
                        f"start by checking {target}",
                        f"start checking {target}",
                        f"start with {target}",
                        f"start with the nearby {target}",
                        f"nearby {target}",
                        f"starting with {target}",
                        f"check {target}",
                        f"checking {target}",
                        f"visit {target}",
                    ]
                    medium = [
                        f"go to {target_base}",
                        f"check {target_base}",
                        f"checking {target_base}",
                    ]
                    if _contains_any(high):
                        score = max(score, 90)
                    elif target_base and _contains_any(medium):
                        score = max(score, 70)
            elif verb == "open":
                container = slots.get("container", "")
                container_base = _base(container)
                if container:
                    high = [
                        f"open {container}",
                        f"check inside {container}",
                        f"look inside {container}",
                        f"look in {container}",
                        f"search inside {container}",
                    ]
                    medium = [
                        f"check {container}",
                        f"checking {container}",
                        f"open {container_base}",
                    ]
                    if _contains_any(high):
                        score = max(score, 88)
                    elif container_base and _contains_any(medium):
                        score = max(score, 68)
            elif verb == "take":
                obj = slots.get("object", "")
                source = slots.get("source", "")
                obj_base = _base(obj)
                if obj and (
                    f"take {obj}" in text
                    or f"pick up {obj}" in text
                    or (obj_base and f"take {obj_base}" in text)
                    or (obj_base and "take it" in text)
                ):
                    score = max(score, 86)
                if source and score and source in text:
                    score += 6
            elif verb == "move":
                obj = slots.get("object", "")
                dest = slots.get("destination", "")
                obj_base = _base(obj)
                if (
                    obj
                    and dest
                    and (
                        f"move {obj} to {dest}" in text
                        or f"put {obj} in {dest}" in text
                        or f"put {obj} on {dest}" in text
                        or f"put {obj} in on {dest}" in text
                        or (obj_base and f"put {obj_base} in {dest}" in text)
                        or (obj_base and f"put {obj_base} on {dest}" in text)
                        or (obj_base and "put it" in text and dest in text)
                    )
                ):
                    score = max(score, 92)
            elif verb in {"heat", "cool", "clean", "use", "examine"}:
                if command_norm and command_norm in text:
                    score = max(score, 90)

            if score:
                scored.append((score, -index, command))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    def _deterministic_delivery_repair(
        self,
        *,
        processed_action: str,
        env_ref: Any,
        task_config: dict,
        step_index: int,
        admissible_actions: list[str],
    ) -> str | None:
        """Map a narrow official delivery-ready state to placement.

        Some official ALFWorld tasks expose placement as `move X to Y`.
        When the solver is already holding the goal object at the destination
        and the exact move command is currently admissible, continuing to search
        usually loses the episode, especially for two-object tasks. Select that
        delivery action directly. No policy/rerank score is used here.
        """
        action_norm = self._normalize_action_text(processed_action).replace("_", " ")
        if action_norm.startswith("move "):
            return None

        query = self._build_external_query(
            env_ref=env_ref,
            task_config=task_config,
            step_index=step_index,
        )
        if query is None or int(getattr(query, "held_relevant_count", 0) or 0) <= 0:
            return None

        goal_roles = getattr(query, "goal_roles", {}) or {}
        destination = self._normalize_action_text(str(goal_roles.get("destination", "") or "")).replace("_", " ")
        target_object = self._normalize_action_text(str(goal_roles.get("object", "") or "")).replace("_", " ")
        if not destination or not target_object:
            return None
        current_location = self._normalize_action_text(str(getattr(query, "location", "") or "")).replace("_", " ")
        destination_reached = bool(getattr(query, "destination_reached", False)) or (
            bool(current_location) and destination in current_location
        )
        if not destination_reached:
            return None
        if self._memco_core_process_requirement_pending(query=query, env_ref=env_ref):
            return None

        held_objects = [
            self._normalize_action_text(str(item)).replace("_", " ")
            for item in ((getattr(query, "dynamic_context", {}) or {}).get("held_objects", []) or [])
        ]

        def _mentions_target_object(command_norm: str) -> bool:
            if target_object and target_object in command_norm:
                return True
            return any(obj and obj in command_norm for obj in held_objects)

        ranked: list[str] = []
        for command in admissible_actions:
            command_norm = self._normalize_action_text(command).replace("_", " ")
            if not command_norm.startswith("move ") or " to " not in command_norm:
                continue
            _, dest_text = command_norm.rsplit(" to ", 1)
            if destination not in dest_text:
                continue
            if not _mentions_target_object(command_norm):
                continue
            ranked.append(command)

        if not ranked:
            return None
        # Prefer the command whose object slot is actually held, otherwise keep
        # the environment's admissible-command order stable.
        for command in ranked:
            command_norm = self._normalize_action_text(command).replace("_", " ")
            if any(obj and obj in command_norm for obj in held_objects):
                return command
        return ranked[0]

    def _deterministic_object_guard_repair(
        self,
        *,
        processed_action: str,
        admissible_actions: list[str],
        env_ref: Any,
        task_config: dict,
        step_index: int,
    ) -> str | None:
        """Prevent obvious wrong-object actions in graph_policy mode.

        Graph memory may provide useful location priors, but ALFWorld object
        actions must preserve exact target identity. If the solver tries to
        take/process/deliver a visibly similar non-target object (for example
        mug while the task asks for cup), either replace it with an admissible
        same-location target action or turn it into a harmless thought. This
        is deterministic guardrail logic and only affects graph_policy mode.
        """
        query = self._build_external_query(
            env_ref=env_ref,
            task_config=task_config,
            step_index=step_index,
        )
        if query is None:
            return None

        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = self._normalize_action_text(str(goal_roles.get("object", "") or "")).replace("_", " ")
        if not target:
            return None

        action_norm = self._normalize_action_text(processed_action).replace("_", " ")
        match = re.match(
            r"^(take|heat|cool|clean|move|put)\s+(.+?)(?:\s+from\s+(.+)|\s+with\s+(.+)|\s+to\s+(.+)|\s+in/on\s+(.+)|\s+in\s+(.+)|\s+on\s+(.+)|$)",
            action_norm,
        )
        if not match:
            return None
        verb = match.group(1)
        obj_text = str(match.group(2) or "").strip()
        obj_base = re.sub(r"\s+\d+$", "", obj_text).strip()
        if not obj_base or obj_base == target:
            return None

        # Only guard object-manipulation verbs. Navigation/open actions are
        # location search, not object identity commitments.
        if verb not in {"take", "heat", "cool", "clean", "move", "put"}:
            return None

        source_or_dest = " ".join(part for part in match.groups()[2:] if part).strip()
        candidates: list[tuple[int, str]] = []
        for command in admissible_actions:
            command_norm = self._normalize_action_text(command).replace("_", " ")
            if not command_norm.startswith(f"{verb} "):
                # Official final placement often appears as move even when
                # solver says put.
                if not (verb == "put" and command_norm.startswith("move ")):
                    continue
            if not re.search(rf"\b{re.escape(target)}\s+\d+\b|\b{re.escape(target)}\b", command_norm):
                continue
            score = 10
            if source_or_dest and source_or_dest in command_norm:
                score += 10
            candidates.append((score, command))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]

        return f"think: I should not use {obj_base} because the target object is {target}; continue searching for an exact {target}."

    def _memco_core_graph_policy_direct_scene_source_action(
        self,
        *,
        processed_action: str,
        admissible_actions: list[str],
        env_ref: Any,
        task_config: dict,
        step_index: int,
        broad_only: bool,
    ) -> str | None:
        """Project exact local graph source relations into a search action.

        This is intentionally graph_policy-only at the call site. It reads
        local `scene_relation` artifacts that were built from successful
        `take(target_object, source)` transitions, so it uses graph semantics
        instead of object co-occurrence in a state observation.
        """
        query = self._build_external_query(env_ref=env_ref, task_config=task_config, step_index=step_index)
        if query is None:
            return None
        if getattr(query, "held_relevant_count", 0) > 0:
            return None
        progress = str(getattr(query, "progress_state", "") or "")
        if progress and not progress.startswith("search") and progress not in {"locate_target", "inspect_container"}:
            return None

        action_norm = self._normalize_action_text(processed_action).replace("_", " ")
        if broad_only:
            match = re.match(r"^(go to|open|examine)\s+(.+)$", action_norm)
            if not match:
                return None
            target_base = re.sub(r"\s+\d+$", "", match.group(2).strip())
            if target_base not in {"cabinet", "drawer"}:
                return None
        else:
            if not any(marker in action_norm for marker in ("think", "find", "search", "check", "start with", "try ")):
                return None
            if any(marker in action_norm for marker in ("take ", "clean ", "heat ", "cool ", "move ", "put ")):
                return None

        goal_roles = getattr(query, "goal_roles", {}) or {}
        target_object = self._memco_core_base_token(str(goal_roles.get("object", "") or ""))
        destination = self._memco_core_base_token(str(goal_roles.get("destination", "") or ""))
        tool = self._memco_core_base_token(str(goal_roles.get("tool", "") or ""))
        task_family = str(getattr(query, "task_family", "") or "").lower()
        if not target_object:
            return None

        owner_scene = self._resolve_external_owner_scene(task_config, env_ref)
        local_memory = self._external_local_memories.get(owner_scene)
        artifacts = getattr(local_memory, "artifacts_by_id", {}) if local_memory is not None else {}
        if not artifacts:
            return None

        def norm_entity(value: str) -> str:
            return self._normalize_action_text(str(value or "")).replace(" ", "_")

        checked: set[str] = set()
        for row in list(getattr(env_ref, "current_history", []) or []):
            past_action = self._normalize_action_text(str(row.get("Action", "") or "")).replace("_", " ")
            past_obs = self._normalize_action_text(str(row.get("Observation", "") or "")).replace("_", " ")
            target = ""
            for prefix in ("open ", "examine ", "go to "):
                if past_action.startswith(prefix):
                    target = past_action[len(prefix) :].strip()
                    break
            if not target:
                continue
            if past_action.startswith(("open ", "examine ")):
                checked.add(norm_entity(target))
            elif (
                past_action.startswith("go to ")
                and "is closed" not in past_obs
                and ("you see" in past_obs or "on the" in past_obs or "in it" in past_obs)
            ):
                checked.add(norm_entity(target))

        admissible_by_norm = {self._normalize_action_text(cmd).replace("_", " "): cmd for cmd in admissible_actions}

        def search_actions(source_instance: str, source_base: str) -> list[str]:
            instance_space = source_instance.replace("_", " ")
            base_space = source_base.replace("_", " ")
            ranked: list[tuple[int, int, str]] = []
            for norm_cmd, original in admissible_by_norm.items():
                match = re.match(r"^(go to|open|examine)\s+(.+)$", norm_cmd)
                if not match:
                    continue
                target = match.group(2).strip()
                target_key = norm_entity(target)
                if target_key in checked:
                    continue
                target_base = re.sub(r"\s+\d+$", "", target).strip()
                priority = 0
                if instance_space and target == instance_space:
                    priority = {"go to": 7, "open": 6, "examine": 5}.get(match.group(1), 0)
                elif base_space and target_base == base_space:
                    priority = {"go to": 3, "open": 2, "examine": 1}.get(match.group(1), 0)
                if priority <= 0:
                    continue
                ordinal = 999
                ordinal_match = re.search(r"\b(\d+)\b", target)
                if ordinal_match:
                    try:
                        ordinal = int(ordinal_match.group(1))
                    except Exception:
                        ordinal = 999
                ranked.append((priority, ordinal, original))
            return [cmd for _priority, _ordinal, cmd in sorted(ranked, key=lambda item: (-item[0], item[1]))]

        candidates: list[tuple[float, str]] = []
        artifact_values = artifacts.values() if isinstance(artifacts, dict) else artifacts
        for artifact in artifact_values:
            payload = getattr(artifact, "payload", {}) or {}
            anchor = getattr(artifact, "anchor", {}) or {}
            if str(payload.get("pattern_kind", "") or "") != "scene_relation":
                continue
            if str(payload.get("relation_kind", "") or "") != "object_location_prior":
                continue
            if str(payload.get("object_role", "") or "") != "target_object":
                continue
            anchor_family = str(anchor.get("task_family", "") or "").lower()
            if task_family and anchor_family and anchor_family != task_family:
                continue
            goal_signature = self._normalize_action_text(str(anchor.get("goal_signature", "") or ""))
            if not goal_signature.startswith(f"{target_object}->"):
                continue
            source_instance = norm_entity(str(payload.get("source_instance", "") or ""))
            source_base = self._memco_core_base_token(str(payload.get("source_base", "") or source_instance))
            if not source_base or not source_instance:
                continue
            if source_base in {destination, tool}:
                continue
            actions = search_actions(source_instance, source_base)
            if not actions:
                continue
            stats = getattr(artifact, "stats", None)
            support = float(getattr(stats, "support", 0) or 0)
            success = float(getattr(stats, "success", 0) or 0)
            failure = float(getattr(stats, "failure", 0) or 0)
            confidence = float(getattr(stats, "confidence", 0.0) or 0.0)
            role_bonus = 0.12 if str(payload.get("source_role", "") or "") == "support_surface" else 0.0
            exact_goal_bonus = 0.16 if destination and goal_signature == f"{target_object}->{destination}" else 0.0
            score = (
                0.8
                + role_bonus
                + exact_goal_bonus
                + min(support, 4.0) * 0.04
                + min(success, 3.0) * 0.04
                + min(confidence, 1.0) * 0.08
                - min(failure, 3.0) * 0.08
            )
            if source_base in {"cabinet", "drawer"}:
                score -= 0.05
            candidates.append((score, actions[0]))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[0][1]
        if self._normalize_action_text(selected) == self._normalize_action_text(processed_action):
            return None
        return selected

    def _memco_core_extract_safe_search_intent(
        self,
        *,
        raw_response: str,
        processed_action: str,
        admissible_actions: list[str],
    ) -> str | None:
        """Project only explicit search/navigation intent from a think output.

        Broad action projection is risky because it can turn free-form text
        into wrong object manipulation. This helper only returns currently
        admissible go/open/examine commands, and only when the model text
        explicitly names the same command or the same location target.
        """
        text = " ".join(str(part or "") for part in (raw_response, processed_action))
        normalized_text = self._normalize_search_text_for_projection(text)
        if not normalized_text:
            return None
        if not any(
            marker in normalized_text
            for marker in ("think", "i will", "start with", "check ", "go to ", "going to ", "open ", "examine ")
        ):
            return None

        matches: list[tuple[int, str]] = []
        for command in admissible_actions:
            command_norm = self._normalize_search_text_for_projection(command)
            match = re.match(r"^(go to|open|examine)\s+(.+)$", command_norm)
            if not match:
                continue
            target = match.group(2).strip()
            priority = {"go to": 3, "open": 2, "examine": 1}.get(match.group(1), 0)
            explicit_command = command_norm in normalized_text
            explicit_target = bool(
                re.search(
                    rf"\b(?:start with|check|checking|try|go to|going to|open|examine)\s+(?:the\s+)?{re.escape(target)}\b",
                    normalized_text,
                )
            )
            if explicit_command or explicit_target:
                matches.append((priority + (3 if explicit_command else 0), command))

        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    def _normalize_search_text_for_projection(self, text: str) -> str:
        normalized = self._normalize_action_text(str(text or "")).replace("_", " ")
        # Solver often writes object/location lists as `fridge (1)` while the
        # ALFWorld command is `go to fridge 1`. Normalize this list notation
        # before exact admissible matching. This stays inside graph_policy's
        # repair path and does not affect the environment adapter.
        normalized = re.sub(r"\b([a-z][a-z0-9]*)\s*\(\s*(\d+)\s*\)", r"\1 \2", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _memco_core_has_explicit_search_target_text(self, normalized_text: str) -> bool:
        text = str(normalized_text or "")
        if not text or "think" not in text:
            return False
        if any(marker in text for marker in ("take ", "clean ", "heat ", "cool ", "move ", "put ")):
            return False
        return bool(
            re.search(
                r"\b(?:start with|check|checking|try|go to|going to|open|examine)\s+(?:the\s+)?[a-z][a-z0-9]*(?:\s+\d+)?\b",
                text,
            )
        )

    def _memco_core_graph_policy_queued_search_from_text(
        self,
        *,
        raw_response: str,
        processed_action: str,
        admissible_actions: list[str],
    ) -> str | None:
        """Prefer MemCo's queued source action over a free-form search thought."""
        queued = str(self._memco_core_graph_policy_next_source_action or "").strip()
        if not queued:
            return None
        normalized = {self._normalize_action_text(cmd): cmd for cmd in admissible_actions}
        queued_admissible = normalized.get(self._normalize_action_text(queued))
        if not queued_admissible:
            return None
        queue_head = (self._memco_core_graph_policy_source_queue or [{}])[0]
        try:
            queue_score = float(queue_head.get("score", 0.0) or 0.0)
        except Exception:
            queue_score = 0.0
        try:
            queue_confidence = float(queue_head.get("confidence", 0.0) or 0.0)
        except Exception:
            queue_confidence = 0.0
        try:
            queue_support = int(queue_head.get("support", 0) or 0)
        except Exception:
            queue_support = 0
        high_confidence_queue = queue_score >= 1.05 or (queue_confidence >= 0.62 and queue_support >= 2)
        text = self._normalize_action_text(
            " ".join(str(part or "") for part in (raw_response, processed_action))
        ).replace("_", " ")
        if not text or "think" not in text:
            return None
        if not any(
            marker in text
            for marker in ("find", "search", "check", "start with", "try ", "go to ", "open ", "examine ")
        ):
            return None
        queued_norm = self._normalize_action_text(queued_admissible).replace("_", " ")
        if queued_norm in text:
            return queued_admissible
        if high_confidence_queue and not any(
            marker in text
            for marker in (
                "now i have",
                "already have",
                "holding",
                "i am holding",
                "i have picked",
                "i have taken",
            )
        ):
            return queued_admissible
        # Only override search/location thoughts. Object manipulation thoughts
        # stay with the solver unless other deterministic guards apply.
        if any(marker in text for marker in ("take ", "clean ", "heat ", "cool ", "move ", "put ")):
            return None
        return queued_admissible

    def _memco_core_graph_policy_phase_fallback_action(
        self,
        *,
        env_ref: Any,
        task_config: dict,
        step_index: int,
        admissible_actions: list[str],
    ) -> str | None:
        """Map free-form graph_policy output to the current task phase.

        This is deliberately narrower than source reranking. It only fires
        after the solver produced no concrete admissible action. If the target
        is held, the next useful phase is process/tool or delivery, not another
        source search. This protects graph_policy from the common failure where
        memory source priors keep pushing search after acquisition.
        """
        query = self._build_external_query(
            env_ref=env_ref,
            task_config=task_config,
            step_index=step_index,
        )
        if query is None or int(getattr(query, "held_relevant_count", 0) or 0) <= 0:
            return None

        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = self._memco_core_base_token(str(goal_roles.get("object", "") or ""))
        destination = self._memco_core_base_token(str(goal_roles.get("destination", "") or ""))
        tool = self._memco_core_base_token(str(goal_roles.get("tool", "") or ""))
        current_location = self._memco_core_base_token(str(getattr(query, "location", "") or ""))
        process_pending = self._memco_core_process_requirement_pending(query=query, env_ref=env_ref)

        def _norm(value: str) -> str:
            return self._normalize_action_text(value).replace("_", " ")

        def _mentions_target(command_norm: str) -> bool:
            if target and target in self._memco_core_base_token(command_norm):
                return True
            held = (getattr(query, "dynamic_context", {}) or {}).get("held_objects", []) or []
            return any(
                self._memco_core_base_token(str(item)) in self._memco_core_base_token(command_norm) for item in held
            )

        if process_pending:
            for command in admissible_actions:
                command_norm = _norm(command)
                if not command_norm.startswith(("heat ", "cool ", "clean ")):
                    continue
                if _mentions_target(command_norm):
                    return command
            if tool and tool not in current_location:
                for command in admissible_actions:
                    command_norm = _norm(command)
                    if command_norm.startswith("go to ") and tool in self._memco_core_base_token(command_norm):
                        return command
            return None

        if destination and destination not in current_location:
            for command in admissible_actions:
                command_norm = _norm(command)
                if command_norm.startswith("go to ") and destination in self._memco_core_base_token(command_norm):
                    return command
        return None

    def _memco_core_process_requirement_pending(self, *, query: Any, env_ref: Any) -> bool:
        """Return True when heat/cool/clean must still happen before delivery."""
        task_family = str(getattr(query, "task_family", "") or "").lower()
        required = ""
        if "pick_heat" in task_family or "heat" in task_family:
            required = "heat"
        elif "pick_cool" in task_family or "cool" in task_family:
            required = "cool"
        elif "pick_clean" in task_family or "clean" in task_family:
            required = "clean"
        if not required:
            return False

        goal_roles = getattr(query, "goal_roles", {}) or {}
        target_object = self._memco_core_base_token(str(goal_roles.get("object", "") or ""))
        for row in list(getattr(env_ref, "current_history", []) or []):
            action = self._normalize_action_text(str(row.get("Action", "") or "")).replace("_", " ")
            if not action.startswith(required + " "):
                continue
            if not target_object or target_object in self._memco_core_base_token(action):
                observation = str(row.get("Observation", "") or "")
                if "nothing happens" not in observation.lower():
                    return False
        return True

    @staticmethod
    def _memco_core_is_two_object_second_search(query: Any) -> bool:
        """Whether MemCo is in the post-first-delivery search for a second target."""
        try:
            required = int(getattr(query, "required_count", 0) or 0)
            placed = int(getattr(query, "placed_relevant_count", 0) or 0)
            held = int(getattr(query, "held_relevant_count", 0) or 0)
            remaining = int(getattr(query, "remaining_relevant_count", 0) or 0)
        except Exception:
            return False
        progress_state = str(getattr(query, "progress_state", "") or "")
        progress_hint = str(getattr(query, "progress_hint", "") or "")
        task_family = str(getattr(query, "task_family", "") or "")
        is_two_object = (
            required > 1
            or "two_obj" in task_family
            or progress_hint.startswith("multi_object")
            or "search_second" in progress_state
        )
        return bool(is_two_object and placed > 0 and held <= 0 and remaining > 0)

    def _render_graph_aware_lightweight_evidence(self, *, query: Any, bundle: Any, env_ref: Any) -> str:
        """Render lightweight retrieval as explicit local/global graph evidence.

        The default SupportBundle text is intentionally compact, but it tends to
        flatten graph structure into generic hints. This renderer keeps the
        nvdamas workflow unchanged while exposing the parts that make MemCo useful:
        matched local state, successful local continuations, local cautions, and
        transferable global rules.
        """
        import re

        def _clean(summary: str) -> str:
            cleaned = str(summary or "").strip()
            for prefix in (
                "Scene relation: ",
                "Plan: ",
                "Workflow pattern: ",
                "Workflow: ",
                "Reflection: ",
                "Closure: ",
                "Local state fact: ",
                "Prototype: ",
                "Precondition: ",
                "Blocked: ",
            ):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
            return re.sub(r"\s+", " ", cleaned).strip()

        def _norm(value: str) -> str:
            return re.sub(r"\s+", " ", str(value or "").replace("_", " ").replace("-", " ").lower()).strip()

        def _dedupe(items: list[str], limit: int) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for item in items:
                text = str(item or "").strip()
                key = _norm(text)
                if not text or key in seen:
                    continue
                seen.add(key)
                out.append(text)
                if len(out) >= limit:
                    break
            return out

        def _is_template_like(text: str) -> bool:
            lowered = _norm(text)
            return any(
                marker in lowered
                for marker in (
                    "target object",
                    "goal destination",
                    "support surface",
                    "container=container",
                    "object=target object",
                    "destination=goal destination",
                    "go(target=",
                    "take(object=target object",
                    "move(destination=goal destination",
                )
            )

        def _is_generic(text: str) -> bool:
            lowered = _norm(text)
            return any(
                marker in lowered
                for marker in (
                    "within search open advances",
                    "within search open supports",
                    "within search examine advances",
                    "within search examine supports",
                    "wrong target object tends to derail",
                )
            )

        def _is_failure_like(item: Any) -> bool:
            ctype = str(getattr(item, "candidate_type", "") or "").lower()
            branch = str(getattr(item, "branch_tag", "") or "").lower()
            pattern = str(getattr(item, "pattern_kind", "") or "").lower()
            dynamic = getattr(item, "dynamic", {}) or {}
            relation = str(dynamic.get("relation_kind", "") or "").lower()
            return (
                "failure" in ctype
                or "repair" in ctype
                or branch in {"failure_branch", "repair_branch"}
                or pattern in {"reflection", "repair", "failure", "anti_pattern"}
                or relation == "searched_empty"
            )

        def _format_item(item: Any, *, with_actions: bool = False) -> str:
            text = _clean(getattr(item, "summary", ""))
            if not text:
                return ""
            if with_actions:
                patterns = [
                    str(pattern).strip()
                    for pattern in (getattr(item, "action_patterns", ()) or ())
                    if str(pattern).strip()
                ]
                if patterns:
                    text = f"{text}; observed continuation: {' -> '.join(patterns[:3])}"
            score = getattr(item, "score", None)
            if isinstance(score, (int, float)):
                text = f"{text} [score={score:.2f}]"
            return text

        def _rank_key(item: Any) -> tuple[float, float, float, float]:
            return (
                float(getattr(item, "score", 0.0) or 0.0),
                float(getattr(item, "state_relevance", 0.0) or 0.0),
                float(getattr(item, "goal_relevance", 0.0) or 0.0),
                float(getattr(item, "task_relevance", 0.0) or 0.0),
            )

        route_weights = getattr(bundle, "routing_weights", {}) or {}

        def _route_weight(slot: str, source: str) -> float:
            weights = route_weights.get(slot, {}) or {}
            value = weights.get(source, 0.0)
            return float(value) if isinstance(value, (int, float)) else 0.0

        dynamic = getattr(query, "dynamic_context", {}) or {}
        visible = [str(item) for item in dynamic.get("visible_objects", []) or [] if str(item).strip()]
        held = [str(item) for item in dynamic.get("held_objects", []) or [] if str(item).strip()]
        exhausted = [str(item) for item in dynamic.get("exhausted_locations", []) or [] if str(item).strip()]
        exhausted_terms = {_norm(item) for item in exhausted if str(item).strip()}
        admissible = [
            str(cmd).strip() for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()
        ]
        admissible_norm = {_norm(cmd) for cmd in admissible}
        goal_roles = getattr(query, "goal_roles", {}) or {}
        target_object = _norm(str(goal_roles.get("object", "") or ""))
        two_object_second_search = self._memco_core_is_two_object_second_search(query)
        goal_bits = [
            f"object={goal_roles.get('object')}" if goal_roles.get("object") else "",
            f"destination={goal_roles.get('destination')}" if goal_roles.get("destination") else "",
            f"tool={goal_roles.get('tool')}" if goal_roles.get("tool") else "",
        ]

        local_state_sources = (
            list(getattr(bundle, "fact_items", []) or [])
            + list(getattr(bundle, "relation_items", []) or [])
            + list(getattr(bundle, "local_graph_contribution", []) or [])
        )
        local_success_sources = (
            list(getattr(bundle, "plan_items", []) or [])
            + list(getattr(bundle, "workflow_items", []) or [])
            + list(getattr(bundle, "precondition_items", []) or [])
            + list(getattr(bundle, "local_promoted_contribution", []) or [])
            + [item for item in (getattr(bundle, "local_items", []) or []) if not _is_failure_like(item)]
        )
        local_failure_sources = (
            list(getattr(bundle, "repair_items", []) or [])
            + list(getattr(bundle, "reflection_items", []) or [])
            + [
                item
                for item in (
                    list(getattr(bundle, "local_graph_contribution", []) or [])
                    + list(getattr(bundle, "local_promoted_contribution", []) or [])
                    + list(getattr(bundle, "local_items", []) or [])
                )
                if _is_failure_like(item)
            ]
        )
        global_sources = list(getattr(bundle, "global_promoted_contribution", []) or []) + list(
            getattr(bundle, "global_items", []) or []
        )
        global_task_plan_sources = list(getattr(bundle, "global_task_plan_items", []) or [])
        global_task_plan_ids = {str(getattr(item, "candidate_id", "") or "") for item in global_task_plan_sources}
        global_sources = [
            item for item in global_sources if str(getattr(item, "candidate_id", "") or "") not in global_task_plan_ids
        ]

        def _is_checked_positive_location(text: str) -> bool:
            lowered = _norm(text)
            if not target_object or target_object not in lowered:
                return False
            positive_markers = ("visible", "found at", "was found at", "located at", "contains")
            if not any(marker in lowered for marker in positive_markers):
                return False
            return any(location and location in lowered for location in exhausted_terms)

        def _select_items(
            items: list[Any], *, limit: int, with_actions: bool = False, allow_failure: bool = True
        ) -> list[str]:
            ranked = sorted(items, key=_rank_key, reverse=True)
            rendered: list[str] = []
            for item in ranked:
                text = _format_item(item, with_actions=with_actions)
                if not text:
                    continue
                if _is_template_like(text) or _is_generic(text):
                    continue
                if _is_checked_positive_location(text):
                    continue
                if not allow_failure and _is_failure_like(item):
                    continue
                if two_object_second_search and with_actions:
                    lowered = _norm(text)
                    has_search_progress = any(
                        marker in lowered
                        for marker in (
                            "go(",
                            "open(",
                            "take(",
                            "locate target",
                            "acquire target",
                            "found at",
                            "visible",
                        )
                    )
                    delivery_only = "move(" in lowered and not has_search_progress
                    if delivery_only:
                        continue
                rendered.append(text)
            return _dedupe(rendered, limit)

        def _display_location(location: str) -> str:
            text = str(location or "").strip()
            if text.startswith("a_"):
                text = text[2:]
            return text.replace("_", " ")

        def _location_key(location: str) -> str:
            return _norm(_display_location(location))

        def _admissible_for_location(location: str) -> list[str]:
            key = _location_key(location)
            if not key:
                return []
            matches: list[str] = []
            for command in admissible:
                lowered = _norm(command)
                if key in lowered and any(lowered.startswith(prefix) for prefix in ("go to ", "open ", "examine ")):
                    matches.append(command)
            return matches

        def _extract_target_locations(item: Any) -> list[str]:
            summary = str(getattr(item, "summary", "") or "")
            raw_lowered = summary.lower()
            lowered = _norm(summary)
            dynamic_payload = getattr(item, "dynamic", {}) or {}
            artifact_payload = (
                dynamic_payload.get("payload", {}) if isinstance(dynamic_payload.get("payload", {}), dict) else {}
            )
            relation_kind = str(
                dynamic_payload.get("relation_kind", "") or artifact_payload.get("relation_kind", "") or ""
            )
            locations: list[str] = []

            if not target_object:
                return locations
            payload_refs = [
                str(ref)
                for ref in (
                    list(artifact_payload.get("graph_refs", []) or [])
                    + [artifact_payload.get("state_ref", ""), artifact_payload.get("next_state_ref", "")]
                )
                if str(ref).strip()
            ]
            payload_blob = _norm(" ".join(payload_refs))
            raw_payload_blob = " ".join(payload_refs).lower()

            def _state_ref_locations_with_target() -> list[str]:
                found: list[str] = []
                raw_target_markers = (f"a_{target_object}", f"{target_object}_")
                norm_target_markers = (f"a {target_object}", target_object)
                for ref in payload_refs:
                    ref_raw = ref.lower()
                    ref_norm = _norm(ref)
                    if not ref_raw.startswith("state:") and "|state:" not in ref_raw:
                        continue
                    state_ref = ref[ref_raw.index("state:") :] if "|state:" in ref_raw else ref
                    if not (
                        any(marker in ref_raw for marker in raw_target_markers)
                        or any(marker in ref_norm for marker in norm_target_markers)
                    ):
                        continue
                    parts = state_ref.split("|")
                    if len(parts) >= 2 and parts[1].strip():
                        found.append(parts[1].strip())
                return found

            state_ref_locations = _state_ref_locations_with_target()
            target_markers = (
                target_object in lowered,
                target_object in payload_blob,
                f"a {target_object}" in lowered,
                f"a {target_object}" in payload_blob,
                f"a_{target_object}" in str(summary).lower(),
                f"a_{target_object}" in raw_payload_blob,
                str(dynamic_payload.get("object_role", "") or artifact_payload.get("object_role", "") or "")
                == "target_object",
            )
            positive_markers = (
                "relevant visible" in lowered,
                "visible" in lowered,
                "found at" in lowered,
                "was found at" in lowered,
                "located at" in lowered,
                "contains" in lowered,
                relation_kind == "object_location_prior",
                bool(state_ref_locations),
            )
            if not any(target_markers) or not any(positive_markers):
                return locations

            locations.extend(state_ref_locations)
            for key in ("source_instance", "location", "target", "container"):
                value = str(dynamic_payload.get(key, "") or artifact_payload.get(key, "") or "").strip()
                if value:
                    locations.append(value)
            for pattern in (
                r"\bnear\s+([a-z]+_\d+)\b",
                r"\bin\s+([a-z]+_\d+)\b",
                r"\bon\s+([a-z]+_\d+)\b",
                r"\bat\s+([a-z]+_\d+)\b",
                r"\bfrom\s+([a-z]+_\d+)\b",
                r"\b([a-z]+_\d+);\s*relevant\s+visible\b",
            ):
                locations.extend(match.group(1) for match in re.finditer(pattern, raw_lowered))
            for pattern in (
                r"\bnear\s+([a-z]+\s+\d+)\b",
                r"\bin\s+([a-z]+\s+\d+)\b",
                r"\bon\s+([a-z]+\s+\d+)\b",
                r"\bat\s+([a-z]+\s+\d+)\b",
                r"\bfrom\s+([a-z]+\s+\d+)\b",
                r"\b([a-z]+\s+\d+);\s*relevant\s+visible\b",
            ):
                locations.extend(match.group(1).replace(" ", "_") for match in re.finditer(pattern, lowered))
            return _dedupe(locations, 4)

        def _search_priority_items(items: list[Any], *, limit: int = 2) -> list[str]:
            progress = str(getattr(query, "progress_state", "") or "")
            if not progress.startswith("search") or not target_object:
                return []
            ranked = sorted(items, key=_rank_key, reverse=True)
            priorities: list[str] = []
            for item in ranked:
                text = _format_item(item, with_actions=False)
                if not text or _is_failure_like(item) or _is_template_like(text) or _is_generic(text):
                    continue
                for location in _extract_target_locations(item):
                    location_display = _display_location(location)
                    location_norm = _location_key(location)
                    if not location_norm or location_norm in exhausted_terms:
                        continue
                    commands = _admissible_for_location(location)
                    if commands:
                        action_hint = f"available action: {commands[0]}"
                    else:
                        action_hint = f"check {location_display} when reachable"
                    priorities.append(
                        f"Target-object search priority: local graph links {goal_roles.get('object') or target_object} "
                        f"with {location_display}; {action_hint}. Evidence: {_clean(text)}"
                    )
            return _dedupe(priorities, limit)

        local_state_limit = 3 if _route_weight("fact", "local") >= 0.75 else 2
        local_success_limit = (
            3 if max(_route_weight("workflow", "local"), _route_weight("plan", "local")) >= 0.55 else 2
        )
        global_transfer_limit = (
            2
            if max(
                _route_weight("workflow", "global"),
                _route_weight("precondition", "global"),
                _route_weight("closure", "global"),
            )
            >= 0.35
            else 1
        )

        enable_search_priority = bool(self.global_config.get("memco_enable_search_priority", False))
        search_priority = (
            _search_priority_items(local_state_sources + local_success_sources, limit=2)
            if enable_search_priority
            else []
        )
        local_state = _select_items(local_state_sources, limit=local_state_limit)
        local_success = _select_items(
            local_success_sources, limit=local_success_limit, with_actions=True, allow_failure=False
        )
        local_failure = _select_items(local_failure_sources, limit=2)
        global_task_plan = _select_items(global_task_plan_sources, limit=1, with_actions=False, allow_failure=False)
        global_transfer = _select_items(
            global_sources, limit=global_transfer_limit, with_actions=True, allow_failure=False
        )

        workflow_hints = _dedupe(
            [str(item).strip() for item in (getattr(bundle, "workflow_hints", []) or []) if str(item).strip()],
            2,
        )
        blocked = _dedupe(
            [str(item).strip() for item in (getattr(bundle, "blocked_actions", []) or []) if str(item).strip()],
            2,
        )
        warnings = _dedupe(
            [str(item).strip() for item in (getattr(bundle, "warnings", []) or []) if str(item).strip()],
            2,
        )

        lines: list[str] = []
        lines.append("Graph-aware lightweight memory. Current observation and admissible actions have priority.")
        local_route = max(
            _route_weight("fact", "local"),
            _route_weight("workflow", "local"),
            _route_weight("precondition", "local"),
        )
        global_route = max(
            _route_weight("task_plan", "global"),
            _route_weight("workflow", "global"),
            _route_weight("precondition", "global"),
            _route_weight("closure", "global"),
        )
        lines.append(
            "Routing balance: "
            f"local_step={local_route:.2f}, global_transfer={global_route:.2f}. "
            "Use global memory for task-level skeletons; use local graph evidence for current-state actions."
        )
        lines.append("Treat graph memory as supplementary evidence, not as a direct action instruction.")
        if two_object_second_search:
            lines.append(
                "Two-object progress: one target has already been delivered. "
                "Search for another target instance now; do not return to the destination until holding it."
            )
            lines.append(
                "For this second-target search, trust local graph evidence over global transfer. "
                "Ignore global exact-location examples from other scenes."
            )
        query_parts = [
            f"progress={getattr(query, 'progress_state', '') or 'unknown'}",
            f"stage={getattr(query, 'current_stage', '') or 'unknown'}",
            f"location={getattr(query, 'location', '') or 'unknown'}",
        ] + [bit for bit in goal_bits if bit]
        lines.append("Current graph query: " + "; ".join(query_parts) + ".")
        if global_task_plan:
            lines.append("Global task-level memory:")
            lines.extend(f"- {item}" for item in global_task_plan)
        if visible or held or exhausted:
            context_parts = []
            if held:
                context_parts.append("held=" + ", ".join(held[:3]))
            if visible:
                context_parts.append("visible=" + ", ".join(visible[:5]))
            if exhausted:
                context_parts.append("already_checked=" + ", ".join(exhausted[:5]))
            lines.append("Current graph state: " + "; ".join(context_parts) + ".")
        if search_priority:
            lines.append("Local graph search priority:")
            lines.extend(f"- {item}" for item in search_priority)
        if local_state:
            lines.append("Local matched state evidence:")
            lines.extend(f"- {item}" for item in local_state)
        if local_success:
            lines.append("Local successful continuations:")
            lines.extend(f"- {item}" for item in local_success)
        if workflow_hints:
            lines.append("Local workflow hints:")
            lines.extend(f"- {item}" for item in workflow_hints)
        if local_failure or blocked or warnings:
            lines.append("Local failure avoidance:")
            lines.extend(f"- {item}" for item in local_failure)
            lines.extend(f"- Avoid/risky: {item}" for item in blocked)
            lines.extend(f"- Warning: {item}" for item in warnings)
        if global_transfer and not two_object_second_search:
            lines.append("Global transferable memory:")
            lines.extend(f"- {item}" for item in global_transfer)
        if len(lines) <= 4:
            return bundle.to_planner_text(
                max_items=3,
                current_location=str(getattr(query, "location", "") or ""),
                exhausted_locations=tuple(exhausted),
                visible_objects=tuple(visible),
                held_objects=tuple(held),
                task_family=str(getattr(query, "task_family", "") or ""),
            )
        return "\n".join(lines)

    def _render_graph_policy_evidence(
        self,
        *,
        query: Any,
        bundle: Any,
        env_ref: Any,
        local_memory: Any = None,
        global_memory: Any = None,
    ) -> str:
        """Render MemCo graph_policy through the graph bundle's routed support.

        Graph policy has two different time scales:
        - an episode-level global task skeleton, retrieved from promoted global
          memory and kept across steps;
        - step-level local graph grounding/domain evidence, retrieved every step.

        Keeping the global skeleton persistent avoids dropping transferable
        two-object plans after the first object has been delivered, while leaving
        the nvdamas solver/env workflow unchanged.
        """
        dynamic = getattr(query, "dynamic_context", {}) or {}
        visible = tuple(str(item) for item in dynamic.get("visible_objects", ()) or () if str(item).strip())
        held = tuple(str(item) for item in dynamic.get("held_objects", ()) or () if str(item).strip())
        exhausted = tuple(str(item) for item in dynamic.get("exhausted_locations", ()) or () if str(item).strip())

        route_weights = getattr(bundle, "routing_weights", {}) or {}

        def _route_weight(slot: str, source: str) -> float:
            weights = route_weights.get(slot, {}) or {}
            value = weights.get(source, 0.0)
            return float(value) if isinstance(value, (int, float)) else 0.0

        def _clean(summary: str) -> str:
            cleaned = str(summary or "").strip()
            for prefix in (
                "Scene relation: ",
                "Plan: ",
                "Workflow pattern: ",
                "Workflow: ",
                "Reflection: ",
                "Closure: ",
                "Local state fact: ",
                "Prototype: ",
                "Precondition: ",
                "Blocked: ",
            ):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
            return cleaned

        def _dedupe(items: list[str], limit: int) -> list[str]:
            seen: set[str] = set()
            rendered: list[str] = []
            for item in items:
                text = str(item or "").strip()
                key = text.lower()
                if not text or key in seen:
                    continue
                seen.add(key)
                rendered.append(text)
                if len(rendered) >= limit:
                    break
            return rendered

        def _norm(value: str) -> str:
            return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())

        def _skeleton_key() -> str:
            goal_roles = getattr(query, "goal_roles", {}) or {}
            return "|".join(
                _norm(str(value or ""))
                for value in (
                    getattr(query, "task_family", ""),
                    goal_roles.get("object", ""),
                    goal_roles.get("destination", ""),
                    goal_roles.get("tool", ""),
                    getattr(query, "required_count", ""),
                )
            )

        def _is_concrete_scene_hint(text: str) -> bool:
            lowered = _norm(text)
            raw = str(text or "").lower()
            if any(
                marker in lowered
                for marker in ("found at", "was found at", "located at", "in this layout", "scene relation")
            ):
                return True
            if re.search(r"\b[a-z]+_\d+\b", raw):
                return True
            if re.search(r"\b[a-z]+\s+\d+\b", lowered):
                return True
            return False

        def _contains_non_action_think_step(text: str) -> bool:
            lowered = str(text or "").lower()
            normed = _norm(text)
            return bool(
                "think:" in lowered
                or "-> think:" in lowered
                or "think: ->" in lowered
                or normed.startswith("think ")
                or " think ->" in normed
            )

        def _is_usable_global_skeleton(text: str) -> bool:
            lowered = _norm(text)
            if not lowered:
                return False
            if any(marker in lowered for marker in ("wrong target object", "blocked:", "failed", "failure")):
                return False
            if _contains_non_action_think_step(text):
                return False
            if any(
                marker in lowered
                for marker in (
                    "within search",
                    "open advances",
                    "open supports",
                    "examine advances",
                    "examine supports",
                    "wrong target object tends to derail",
                )
            ):
                return False
            if "container=container" in lowered:
                return False
            if _is_concrete_scene_hint(text):
                return False
            return any(
                marker in lowered
                for marker in (
                    "workflow",
                    "closure pattern",
                    "plan",
                    "->",
                    "take(",
                    "go(",
                    "move(",
                    "clean(",
                    "cool(",
                    "heat(",
                    "repeat acquire",
                )
            )

        def _allow_persistent_global() -> bool:
            try:
                required_count = int(getattr(query, "required_count", 0) or 0)
            except Exception:
                required_count = 0
            task_family = _norm(str(getattr(query, "task_family", "") or ""))
            progress = _norm(str(getattr(query, "progress_state", "") or ""))
            goal = _norm(str(getattr(query, "goal", "") or ""))
            return bool(required_count > 1 or "two" in task_family or "two" in goal or "search second" in progress)

        def _rank_key(item: Any) -> tuple[float, float, float, float]:
            return (
                self._memco_core_feedback_adjustment(item) if self._memco_core_feedback_enabled() else 0.0,
                float(getattr(item, "task_relevance", 0.0) or 0.0),
                float(getattr(item, "goal_relevance", 0.0) or 0.0),
                float(getattr(item, "score", 0.0) or 0.0),
            )

        def _select_global_skeleton() -> list[str]:
            items = (
                list(getattr(bundle, "global_task_plan_items", []) or [])
                + list(getattr(bundle, "global_promoted_contribution", []) or [])
                + list(getattr(bundle, "global_items", []) or [])
            )
            ranked = sorted(items, key=_rank_key, reverse=True)
            selected: list[str] = []
            for item in ranked:
                text = _clean(str(getattr(item, "summary", "") or ""))
                if self._memco_core_feedback_item_state(item) == "quarantined":
                    continue
                if not _is_usable_global_skeleton(text):
                    continue
                if text not in selected:
                    selected.append(text)
                    self._memco_core_record_feedback_item(item, slot="global_skeleton", rendered_text=text)
                if len(selected) >= 2:
                    break
            return selected

        key = _skeleton_key()
        if key and key != self._memco_core_episode_global_skeleton_key:
            self._memco_core_episode_global_skeleton_key = key
            self._memco_core_episode_global_skeleton = []
        current_skeleton = _select_global_skeleton() if _allow_persistent_global() else []
        if current_skeleton:
            self._memco_core_episode_global_skeleton = current_skeleton
        persistent_skeleton = list(self._memco_core_episode_global_skeleton) if _allow_persistent_global() else []

        goal_roles = getattr(query, "goal_roles", {}) or {}

        local_graph_weight = max(
            _route_weight("fact", "local"),
            _route_weight("relation", "local"),
        )
        local_domain_weight = max(
            _route_weight("plan", "local"),
            _route_weight("workflow", "local"),
            _route_weight("precondition", "local"),
            _route_weight("closure", "local"),
        )
        global_transfer_weight = max(
            _route_weight("task_plan", "global"),
            _route_weight("workflow", "global"),
            _route_weight("precondition", "global"),
            _route_weight("closure", "global"),
        )
        persistent_global_weight = 0.65 if persistent_skeleton else 0.0

        target_terms = {
            _norm(str(value or ""))
            for value in (
                goal_roles.get("object", ""),
                goal_roles.get("destination", ""),
                goal_roles.get("tool", ""),
                getattr(query, "task_family", ""),
                getattr(query, "progress_state", ""),
                getattr(query, "current_stage", ""),
            )
            if _norm(str(value or ""))
        }
        visible_terms = {_norm(item) for item in visible}
        held_terms = {_norm(item) for item in held}
        exhausted_terms = {_norm(item) for item in exhausted}
        checked_source_terms: set[str] = set()
        for row in list(getattr(env_ref, "current_history", []) or []):
            action_text = _norm(str(row.get("Action", "") or ""))
            observation_text = _norm(str(row.get("Observation", "") or ""))
            target_text = ""
            for prefix in ("open ", "examine ", "go to "):
                if action_text.startswith(prefix):
                    target_text = action_text[len(prefix) :].strip()
                    break
            if not target_text:
                continue
            if action_text.startswith(("open ", "examine ")):
                checked_source_terms.add(target_text)
                continue
            if (
                action_text.startswith("go to ")
                and "is closed" not in observation_text
                and ("you see" in observation_text or "on the" in observation_text or "in it" in observation_text)
            ):
                checked_source_terms.add(target_text)
        sourcehint_enabled = self._external_retrieval_mode == "graph_policy_feedback"
        # GraphPolicy should translate local graph source evidence into
        # current-state action priority, while still leaving the solver/env
        # workflow untouched. This is prompt routing only; repair_action remains
        # exact-admissible and deterministic for all graph_policy modes.
        candidate_mode = self._external_retrieval_mode == "graph_policy_candidate"
        quality_enabled = self._external_retrieval_mode in {
            "graph_policy",
            "graph_policy_rerank",
            "graph_policy_feedback",
            "graph_policy_candidate",
            "graph_policy_quality",
        }
        sourcebase_ranking_enabled = self._external_retrieval_mode in {
            "graph_policy",
            "graph_policy_rerank",
            "graph_policy_candidate",
        }
        searched_source_rerank_enabled = self._external_retrieval_mode in {
            "graph_policy",
            "graph_policy_rerank",
            "graph_policy_candidate",
        }
        if quality_enabled:
            self._memco_core_quality_last_action_candidates = []
        admissible = [
            str(cmd).strip() for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()
        ]
        admissible_blob = _norm(" ".join(admissible[:30]))

        def _item_text(item: Any) -> str:
            return _clean(str(getattr(item, "summary", "") or ""))

        def _is_failure_like(item: Any, text: str = "") -> bool:
            blob = _norm(text or _item_text(item))
            return any(
                marker in blob
                for marker in ("wrong target object", "blocked", "failure", "fails under", "avoid ", "derail")
            )

        def _task_relevance(text: str) -> float:
            lowered = _norm(text)
            if not lowered:
                return 0.0
            score = 0.0
            for term in target_terms:
                if term and term in lowered:
                    score += 0.28
            for term in visible_terms:
                if term and term in lowered:
                    score += 0.16
            for term in held_terms:
                if term and term in lowered:
                    score += 0.2
            if admissible_blob and any(token and token in lowered for token in admissible_blob.split()[:40]):
                score += 0.12
            is_local_scene_prior = _is_concrete_scene_hint(text) and any(
                term and term in lowered
                for term in (
                    _norm(str(goal_roles.get("object", "") or "")),
                    _norm(str(goal_roles.get("destination", "") or "")),
                    _norm(str(goal_roles.get("tool", "") or "")),
                )
            )
            is_positive_source_hint = any(
                marker in lowered
                for marker in ("found at", "was found at", "located at", "visible", "contains", "scene relation")
            )
            if any(term and term in lowered for term in exhausted_terms) and any(
                marker in lowered for marker in ("found at", "located at", "near ", "in ", "on ")
            ):
                score -= 0.45
            elif sourcehint_enabled and is_local_scene_prior and is_positive_source_hint:
                score += 0.3
            elif _is_concrete_scene_hint(text):
                score -= 0.35
            return score

        def _base_location(value: str) -> str:
            text = _norm(str(value or ""))
            text = re.sub(r"\b(?:a|an|the|some)_", "", text)
            text = re.sub(r"_[0-9]+\b", "", text)
            text = re.sub(r"\s+[0-9]+\b", "", text)
            return text.strip()

        def _item_payload(item: Any) -> dict[str, Any]:
            dynamic = getattr(item, "dynamic", {}) or {}
            payload = dynamic.get("payload", {}) if isinstance(dynamic, dict) else {}
            merged: dict[str, Any] = {}
            if isinstance(payload, dict):
                merged.update(payload)
            if isinstance(dynamic, dict):
                merged.update({key: value for key, value in dynamic.items() if key != "payload"})
            return merged

        def _display_location(value: str) -> str:
            return str(value or "").strip().replace("_", " ")

        def _exact_location(value: str) -> str:
            text = str(value or "").strip().lower().replace(" ", "_")
            for prefix in ("a_", "an_", "the_", "some_"):
                if text.startswith(prefix):
                    text = text[len(prefix) :]
            return text

        def _source_instance_from_item(item: Any, text: str) -> str:
            payload = _item_payload(item)
            for key in ("source_instance", "location", "target", "container"):
                value = str(payload.get(key, "") or "").strip()
                if value:
                    return _exact_location(value)
            lowered = _norm(text)
            for pattern in (r"(?:found at|located at|near|in|on|at|from)\s+([a-z_]+(?:_\d+)?|[a-z]+\s+\d+)",):
                match = re.search(pattern, lowered)
                if match:
                    return _exact_location(match.group(1))
            return ""

        def _source_base_from_item(item: Any, text: str) -> str:
            dynamic = _item_payload(item)
            for key in ("source_base", "source_instance"):
                value = str(dynamic.get(key, "") or "").strip()
                if value:
                    return _base_location(value)
            lowered = _norm(text)
            match = re.search(r"(?:found at|located at|near)\s+([a-z_]+(?:_[0-9]+)?)", lowered)
            if match:
                return _base_location(match.group(1))
            for candidate in (
                "countertop",
                "sinkbasin",
                "shelf",
                "fridge",
                "microwave",
                "cabinet",
                "drawer",
                "stoveburner",
                "coffeemachine",
                "garbagecan",
                "diningtable",
            ):
                if candidate in lowered:
                    return candidate
            return ""

        def _source_role_from_item(item: Any, text: str) -> str:
            dynamic = _item_payload(item)
            role = str(dynamic.get("source_role", "") or "").strip()
            if role:
                return role
            base = _source_base_from_item(item, text)
            return self._memco_core_location_role(base) if base else ""

        def _goal_signature_matches_object(item: Any) -> bool:
            target_object = _base_location(str(goal_roles.get("object", "") or ""))
            if not target_object:
                return False
            payload = _item_payload(item)
            anchor = payload.get("anchor", {}) if isinstance(payload.get("anchor", {}), dict) else {}
            blobs = [
                str(anchor.get("goal_signature", "") or ""),
                str(payload.get("goal_signature", "") or ""),
                str(getattr(item, "candidate_id", "") or ""),
            ]
            for blob in blobs:
                signature = _norm(blob)
                if f"goal_signature={target_object}->" in signature or signature.startswith(f"{target_object}->"):
                    return True
            return False

        def _source_type_prior_transfer_level(item: Any) -> str:
            """Classify source-type prior evidence for current target transfer.

            exact:
                The source prior was learned for the current target object.
            role:
                The source prior was learned for the same task family but a
                different object. This is useful for unseen/cross-domain
                transfer only as a weak role-level prior.
            none:
                The prior is not relevant enough for source routing.
            """
            payload = _item_payload(item)
            if str(payload.get("pattern_kind", "") or "") != "source_type_prior":
                return "none"
            target_object = _base_location(str(goal_roles.get("object", "") or ""))
            goal_object = _base_location(str(payload.get("goal_object", "") or ""))
            if not target_object or not goal_object or goal_object == target_object:
                return "exact"

            query_family = _norm(str(getattr(query, "task_family", "") or ""))
            anchor = payload.get("anchor", {}) if isinstance(payload.get("anchor", {}), dict) else {}
            item_family = _norm(
                " ".join(
                    str(value or "")
                    for value in (
                        payload.get("task_family", ""),
                        payload.get("task_type", ""),
                        anchor.get("task_family", ""),
                        getattr(item, "task_family", ""),
                    )
                )
            )
            if query_family and query_family in item_family:
                return "role"
            return "none"

        def _target_take_source_from_item(item: Any) -> str:
            """Return source instance only when graph evidence took the current target.

            A state may list many visible objects. For source ranking we must
            not treat a co-visible target as evidence; the graph edge has to be
            an actual take(object=current_target, source=...).
            """
            if not sourcebase_ranking_enabled:
                return ""
            target_object = _base_location(str(goal_roles.get("object", "") or ""))
            if not target_object:
                return ""
            payload = _item_payload(item)
            relation_kind = str(payload.get("relation_kind", "") or "")
            if relation_kind == "object_location_prior" and _goal_signature_matches_object(item):
                return _exact_location(str(payload.get("source_instance", "") or ""))
            blobs: list[str] = []
            for key in ("graph_refs", "action_patterns"):
                value = payload.get(key, [])
                if isinstance(value, (list, tuple)):
                    blobs.extend(str(part or "") for part in value)
                elif value:
                    blobs.append(str(value))
            for blob in blobs:
                for match in re.finditer(r"take\(object=([a-z0-9_]+),source=([a-z0-9_]+)\)", _norm(blob)):
                    obj = _base_location(match.group(1))
                    source = _exact_location(match.group(2))
                    if obj == target_object and source:
                        return source
            return ""

        def _source_evidence_targets_goal_object(item: Any, text: str) -> bool:
            if not sourcebase_ranking_enabled:
                return True
            if _target_take_source_from_item(item):
                return True
            payload = _item_payload(item)
            if str(payload.get("pattern_kind", "") or "") == "source_type_prior":
                return _source_type_prior_transfer_level(item) in {"exact", "role"}
            if str(payload.get("relation_kind", "") or "") in {"object_location_prior", "searched_empty"}:
                return _goal_signature_matches_object(item)
            return False

        def _source_not_checked(base: str, instance: str = "") -> bool:
            if not base and not instance:
                return False
            instance_norm = _norm(_display_location(instance))
            if instance_norm:
                return (
                    not any(instance_norm == _norm(_display_location(item)) for item in exhausted_terms)
                    and instance_norm not in checked_source_terms
                )
            base_norm = _base_location(base)
            return not any(base_norm and base_norm == _base_location(item) for item in exhausted_terms) and not any(
                base_norm and base_norm == _base_location(item) for item in checked_source_terms
            )

        def _searched_source_type_count(base: str) -> int:
            """Count source instances of this base already explored in this episode.

            This is used only by graph_policy modes. It is deliberately
            episode-local: it does not alter persisted local/global memory.
            """
            if not searched_source_rerank_enabled:
                return 0
            base_norm = _base_location(base)
            if not base_norm:
                return 0
            checked: set[str] = set()
            for item in tuple(exhausted_terms) + tuple(checked_source_terms):
                item_base = _base_location(item)
                if item_base == base_norm:
                    checked.add(_exact_location(item) or _norm(item))
            return len(checked)

        def _searched_source_penalty(base: str, instance: str = "") -> float:
            if not searched_source_rerank_enabled:
                return 0.0
            penalty = 0.0
            instance_norm = _norm(_display_location(instance))
            if instance_norm and (
                any(instance_norm == _norm(_display_location(item)) for item in exhausted_terms)
                or instance_norm in checked_source_terms
            ):
                penalty += 0.55
            count = _searched_source_type_count(base or instance)
            if count >= 1:
                penalty += min(0.42, 0.07 * count)
            # Long cabinet/drawer/shelf sweeps are the dominant failure mode in
            # living unseen two-object tasks. Once several instances of the
            # same broad source type have been checked, prefer remaining source
            # types with positive evidence before continuing that sweep.
            if _base_location(base or instance) in {"shelf", "drawer", "cabinet"} and count >= 3:
                penalty += min(0.24, 0.04 * (count - 2))
            return min(penalty, 0.75)

        def _is_goal_destination_source(base: str, instance: str = "") -> bool:
            destination = _base_location(str(goal_roles.get("destination", "") or ""))
            if not destination or getattr(query, "held_relevant_count", 0) > 0:
                return False
            return destination in {_base_location(base), _base_location(instance)}

        def _goal_destination_source_penalty(base: str, instance: str = "") -> float:
            """Keep goal destination as a weak source candidate instead of dropping it.

            Some ALFWorld tasks place the target object on/in the eventual goal
            destination, especially support surfaces such as sofa/shelf. The
            previous hard filter improved some seen cases but hurt transfer:
            global role-level memory could not suggest sofa/fridge as a place
            to search. Penalize it instead, so stronger local evidence wins
            while unseen/cross-domain runs still get a fallback.
            """
            if not _is_goal_destination_source(base, instance):
                return 0.0
            role = self._memco_core_location_role(_base_location(base or instance))
            return 0.12 if role == "support_surface" else 0.22

        def _is_processing_tool_source(base: str, instance: str = "") -> bool:
            tool = _base_location(str(goal_roles.get("tool", "") or ""))
            if not tool:
                return False
            base_norm = _base_location(base)
            instance_norm = _base_location(instance)
            if tool not in {base_norm, instance_norm}:
                return False
            # In search states, a clean/heat/cool tool is where the object is
            # processed after acquisition, not evidence that the target starts
            # there. If the target is actually visible there, local grounding
            # will handle it separately.
            goal_object = _norm(str(goal_roles.get("object", "") or ""))
            if goal_object and any(goal_object in _norm(item) for item in visible):
                return False
            return getattr(query, "held_relevant_count", 0) <= 0

        def _quality_confidence(item: Any) -> float:
            positive = float(getattr(item, "positive", 0) or 0)
            negative = float(getattr(item, "negative", 0) or 0)
            total = positive + negative
            if total > 0:
                return positive / total
            score = float(getattr(item, "score", 0.0) or 0.0)
            return max(0.0, min(score, 1.0))

        def _quality_support(item: Any) -> int:
            return int((getattr(item, "positive", 0) or 0) + (getattr(item, "negative", 0) or 0))

        def _is_positive_local_source_hint(item: Any, text: str) -> bool:
            lowered = _norm(text)
            dynamic = _item_payload(item)
            relation_kind = str(dynamic.get("relation_kind", "") or "").lower()
            if relation_kind and relation_kind != "object_location_prior":
                return False
            if "not found" in lowered or "empty" in lowered:
                return False
            return any(
                marker in lowered
                for marker in ("found at", "was found at", "located at", "visible", "contains", "scene relation")
            )

        def _search_action_target(command: str) -> tuple[str, str]:
            lowered = _norm(command)
            for prefix in ("go to ", "open ", "examine "):
                if lowered.startswith(prefix):
                    target = lowered[len(prefix) :].strip()
                    return _base_location(target), _exact_location(target)
            return "", ""

        def _admissible_search_actions(base: str, instance: str, *, allow_checked: bool = False) -> list[str]:
            invalid_bases = {"", "room", "current", "unknown", "middle", "inventory", "floor"}
            base_norm = _base_location(base)
            if base_norm in invalid_bases:
                return []
            instance_norm = _norm(_display_location(instance))
            candidates: list[tuple[int, int, str]] = []
            for command in admissible:
                lowered = _norm(command)
                priority = 0
                for prefix, weight in (("go to ", 3), ("open ", 2), ("examine ", 1)):
                    if not lowered.startswith(prefix):
                        continue
                    target = lowered[len(prefix) :].strip()
                    target_base = _base_location(target)
                    target_exact = _norm(_display_location(target))
                    if (
                        target_exact
                        and not allow_checked
                        and not _source_not_checked(target_base, target_exact.replace(" ", "_"))
                    ):
                        continue
                    if instance_norm and target_exact == instance_norm:
                        priority = weight + 3
                    elif not instance_norm and base_norm and target_base == base_norm:
                        priority = weight
                    if priority > 0:
                        ordinal = 999
                        match = re.search(r"\b(\d+)\b", target_exact)
                        if match:
                            try:
                                ordinal = int(match.group(1))
                            except Exception:
                                ordinal = 999
                        candidates.append((priority, ordinal, command))
                    break
            return [cmd for _priority, _ordinal, cmd in sorted(candidates, key=lambda pair: (-pair[0], pair[1]))]

        def _source_queue_key() -> str:
            return "|".join(
                _norm(str(value or ""))
                for value in (
                    getattr(query, "task_family", ""),
                    goal_roles.get("object", ""),
                    goal_roles.get("destination", ""),
                    goal_roles.get("tool", ""),
                    getattr(query, "required_count", ""),
                    getattr(query, "goal", ""),
                )
            )

        def _source_queue_signature(base: str, instance: str) -> str:
            return f"{_base_location(base)}::{_exact_location(instance)}"

        def _source_queue_active(entry: dict[str, Any]) -> dict[str, Any] | None:
            base = str(entry.get("base", "") or "")
            instance = str(entry.get("instance", "") or "")
            allow_checked = bool(entry.get("allow_checked_source", False))
            if not allow_checked and not _source_not_checked(base, instance):
                return None
            actions = _admissible_search_actions(base, instance, allow_checked=allow_checked)
            if not actions:
                return None
            refreshed = dict(entry)
            refreshed["action"] = actions[0]
            return refreshed

        def _gate_source_queue_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Keep role-level transfer as backoff, not as a replacement for exact graph evidence."""
            if not sourcebase_ranking_enabled:
                return candidates

            def _entry_score(entry: dict[str, Any]) -> float:
                try:
                    return float(entry.get("score", 0.0) or 0.0)
                except Exception:
                    return 0.0

            def _entry_kind(entry: dict[str, Any]) -> str:
                scope = str(entry.get("source_scope", "") or "")
                transfer_level = str(entry.get("transfer_level", "") or "exact")
                if scope == "previous_success_source":
                    return "previous"
                if transfer_level == "exact":
                    return "exact"
                if transfer_level == "role":
                    return "role_global" if scope == "global" else "role_local"
                return "fallback"

            def _is_strong_exact(entry: dict[str, Any]) -> bool:
                if _entry_kind(entry) == "previous":
                    return True
                if _entry_kind(entry) != "exact":
                    return False
                try:
                    score = float(entry.get("score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
                try:
                    confidence = float(entry.get("confidence", 0.0) or 0.0)
                except Exception:
                    confidence = 0.0
                try:
                    support = int(float(entry.get("support", 0) or 0))
                except Exception:
                    support = 0
                scope = str(entry.get("source_scope", "") or "")
                base = _base_location(str(entry.get("base", "") or entry.get("instance", "") or ""))
                if (
                    self._external_retrieval_mode == "graph_policy"
                    and base in {"drawer", "cabinet", "shelf"}
                    and _searched_source_type_count(base) >= 2
                ):
                    return False
                if scope in {"local_scene_relation", "previous_success_source"}:
                    return score >= 0.72 or confidence >= 0.62 or support >= 1
                if scope == "local":
                    return score >= 0.82 or (confidence >= 0.62 and support >= 2)
                return score >= 0.9 and confidence >= 0.62 and support >= 2

            ordered = sorted(candidates, key=_entry_score, reverse=True)
            previous = [entry for entry in ordered if _entry_kind(entry) == "previous"]
            exact = [entry for entry in ordered if _entry_kind(entry) == "exact"]
            strong_exact = [entry for entry in exact if _is_strong_exact(entry)]
            role_local = [entry for entry in ordered if _entry_kind(entry) == "role_local"]
            role_global = [entry for entry in ordered if _entry_kind(entry) == "role_global"]
            fallback = [entry for entry in ordered if _entry_kind(entry) == "fallback"]

            gated: list[dict[str, Any]] = []

            def _add(entries: list[dict[str, Any]], max_new: int | None = None, reason: str = "") -> None:
                added = 0
                seen = {
                    str(
                        entry.get("signature", "")
                        or _source_queue_signature(entry.get("base", ""), entry.get("instance", ""))
                    )
                    for entry in gated
                }
                for entry in entries:
                    sig = str(
                        entry.get("signature", "")
                        or _source_queue_signature(entry.get("base", ""), entry.get("instance", ""))
                    )
                    if sig in seen:
                        continue
                    enriched = dict(entry)
                    if reason and not enriched.get("selected_reason"):
                        enriched["selected_reason"] = reason
                    gated.append(enriched)
                    seen.add(sig)
                    added += 1
                    if max_new is not None and added >= max_new:
                        break

            # In two-object tasks the source that yielded the first target is
            # the strongest local clue for finding the second one. Keep it in
            # front of broader source-type transfer.
            if candidate_mode:
                _add(previous[:1], reason="episode-local previous target source")
                _add(strong_exact, reason="strong current-state graph candidate")
                if len(gated) < 2:
                    weak_exact = [entry for entry in exact if entry not in strong_exact]
                    _add(weak_exact, max_new=1, reason="weak exact local candidate retained as context")
                _add(role_local, max_new=1, reason="local role-level candidate frontier")
                _add(role_global, max_new=2, reason="global phase-conditioned candidate frontier")
                _add(fallback, max_new=2, reason="source-base backoff frontier")
                return gated

            _add(previous[:1], reason="episode-local previous target source")
            _add(exact, reason="exact current-state source candidate")

            strong_exact_count = len(previous[:1]) + len(strong_exact)
            exact_count = len(gated)
            if strong_exact_count >= 2:
                return gated

            # Role-level priors are useful for unseen/cross-domain transfer,
            # but only as backoff when exact target-object source evidence is
            # missing, sparse, or too weak to confidently suppress transfer.
            if strong_exact_count == 1:
                _add(role_local, max_new=1, reason="local role-level backoff after sparse exact evidence")
                if len(gated) < 2:
                    _add(role_global, max_new=1, reason="global role-level frontier after sparse exact evidence")
                if len(gated) < 2:
                    _add(fallback, max_new=1, reason="source-base fallback after sparse exact evidence")
                return gated

            _add(role_local, max_new=2, reason="local role-level frontier")
            if len(gated) < 2:
                _add(role_global, max_new=1, reason="global role-level frontier")
            if len(gated) < 2:
                _add(fallback, max_new=2, reason="source-base fallback frontier")
            return gated

        def _reset_source_queue_if_needed() -> None:
            queue_key = _source_queue_key()
            if queue_key and queue_key != self._memco_core_graph_policy_source_key:
                self._memco_core_graph_policy_source_key = queue_key
                self._memco_core_graph_policy_source_queue = []
                self._memco_core_graph_policy_next_source_action = ""

        def _merge_source_queue(candidates: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
            _reset_source_queue_if_needed()
            merged: dict[str, dict[str, Any]] = {}
            for entry in self._memco_core_graph_policy_source_queue:
                active = _source_queue_active(entry)
                if not active:
                    continue
                sig = str(
                    active.get("signature", "")
                    or _source_queue_signature(active.get("base", ""), active.get("instance", ""))
                )
                active["signature"] = sig
                merged[sig] = active
            for entry in candidates:
                active = _source_queue_active(entry)
                if not active:
                    continue
                sig = str(
                    active.get("signature", "")
                    or _source_queue_signature(active.get("base", ""), active.get("instance", ""))
                )
                active["signature"] = sig
                previous = merged.get(sig)
                if previous is None or float(active.get("score", 0.0) or 0.0) > float(
                    previous.get("score", 0.0) or 0.0
                ):
                    merged[sig] = active
            queue_candidates = sorted(
                merged.values(),
                key=lambda entry: (
                    -float(entry.get("score", 0.0) or 0.0),
                    str(entry.get("base", "")),
                    str(entry.get("instance", "")),
                ),
            )
            queue_candidates = _gate_source_queue_candidates(queue_candidates)
            queue = queue_candidates[:limit]
            self._memco_core_graph_policy_source_queue = queue
            self._memco_core_graph_policy_next_source_action = str(queue[0].get("action", "") or "") if queue else ""
            return queue

        def _source_base_stats(items: list[Any]) -> dict[str, dict[str, float]]:
            """Estimate transferable source_base utility from local graph memory.

            This is graph_policy-only. It turns many specific location hints
            into a base-level distribution such as countertop/cabinet/drawer,
            with success-like evidence increasing the base score and
            failure/empty evidence decreasing it.
            """
            if not sourcebase_ranking_enabled:
                return {}
            stats: dict[str, dict[str, float]] = {}
            target_object = _norm(str(goal_roles.get("object", "") or ""))
            for item in items:
                payload = _item_payload(item)
                is_source_type_prior = str(payload.get("pattern_kind", "") or "") == "source_type_prior"
                if not _source_is_local(item) and not is_source_type_prior:
                    continue
                text = _item_text(item)
                if not text:
                    continue
                if target_object and not _source_evidence_targets_goal_object(item, text):
                    continue
                target_take_source = _target_take_source_from_item(item)
                instance = target_take_source or _source_instance_from_item(item, text)
                base = _base_location(instance) if target_take_source else _source_base_from_item(item, text)
                if not base or _is_processing_tool_source(base, instance):
                    continue
                lowered = _norm(text)
                if (
                    target_object
                    and not target_take_source
                    and target_object not in lowered
                    and _quality_goal_match(item, text) < 0.22
                ):
                    continue
                transfer_level = _source_type_prior_transfer_level(item) if is_source_type_prior else "exact"
                transfer_factor = 1.0
                if transfer_level == "role":
                    # Role-level source priors are the bridge for unseen and
                    # cross-domain tasks, but they must not overpower exact
                    # object/location evidence.
                    transfer_factor = 0.46 if _source_is_global(item) else 0.34
                positive = float(getattr(item, "positive", 0) or 0)
                negative = float(getattr(item, "negative", 0) or 0)
                support = positive + negative
                score = float(getattr(item, "score", 0.0) or 0.0)
                goal_match = _quality_goal_match(item, text)
                search_phase = str(getattr(query, "progress_state", "") or "").startswith("search")
                if _source_is_global(item):
                    # Routing weights should control how much global transfer
                    # can shape source search. Global source-type priors help
                    # unseen transfer, but only when the current route assigns
                    # real weight to global evidence.
                    source_weight = (0.45 if search_phase else 0.35) + (0.55 if search_phase else 0.65) * max(
                        0.0, min(global_transfer_weight, 1.0)
                    )
                else:
                    source_weight = 0.55 + 0.45 * max(
                        0.0,
                        min(max(local_graph_weight, local_domain_weight), 1.0),
                    )
                success_like = 0.0
                failure_like = 0.0
                if _is_positive_local_source_hint(item, text) or (is_source_type_prior and positive > negative):
                    success_like += max(1.0, positive)
                if positive > negative:
                    success_like += min(positive - negative, 3.0) * 0.5
                if (
                    "not found" in lowered
                    or "empty" in lowered
                    or _is_failure_like(item, text)
                    or (is_source_type_prior and negative > positive)
                ):
                    failure_like += max(1.0, negative)
                if negative > positive:
                    failure_like += min(negative - positive, 3.0) * 0.5
                if success_like <= 0.0 and failure_like <= 0.0 and support <= 0.0:
                    # A weak retrieved relation can still contribute, but only
                    # as a tiny prior if it is task-relevant.
                    success_like += max(0.0, min(score, 0.6)) * max(0.0, min(goal_match, 1.0))
                success_like *= transfer_factor
                failure_like *= transfer_factor
                destination_penalty = _goal_destination_source_penalty(base, instance)
                stat = stats.setdefault(
                    base,
                    {"success": 0.0, "failure": 0.0, "support": 0.0, "score": 0.0, "goal": 0.0},
                )
                stat["success"] += source_weight * success_like
                stat["failure"] += source_weight * (failure_like + destination_penalty)
                stat["support"] += support
                stat["score"] = max(stat["score"], score)
                stat["goal"] = max(stat["goal"], goal_match)
            return stats

        def _source_base_rank(base: str, stats: dict[str, dict[str, float]]) -> float:
            if not sourcebase_ranking_enabled:
                return 0.0
            stat = stats.get(_base_location(base), {})
            if not stat:
                return 0.0
            success = float(stat.get("success", 0.0) or 0.0)
            failure = float(stat.get("failure", 0.0) or 0.0)
            goal_match = float(stat.get("goal", 0.0) or 0.0)
            score = float(stat.get("score", 0.0) or 0.0)
            support = float(stat.get("support", 0.0) or 0.0)
            rank = (
                min(success, 5.0) * 0.12
                - min(failure, 5.0) * 0.14
                + min(goal_match, 1.2) * 0.18
                + min(score, 1.0) * 0.1
                + min(support, 6.0) * 0.015
            )
            rank -= _searched_source_penalty(base)
            if base in {"cabinet", "drawer"} and success <= 0.0:
                rank -= 0.12
            return max(-0.35, min(rank, 0.55))

        def _direct_local_scene_relation_candidates() -> list[dict[str, Any]]:
            """Read exact target-object source relations from the local graph.

            QueryBasedRetriever keeps top-k support compact, so exact
            scene_relation artifacts can be absent from the routed bundle even
            when the local graph contains them. For graph_policy source
            priority, scan only the structured local artifacts that say the
            current target_object was actually found at a source in a matching
            task family/goal signature. This avoids co-visible-object noise.
            """
            if not sourcebase_ranking_enabled or local_memory is None:
                return []
            target_object = _base_location(str(goal_roles.get("object", "") or ""))
            goal_destination = _base_location(str(goal_roles.get("destination", "") or ""))
            task_family = _norm(str(getattr(query, "task_family", "") or ""))
            if not target_object:
                return []
            artifacts = getattr(local_memory, "artifacts_by_id", {}) or {}
            artifact_values = artifacts.values() if isinstance(artifacts, dict) else artifacts
            candidates: dict[str, dict[str, Any]] = {}
            for artifact in artifact_values:
                payload = getattr(artifact, "payload", {}) or {}
                anchor = getattr(artifact, "anchor", {}) or {}
                if str(payload.get("pattern_kind", "") or "") != "scene_relation":
                    continue
                if str(payload.get("relation_kind", "") or "") != "object_location_prior":
                    continue
                if str(payload.get("object_role", "") or "") != "target_object":
                    continue
                anchor_family = _norm(str(anchor.get("task_family", "") or ""))
                if task_family and anchor_family and anchor_family != task_family:
                    continue
                goal_signature = _norm(str(anchor.get("goal_signature", "") or ""))
                if not goal_signature.startswith(f"{target_object}->"):
                    continue
                exact_goal_bonus = 0.0
                if goal_destination and goal_signature == f"{target_object}->{goal_destination}":
                    exact_goal_bonus = 0.16
                source_instance = _exact_location(str(payload.get("source_instance", "") or ""))
                source_base = _base_location(str(payload.get("source_base", "") or source_instance))
                if not source_base or _is_processing_tool_source(source_base, source_instance):
                    continue
                actions = _admissible_search_actions(source_base, source_instance)
                if not actions and source_instance:
                    actions = _admissible_search_actions(source_base, "")
                if not actions:
                    continue
                stats = getattr(artifact, "stats", None)
                support = float(getattr(stats, "support", 0) or 0)
                success = float(getattr(stats, "success", 0) or 0)
                failure = float(getattr(stats, "failure", 0) or 0)
                confidence = float(getattr(stats, "confidence", 0.0) or 0.0)
                role_bonus = 0.12 if str(payload.get("source_role", "") or "") == "support_surface" else 0.0
                container_penalty = 0.05 if source_base in {"cabinet", "drawer"} else 0.0
                destination_penalty = _goal_destination_source_penalty(source_base, source_instance)
                score_base = (
                    0.78
                    + exact_goal_bonus
                    + role_bonus
                    + min(support, 4.0) * 0.04
                    + min(success, 3.0) * 0.04
                    + min(confidence, 1.0) * 0.08
                    - min(failure, 3.0) * 0.08
                    - container_penalty
                    - destination_penalty
                )
                for action_index, action in enumerate(actions[:4]):
                    action_base, action_instance = _search_action_target(action)
                    instance = action_instance or source_instance
                    if not instance:
                        continue
                    score = (
                        score_base
                        - 0.04 * action_index
                        - _searched_source_penalty(action_base or source_base, instance)
                    )
                    sig = _source_queue_signature(action_base or source_base, instance)
                    candidate = {
                        "signature": sig,
                        "base": action_base or source_base,
                        "instance": instance,
                        "action": action,
                        "score": score,
                        "confidence": confidence,
                        "support": support,
                        "route_scale": max(0.0, min(local_graph_weight, 1.0)),
                        "transfer_level": "exact",
                        "source_scope": "local_scene_relation",
                        "pattern_kind": "scene_relation",
                        "strength": (
                            "direct local graph relation for current target_object "
                            f"(goal_signature={goal_signature}, support={support:.0f}, confidence={confidence:.2f})"
                        ),
                    }
                    previous = candidates.get(sig)
                    if previous is None or score > float(previous.get("score", 0.0) or 0.0):
                        candidates[sig] = candidate
            return list(candidates.values())

        def _previous_target_source_candidates() -> list[dict[str, Any]]:
            """For two-object tasks, revisit the source that yielded the first target.

            This is deliberately episode-local and graph_policy-only. It does
            not change stored memory; it only prevents broad role-level
            transfer from overriding a strong within-episode clue in
            `search_second`.
            """
            if not sourcebase_ranking_enabled:
                return []
            progress = str(getattr(query, "progress_state", "") or "")
            if "second" not in progress:
                return []
            if getattr(query, "held_relevant_count", 0) > 0:
                return []
            target_object = _base_location(str(goal_roles.get("object", "") or ""))
            if not target_object:
                return []
            target_pattern = re.escape(target_object.replace("_", " "))
            seen_sources: list[tuple[str, str]] = []
            for row in reversed(list(getattr(env_ref, "current_history", []) or [])):
                action_text = self._normalize_action_text(str(row.get("Action", "") or "")).replace("_", " ")
                match = re.search(
                    rf"\btake\s+{target_pattern}(?:\s+\d+)?\s+from\s+([a-z][a-z0-9]*(?:\s+\d+)?)\b",
                    action_text,
                )
                if not match:
                    continue
                source_instance = _exact_location(match.group(1))
                source_base = _base_location(source_instance)
                if not source_base or _is_processing_tool_source(source_base, source_instance):
                    continue
                key = (source_base, source_instance)
                if key not in seen_sources:
                    seen_sources.append(key)
                if len(seen_sources) >= 2:
                    break

            candidates: dict[str, dict[str, Any]] = {}
            for source_index, (source_base, source_instance) in enumerate(seen_sources):
                actions = _admissible_search_actions(source_base, source_instance, allow_checked=True)
                if not actions and source_instance:
                    actions = _admissible_search_actions(source_base, "", allow_checked=True)
                for action_index, action in enumerate(actions[:2]):
                    action_base, action_instance = _search_action_target(action)
                    instance = action_instance or source_instance
                    if not instance:
                        continue
                    score = (
                        2.05
                        - 0.08 * source_index
                        - 0.04 * action_index
                        - _goal_destination_source_penalty(action_base or source_base, instance)
                    )
                    sig = _source_queue_signature(action_base or source_base, instance)
                    candidate = {
                        "signature": sig,
                        "base": action_base or source_base,
                        "instance": instance,
                        "action": action,
                        "score": score,
                        "confidence": 0.86,
                        "support": 1,
                        "route_scale": 1.0,
                        "transfer_level": "exact",
                        "source_scope": "previous_success_source",
                        "pattern_kind": "previous_success_source",
                        "allow_checked_source": True,
                        "strength": (
                            "two-object search_second: previous successful target source "
                            f"`{_display_location(source_instance)}` should be checked before broad transfer"
                        ),
                    }
                    previous = candidates.get(sig)
                    if previous is None or score > float(previous.get("score", 0.0) or 0.0):
                        candidates[sig] = candidate
            return list(candidates.values())

        def _graph_policy_global_backoff_allowed() -> bool:
            if self._external_retrieval_mode != "graph_policy":
                return False
            progress = str(getattr(query, "progress_state", "") or "")
            if not progress.startswith("search"):
                return False
            if getattr(query, "held_relevant_count", 0) > 0:
                return False
            if bool(getattr(query, "goal_object_matches_visible", False)):
                return False
            searched_bases = [
                _base_location(item)
                for item in tuple(exhausted_terms) + tuple(checked_source_terms)
                if _base_location(item)
            ]
            if len(set(searched_bases)) >= 3:
                return True
            return any(searched_bases.count(base) >= 3 for base in {"drawer", "cabinet", "shelf"})

        def _select_quality_source_priors(items: list[Any], *, limit: int = 2) -> list[str]:
            if not quality_enabled:
                self._memco_core_graph_policy_source_queue = []
                self._memco_core_graph_policy_next_source_action = ""
                return []
            if getattr(query, "held_relevant_count", 0) > 0:
                self._memco_core_graph_policy_source_queue = []
                self._memco_core_graph_policy_next_source_action = ""
                return []
            progress = str(getattr(query, "progress_state", "") or "")
            if (
                progress
                and not progress.startswith("search")
                and progress not in {"locate_target", "inspect_container"}
            ):
                self._memco_core_graph_policy_source_queue = []
                self._memco_core_graph_policy_next_source_action = ""
                return []
            if bool(getattr(query, "goal_object_matches_visible", False)):
                self._memco_core_graph_policy_source_queue = []
                self._memco_core_graph_policy_next_source_action = ""
                return []

            base_stats = _source_base_stats(items)
            ranked: dict[str, dict[str, Any]] = {}
            for item in items:
                payload = _item_payload(item)
                is_source_type_prior = str(payload.get("pattern_kind", "") or "") == "source_type_prior"
                if not _source_is_local(item) and not is_source_type_prior:
                    continue
                text = _item_text(item)
                if not text or (not _is_positive_local_source_hint(item, text) and not is_source_type_prior):
                    if not (sourcebase_ranking_enabled and _target_take_source_from_item(item)):
                        continue
                if self._memco_core_feedback_item_state(item) == "quarantined":
                    continue
                graph_policy_global_backoff = (
                    self._external_retrieval_mode == "graph_policy"
                    and is_source_type_prior
                    and _source_is_global(item)
                    and _graph_policy_global_backoff_allowed()
                )
                if (
                    self._external_retrieval_mode == "graph_policy"
                    and is_source_type_prior
                    and _source_is_global(item)
                    and not graph_policy_global_backoff
                ):
                    # Keep global abstract during early search. Once local
                    # source evidence has repeatedly missed, allow it back only
                    # as a weak source-role backoff.
                    continue
                if sourcebase_ranking_enabled and not _source_evidence_targets_goal_object(item, text):
                    continue
                target_take_source = _target_take_source_from_item(item)
                instance = target_take_source or _source_instance_from_item(item, text)
                base = _base_location(instance) if target_take_source else _source_base_from_item(item, text)
                if not base:
                    continue
                if _is_processing_tool_source(base, instance):
                    continue
                if _is_goal_destination_source(base, instance) and not sourcebase_ranking_enabled:
                    continue
                actions = _admissible_search_actions(base, instance)
                if not actions and instance:
                    actions = _admissible_search_actions(base, "")
                if not actions:
                    continue
                support = _quality_support(item)
                adjusted_confidence = (float(getattr(item, "positive", 0) or 0) + 0.5) / max(1.0, support + 1.0)
                # Single-shot scene relations are allowed only when they map to
                # a current admissible search action with enough retriever
                # support. Otherwise they become noisy scene memorization.
                if support <= 0 and float(getattr(item, "score", 0.0) or 0.0) < 0.55:
                    continue
                if support == 1 and float(getattr(item, "score", 0.0) or 0.0) < 0.58:
                    continue
                if support >= 2 and adjusted_confidence < 0.55:
                    continue
                role = _source_role_from_item(item, text)
                role_bonus = 0.12 if role == "support_surface" else 0.0
                container_penalty = 0.04 if base in {"cabinet", "drawer"} else 0.0
                destination_penalty = (
                    _goal_destination_source_penalty(base, instance) if sourcebase_ranking_enabled else 0.0
                )
                base_rank_bonus = _source_base_rank(base, base_stats)
                type_prior_bonus = 0.14 if is_source_type_prior else 0.0
                global_prior_penalty = 0.03 if is_source_type_prior and _source_is_global(item) else 0.0
                transfer_level = (
                    _source_type_prior_transfer_level(item)
                    if sourcebase_ranking_enabled and is_source_type_prior
                    else "exact"
                )
                if graph_policy_global_backoff:
                    transfer_level = "role"
                role_level_penalty = 0.0
                if transfer_level == "role":
                    role_level_penalty = 0.18 if _source_is_global(item) else 0.28
                if searched_source_rerank_enabled and is_source_type_prior and _source_is_global(item):
                    # In rerank mode, global source-type memory is a weak
                    # transfer prior. It can break ties, but local episode
                    # evidence and already-searched sources should dominate.
                    global_prior_penalty += 0.12 if self._external_retrieval_mode == "graph_policy" else 0.05
                route_scale = (
                    0.7 + 0.3 * max(0.0, min(global_transfer_weight, 1.0))
                    if _source_is_global(item)
                    else 0.7 + 0.3 * max(0.0, min(max(local_graph_weight, local_domain_weight), 1.0))
                )
                base_score = route_scale * (
                    float(getattr(item, "score", 0.0) or 0.0)
                    + 0.35 * adjusted_confidence
                    + min(support, 4) * 0.04
                    + role_bonus
                    + base_rank_bonus
                    + type_prior_bonus
                    - container_penalty
                    - destination_penalty
                    - global_prior_penalty
                    - role_level_penalty
                )
                if is_source_type_prior:
                    origin = "global" if _source_is_global(item) else "local"
                    transfer_note = "role-level transfer, " if transfer_level == "role" else ""
                    evidence_strength = (
                        f"{origin} {transfer_note}source-type prior "
                        f"support={support}, confidence={adjusted_confidence:.2f}"
                    )
                else:
                    evidence_strength = (
                        "single-step local graph evidence"
                        if support <= 0
                        else (
                            "single-episode local graph evidence"
                            if support == 1
                            else f"support={support}, confidence={adjusted_confidence:.2f}"
                        )
                    )
                for action_index, action in enumerate(actions[:4]):
                    action_base, action_instance = _search_action_target(action)
                    candidate_instance = action_instance or instance
                    if not candidate_instance:
                        continue
                    exact_bonus = (
                        0.18
                        if instance
                        and _norm(_display_location(instance)) == _norm(_display_location(candidate_instance))
                        else 0.0
                    )
                    expansion_penalty = 0.05 * action_index
                    score = (
                        base_score
                        + exact_bonus
                        - expansion_penalty
                        - _searched_source_penalty(action_base or base, candidate_instance)
                    )
                    sig = _source_queue_signature(action_base or base, candidate_instance)
                    candidate = {
                        "signature": sig,
                        "base": action_base or base,
                        "instance": candidate_instance,
                        "action": action,
                        "score": score,
                        "confidence": adjusted_confidence,
                        "support": support,
                        "route_scale": route_scale,
                        "transfer_level": transfer_level,
                        "source_scope": "global" if _source_is_global(item) else "local",
                        "pattern_kind": "source_type_prior" if is_source_type_prior else "scene_relation",
                        "strength": (
                            evidence_strength
                            if exact_bonus
                            else f"source-base distribution from `{_display_location(base)}` ({evidence_strength})"
                        ),
                    }
                    current = ranked.get(sig)
                    if current is None or score > float(current.get("score", 0.0) or 0.0):
                        ranked[sig] = candidate
                        rendered = (
                            f"graph source-type `{_display_location(base)}` supports trying `{action}`; "
                            "advance through unvisited instances of this source_base before broad cabinet/drawer sweeps "
                            f"unless contradicted ({candidate['strength']})"
                        )
                        self._memco_core_record_feedback_item(item, slot="quality_source_prior", rendered_text=rendered)
            if sourcebase_ranking_enabled:
                for candidate in _previous_target_source_candidates():
                    sig = str(candidate.get("signature", "") or "")
                    if not sig:
                        continue
                    current = ranked.get(sig)
                    if current is None or float(candidate.get("score", 0.0) or 0.0) > float(
                        current.get("score", 0.0) or 0.0
                    ):
                        ranked[sig] = candidate
                for candidate in _direct_local_scene_relation_candidates():
                    sig = str(candidate.get("signature", "") or "")
                    if not sig:
                        continue
                    current = ranked.get(sig)
                    if current is None or float(candidate.get("score", 0.0) or 0.0) > float(
                        current.get("score", 0.0) or 0.0
                    ):
                        ranked[sig] = candidate
            if sourcebase_ranking_enabled:
                for base, stat in base_stats.items():
                    if _is_processing_tool_source(base):
                        continue
                    base_rank = _source_base_rank(base, base_stats)
                    if base_rank < 0.08:
                        continue
                    actions = _admissible_search_actions(base, "")
                    if not actions:
                        continue
                    for action_index, action in enumerate(actions[:5]):
                        action_base, action_instance = _search_action_target(action)
                        if not action_instance:
                            continue
                        score = (
                            0.45
                            + base_rank
                            + min(float(stat.get("goal", 0.0) or 0.0), 1.0) * 0.08
                            - action_index * 0.045
                            - _searched_source_penalty(action_base or base, action_instance)
                            - _goal_destination_source_penalty(action_base or base, action_instance)
                        )
                        sig = _source_queue_signature(action_base or base, action_instance)
                        candidate = {
                            "signature": sig,
                            "base": action_base or base,
                            "instance": action_instance,
                            "action": action,
                            "score": score,
                            "confidence": 0.5 + min(max(base_rank, 0.0), 0.5),
                            "support": int(float(stat.get("support", 0.0) or 0.0)),
                            "route_scale": max(0.0, min(max(local_graph_weight, global_transfer_weight), 1.0)),
                            "transfer_level": "fallback",
                            "source_scope": "source_base_rank",
                            "pattern_kind": "source_base_rank",
                            "strength": (
                                f"source-base rank for `{_display_location(base)}` "
                                f"(success={stat.get('success', 0.0):.1f}, failure={stat.get('failure', 0.0):.1f})"
                            ),
                        }
                        current = ranked.get(sig)
                        if current is None or score > float(current.get("score", 0.0) or 0.0):
                            ranked[sig] = candidate
            selected_candidates = sorted(
                ranked.values(),
                key=lambda entry: float(entry.get("score", 0.0) or 0.0),
                reverse=True,
            )
            self._memco_core_quality_source_prior_candidates = [dict(entry) for entry in selected_candidates[:6]]
            queue = _merge_source_queue(selected_candidates)
            if not queue:
                self._memco_core_quality_last_action_candidates = []
                return []
            next_entry = queue[0]
            alternates = queue[1:limit]
            self._memco_core_quality_last_action_candidates = [
                str(entry.get("action", "") or "")
                for entry in queue[:limit]
                if str(entry.get("action", "") or "").strip()
            ]
            lines = [
                (
                    "next graph-supported source action: "
                    f"`{next_entry.get('action')}` for source `{_display_location(str(next_entry.get('instance', '') or ''))}`; "
                    "take it before continuing broad cabinet/drawer sweeps unless the latest observation contradicts it "
                    f"({next_entry.get('strength')}; selected_reason={next_entry.get('selected_reason', 'score_rank')})"
                )
            ]
            if searched_source_rerank_enabled:
                searched_bases = sorted(
                    {
                        _base_location(item)
                        for item in tuple(exhausted_terms) + tuple(checked_source_terms)
                        if _base_location(item)
                    }
                )
                if searched_bases:
                    lines.append(
                        "episode searched-source rerank is active; already checked source types are lower priority now: "
                        + ", ".join(searched_bases[:6])
                    )
            if alternates:
                lines.append(
                    "queued alternate source actions after the next one: "
                    + " | ".join(
                        f"`{entry.get('action')}`" for entry in alternates if str(entry.get("action", "") or "").strip()
                    )
                )
            return lines

        def _select_quality_search_cautions(source_priors: list[str]) -> list[str]:
            if not quality_enabled:
                return []
            if getattr(query, "held_relevant_count", 0) > 0:
                return []
            counts = dynamic.get("search_attempt_counts", {}) if isinstance(dynamic, dict) else {}
            if not isinstance(counts, dict):
                counts = {}

            def _count_prefix(prefix: str) -> int:
                total = 0
                for key, value in counts.items():
                    base = _base_location(str(key))
                    if base == prefix:
                        try:
                            total += int(value)
                        except Exception:
                            total += 1
                return total

            cabinet_checks = _count_prefix("cabinet")
            drawer_checks = _count_prefix("drawer")
            cautions: list[str] = []
            has_non_container_prior = any(
                not any(container in _norm(item) for container in ("cabinet", "drawer")) for item in source_priors
            )
            if cabinet_checks >= 3 and has_non_container_prior:
                cautions.append(
                    f"after {cabinet_checks} cabinet checks without the target, deprioritize continuing cabinet sweep; "
                    "try an unsearched source prior if admissible"
                )
            if drawer_checks >= 3 and has_non_container_prior:
                cautions.append(
                    f"after {drawer_checks} drawer checks without the target, avoid repeating drawer sweep; "
                    "try an unsearched source prior if admissible"
                )
            return cautions[:2]

        def _select_quality_slot_constraints(source_priors: list[str]) -> list[str]:
            """Render reusable slot/role constraints instead of entity-specific rules.

            Quality mode should expose graph memory as typed policy hints:
            target_object, processing_tool, goal_destination, and source_role.
            The concrete values are filled from the current query, but the
            rules themselves remain transferable across ALFWorld tasks.
            """
            if not quality_enabled:
                return []

            constraints: list[str] = []
            progress = str(getattr(query, "progress_state", "") or "")
            target_object = _base_location(str(goal_roles.get("object", "") or ""))
            goal_destination = _base_location(str(goal_roles.get("destination", "") or ""))
            processing_tool = _base_location(str(goal_roles.get("tool", "") or ""))
            held_relevant = getattr(query, "held_relevant_count", 0) > 0
            visible_target = bool(target_object and any(target_object == _base_location(item) for item in visible))

            if target_object:
                constraints.append(
                    "Target-object slot rule: current target_object="
                    f"`{target_object}`. For take/clean/heat/cool/move/put actions, "
                    "manipulate only objects whose base type matches target_object; "
                    "treat any non-target object manipulation as wrong-object evidence, not progress."
                )

            if progress.startswith("search") and not held_relevant:
                pieces = [
                    "Search source-role rule: choose search actions by source role, not memorized scene coordinates"
                ]
                if source_priors:
                    pieces.append(
                        "prefer unsearched local source priors only when they map to current admissible go/open/examine actions"
                    )
                if processing_tool:
                    pieces.append(
                        f"do not treat processing_tool=`{processing_tool}` as a likely starting source unless the target is visible there"
                    )
                if goal_destination:
                    if sourcebase_ranking_enabled:
                        pieces.append(
                            f"treat goal_destination=`{goal_destination}` as a lower-priority source candidate, "
                            "not a delivery target, until target_object is held"
                        )
                    else:
                        pieces.append(
                            f"do not return to goal_destination=`{goal_destination}` before holding target_object"
                        )
                constraints.append("; ".join(pieces) + ".")

            counts = dynamic.get("search_attempt_counts", {}) if isinstance(dynamic, dict) else {}
            if progress.startswith("search") and isinstance(counts, dict) and not visible_target:
                grouped: dict[tuple[str, str], int] = {}
                for key, value in counts.items():
                    base = _base_location(str(key))
                    if not base:
                        continue
                    try:
                        count = int(value)
                    except Exception:
                        count = 1
                    source_role = self._memco_core_location_role(base)
                    grouped[(source_role, base)] = grouped.get((source_role, base), 0) + max(1, count)
                ranked_failed = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
                for (source_role, source_base), count in ranked_failed[:2]:
                    if count < 3:
                        continue
                    constraints.append(
                        "Recent failed source slot: "
                        f"source_role=`{source_role}`, source_base=`{source_base}` has {count} target-missing checks; "
                        "lower priority for continuing the same source_base sweep unless the current observation shows target_object "
                        "or no alternative admissible source prior exists."
                    )

            return _dedupe(constraints, 4)

        def _quality_slot_binding_line() -> str:
            if not quality_enabled:
                return ""
            target_object = _base_location(str(goal_roles.get("object", "") or ""))
            goal_destination = _base_location(str(goal_roles.get("destination", "") or ""))
            processing_tool = _base_location(str(goal_roles.get("tool", "") or ""))
            bindings = []
            if target_object:
                bindings.append(f"target_object={target_object}")
            if processing_tool:
                bindings.append(f"processing_tool={processing_tool}")
            if goal_destination:
                bindings.append(f"goal_destination={goal_destination}")
            progress = str(getattr(query, "progress_state", "") or "").strip()
            if progress:
                bindings.append(f"progress_state={progress}")
            if not bindings:
                return ""
            return (
                "Slot binding for global templates: "
                + "; ".join(bindings)
                + ". Treat support_surface/container as source-role variables unless they are explicitly bound by the current observation."
            )

        def _instantiate_global_template(text: str) -> str:
            """Fill transferable global workflow placeholders with current slots.

            This keeps global memory abstract, but prevents the solver from
            seeing unbound placeholders like target_object/goal_destination as
            vague instructions. Source roles remain roles because unseen scenes
            should not inherit concrete locations.
            """
            if not quality_enabled:
                return text
            rendered = str(text or "")
            replacements = {
                "target_object": _base_location(str(goal_roles.get("object", "") or "")),
                "goal_destination": _base_location(str(goal_roles.get("destination", "") or "")),
            }
            tool = _base_location(str(goal_roles.get("tool", "") or ""))
            if tool:
                replacements["processing_tool"] = tool
            for placeholder, value in replacements.items():
                if not value:
                    continue
                rendered = re.sub(rf"\b{re.escape(placeholder)}\b", f"{placeholder}={value}", rendered)
            if tool:
                rendered = re.sub(r"\btool=container\b", f"tool=processing_tool={tool}", rendered)
                rendered = re.sub(r"\bwith\s+container\b", f"with processing_tool={tool}", rendered)
            return rendered

        def _candidate_kind(item: Any) -> str:
            kind = getattr(item, "candidate_type", "")
            value = getattr(kind, "value", kind)
            return _norm(str(value or ""))

        def _item_confidence(item: Any) -> float:
            support = _quality_support(item)
            if support <= 0:
                return float(getattr(item, "score", 0.0) or 0.0)
            return (float(getattr(item, "positive", 0) or 0) + 0.5) / (support + 1.0)

        def _quality_goal_match(item: Any, text: str) -> float:
            lowered = _norm(text)
            payload = _item_payload(item)
            anchor_payload = payload.get("anchor", {}) if isinstance(payload.get("anchor", {}), dict) else {}
            item_family = _norm(
                " ".join(
                    str(value or "")
                    for value in (
                        getattr(item, "task_family", ""),
                        payload.get("task_family", ""),
                        payload.get("task_type", ""),
                        anchor_payload.get("task_family", ""),
                    )
                )
            )
            score = 0.0
            task_family = _norm(str(getattr(query, "task_family", "") or ""))
            if task_family and (task_family in item_family or task_family in lowered):
                score += 0.36
            for role, weight in (("object", 0.28), ("destination", 0.24), ("tool", 0.16)):
                term = _norm(str(goal_roles.get(role, "") or ""))
                if term and (
                    term in lowered
                    or term in _norm(str(payload.get(role, "") or ""))
                    or term in _norm(str(payload.get(f"goal_{role}", "") or ""))
                    or term in _norm(str(anchor_payload.get("goal_signature", "") or ""))
                ):
                    score += weight
            try:
                required_count = int(getattr(query, "required_count", 0) or 0)
            except Exception:
                required_count = 0
            if required_count > 1 and any(
                marker in lowered for marker in ("two", "another", "second", "repeat acquire")
            ):
                score += 0.16
            score += 0.25 * float(getattr(item, "goal_relevance", 0.0) or 0.0)
            score += 0.18 * float(getattr(item, "task_relevance", 0.0) or 0.0)
            return score

        def _looks_successful(item: Any, text: str) -> bool:
            if not text or _is_failure_like(item, text) or _contains_non_action_think_step(text):
                return False
            branch = _norm(str(getattr(item, "branch_tag", "") or ""))
            kind = _candidate_kind(item)
            positive = int(getattr(item, "positive", 0) or 0)
            negative = int(getattr(item, "negative", 0) or 0)
            if "failure" in branch or "failure" in kind:
                return False
            if positive > 0 and positive >= negative:
                return True
            if "success" in branch and negative <= positive:
                return True
            return any(
                marker in _norm(text)
                for marker in ("workflow pattern", "closure pattern", "successful", "in similar states, try")
            )

        def _looks_failed(item: Any, text: str) -> bool:
            if not text or _contains_non_action_think_step(text):
                return False
            branch = _norm(str(getattr(item, "branch_tag", "") or ""))
            kind = _candidate_kind(item)
            negative = int(getattr(item, "negative", 0) or 0)
            positive = int(getattr(item, "positive", 0) or 0)
            return (
                _is_failure_like(item, text)
                or "failure" in branch
                or "failure" in kind
                or (negative > 0 and negative >= positive)
            )

        def _render_quality_success(item: Any, text: str) -> str:
            lowered = _norm(text)
            if lowered.startswith("in similar states, try "):
                action = text.split("try ", 1)[1].strip().rstrip(".")
                action_norm = _norm(action)
                if action_norm in {_norm(cmd) for cmd in admissible}:
                    return f"same-goal successful continuation used admissible action `{action}`; prefer it when it matches the current observation"
                return f"same-goal successful continuation pattern: {action}; translate it through currently admissible actions"
            if "workflow pattern:" in text.lower() or "closure pattern:" in text.lower() or "->" in text:
                return f"same-goal successful workflow: {text}; use as order only, not exact locations"
            return f"same-goal successful evidence: {text}"

        def _render_quality_failure(item: Any, text: str) -> str:
            lowered = _norm(text)
            if "fails under" in lowered:
                return f"avoid repeating failed local transition: {text}"
            if "wrong target object" in lowered:
                return "avoid wrong-object transfer; only take or move the object type named in the current goal"
            if "search" in lowered or "stall" in lowered:
                return f"avoid repeating same-class search stall: {text}"
            return f"avoid same-goal failed evidence: {text}"

        def _select_quality_matched_routes(items: list[Any], *, want_failure: bool, limit: int) -> list[str]:
            if not quality_enabled:
                return []
            progress = str(getattr(query, "progress_state", "") or "")
            visible_goal_object = False
            goal_object = _norm(str(goal_roles.get("object", "") or ""))
            if goal_object:
                visible_goal_object = any(goal_object in _norm(item) for item in visible)
            held_relevant = getattr(query, "held_relevant_count", 0) > 0
            admissible_norm = {_norm(cmd) for cmd in admissible}

            def _has_exact_admissible_action_hint(text: str) -> bool:
                lowered = _norm(text)
                if lowered.startswith("in similar states, try "):
                    action = text.split("try ", 1)[1].strip().rstrip(".")
                    return _norm(action) in admissible_norm
                return any(_norm(cmd) and _norm(cmd) in lowered for cmd in admissible[:20])

            def _is_generic_route(text: str) -> bool:
                lowered = _norm(text)
                return any(
                    marker in lowered
                    for marker in (
                        "within search",
                        "supports progress",
                        "advances the search",
                        "avoid think",
                        "prefer take(object=target_object",
                        "take(object=target_object,source=support_surface)",
                        "go(target=goal_destination)",
                    )
                )

            ranked: list[tuple[float, str, Any]] = []
            for item in items:
                text = _item_text(item)
                if not text:
                    continue
                if self._memco_core_feedback_item_state(item) == "quarantined":
                    continue
                if want_failure:
                    if not _looks_failed(item, text):
                        continue
                elif not _looks_successful(item, text):
                    continue
                if _is_generic_route(text):
                    continue
                if not want_failure and progress.startswith("search") and not visible_goal_object:
                    # In search states without the target visible, abstract
                    # "successful continuation" rules such as take/go are
                    # usually premature. Source priors remain the mechanism for
                    # choosing where to search next.
                    continue
                if not want_failure and not (
                    _has_exact_admissible_action_hint(text) or held_relevant or visible_goal_object
                ):
                    continue
                if want_failure and ("think:" in text.lower() or "avoid think" in _norm(text)):
                    continue
                match = _quality_goal_match(item, text)
                if want_failure:
                    if match < 0.42:
                        continue
                elif match < 0.5:
                    continue
                if _is_concrete_scene_hint(text) and not sourcehint_enabled:
                    # Quality routes should transfer behavior, not memorize a
                    # scene coordinate. Exact source priors are handled
                    # separately through admissible-action checks above.
                    continue
                support = _quality_support(item)
                confidence = _item_confidence(item)
                if not want_failure and support > 0 and confidence < 0.5:
                    continue
                if want_failure and support > 0 and confidence > 0.7:
                    continue
                base_score = (
                    match
                    + 0.35 * float(getattr(item, "score", 0.0) or 0.0)
                    + 0.08 * min(support, 4)
                    + (0.16 if not want_failure and _norm(text).startswith("in similar states, try ") else 0.0)
                    + (0.12 if want_failure and _looks_failed(item, text) else 0.0)
                )
                rendered = _render_quality_failure(item, text) if want_failure else _render_quality_success(item, text)
                ranked.append((base_score, rendered, item))
            ranked.sort(key=lambda entry: entry[0], reverse=True)
            selected: list[str] = []
            for _score, rendered, item in ranked:
                if rendered in selected:
                    continue
                selected.append(rendered)
                slot = "quality_failure_avoid" if want_failure else "quality_success_route"
                self._memco_core_record_feedback_item(item, slot=slot, rendered_text=rendered)
                if len(selected) >= limit:
                    break
            return selected

        def _format_item(item: Any) -> str:
            text = _item_text(item)
            if not text:
                return ""
            return text

        def _source_is_global(item: Any) -> bool:
            return str(getattr(item, "source", "") or "").startswith("global")

        def _source_is_local(item: Any) -> bool:
            source = str(getattr(item, "source", "") or "")
            return source.startswith("local") or source == "local_graph"

        def _memory_artifact_support_items(
            memory: Any, *, source_label: str, include_scene_relations: bool
        ) -> list[Any]:
            """Expose structured artifacts to graph_policy source routing.

            The retriever keeps the prompt compact, so source-prior artifacts
            can be absent from the routed bundle even when they exist in the
            local/global memory graph. This adapter is graph_policy-only and
            creates SupportItem-like objects for source-type ranking.
            """
            artifacts = getattr(memory, "artifacts_by_id", {}) or {}
            artifact_values = artifacts.values() if isinstance(artifacts, dict) else artifacts
            rendered: list[Any] = []
            for artifact in artifact_values:
                payload = getattr(artifact, "payload", {}) or {}
                anchor = getattr(artifact, "anchor", {}) or {}
                pattern_kind = str(payload.get("pattern_kind", "") or "")
                if pattern_kind == "scene_relation" and not include_scene_relations:
                    continue
                if pattern_kind not in {"source_type_prior", "scene_relation"}:
                    continue
                if pattern_kind == "scene_relation" and str(payload.get("relation_kind", "") or "") not in {
                    "object_location_prior",
                    "searched_empty",
                }:
                    continue
                stats = getattr(artifact, "stats", None)
                support = int(getattr(stats, "support", 0) or 0)
                positive = int(getattr(stats, "success", 0) or 0)
                negative = int(getattr(stats, "failure", 0) or 0)
                confidence = float(getattr(stats, "confidence", 0.0) or 0.0)
                score = float(getattr(artifact, "specificity", 0.0) or 0.0) + confidence + min(support, 5) * 0.04
                rendered.append(
                    SimpleNamespace(
                        summary=str(getattr(artifact, "summary", "") or ""),
                        source=source_label,
                        pattern_kind=pattern_kind,
                        score=score,
                        positive=positive,
                        negative=negative,
                        task_relevance=0.0,
                        goal_relevance=0.0,
                        dynamic={"payload": dict(payload), "anchor": dict(anchor)},
                        branch_tag="artifact",
                        candidate_type=pattern_kind,
                        candidate_id=str(getattr(artifact, "artifact_id", "")),
                    )
                )
            return rendered

        def _select_role_items(
            items: list[Any],
            *,
            limit: int,
            allow_concrete: bool,
            require_task_relevance: bool,
            allow_failure: bool = False,
            skeleton_only: bool = False,
        ) -> tuple[list[str], int]:
            ranked = sorted(
                items, key=lambda item: (_task_relevance(_item_text(item)),) + _rank_key(item), reverse=True
            )
            selected: list[str] = []
            suppressed = 0
            for item in ranked:
                text = _format_item(item)
                if not text:
                    continue
                if self._memco_core_feedback_item_state(item) == "quarantined":
                    suppressed += 1
                    continue
                if not allow_failure and _is_failure_like(item, text):
                    suppressed += 1
                    continue
                if skeleton_only and not _is_usable_global_skeleton(text):
                    suppressed += 1
                    continue
                if _contains_non_action_think_step(text):
                    suppressed += 1
                    continue
                if not allow_concrete and _is_concrete_scene_hint(text):
                    suppressed += 1
                    continue
                if require_task_relevance and _task_relevance(text) <= 0.0:
                    suppressed += 1
                    continue
                if text not in selected:
                    selected.append(text)
                    self._memco_core_record_feedback_item(item, slot="graph_policy", rendered_text=text)
                if len(selected) >= limit:
                    break
            return selected, suppressed

        local_graph_sources = [
            item
            for item in (
                list(getattr(bundle, "fact_items", []) or [])
                + list(getattr(bundle, "relation_items", []) or [])
                + list(getattr(bundle, "local_graph_contribution", []) or [])
            )
            if _source_is_local(item) or str(getattr(item, "source", "") or "") == ""
        ]
        local_domain_sources = [
            item
            for item in (
                list(getattr(bundle, "local_promoted_contribution", []) or [])
                + list(getattr(bundle, "plan_items", []) or [])
                + list(getattr(bundle, "workflow_items", []) or [])
                + list(getattr(bundle, "precondition_items", []) or [])
                + list(getattr(bundle, "closure_items", []) or [])
            )
            if not _source_is_global(item)
        ]
        global_sources = (
            list(getattr(bundle, "global_task_plan_items", []) or [])
            + [item for item in (getattr(bundle, "workflow_items", []) or []) if _source_is_global(item)]
            + [item for item in (getattr(bundle, "precondition_items", []) or []) if _source_is_global(item)]
            + list(getattr(bundle, "global_promoted_contribution", []) or [])
            + list(getattr(bundle, "global_items", []) or [])
        )
        warning_sources = list(getattr(bundle, "repair_items", []) or []) + list(
            getattr(bundle, "reflection_items", []) or []
        )
        memory_source_type_sources = (
            _memory_artifact_support_items(local_memory, source_label="local_artifact", include_scene_relations=True)
            if sourcebase_ranking_enabled and local_memory is not None
            else []
        )
        if sourcebase_ranking_enabled and global_memory is not None:
            memory_source_type_sources.extend(
                _memory_artifact_support_items(
                    global_memory,
                    source_label="global_artifact",
                    include_scene_relations=False,
                )
            )
        quality_source_sources = (
            local_graph_sources
            + local_domain_sources
            + memory_source_type_sources
            + [
                item
                for item in (getattr(bundle, "local_items", []) or [])
                if _source_is_local(item) or str(getattr(item, "source", "") or "") == ""
            ]
        )
        quality_route_sources = (
            local_graph_sources
            + local_domain_sources
            + warning_sources
            + [
                item
                for item in (getattr(bundle, "local_items", []) or [])
                if _source_is_local(item) or str(getattr(item, "source", "") or "") == ""
            ]
        )

        quality_searching = bool(
            quality_enabled
            and getattr(query, "held_relevant_count", 0) <= 0
            and str(getattr(query, "progress_state", "") or "").startswith("search")
        )
        global_limit = 2 if global_transfer_weight >= 0.35 or persistent_skeleton else 1
        local_graph_limit = 3 if local_graph_weight >= 0.55 else 2
        local_domain_limit = 0 if quality_searching else (2 if local_domain_weight >= 0.45 else 1)
        global_skeleton, suppressed_global = _select_role_items(
            global_sources,
            limit=global_limit,
            allow_concrete=False,
            require_task_relevance=False,
            skeleton_only=True,
        )
        local_grounding, suppressed_local_grounding = _select_role_items(
            local_graph_sources,
            limit=local_graph_limit,
            allow_concrete=sourcehint_enabled,
            require_task_relevance=True,
        )
        if local_domain_limit > 0:
            local_domain, suppressed_local_domain = _select_role_items(
                local_domain_sources,
                limit=local_domain_limit,
                allow_concrete=False,
                require_task_relevance=False,
                skeleton_only=False,
            )
        else:
            local_domain, suppressed_local_domain = [], len(local_domain_sources)
        local_warnings, _suppressed_warnings = _select_role_items(
            warning_sources,
            limit=2,
            allow_concrete=True,
            require_task_relevance=False,
            allow_failure=True,
        )
        quality_success_routes = _select_quality_matched_routes(quality_route_sources, want_failure=False, limit=2)
        quality_failure_avoids = _select_quality_matched_routes(quality_route_sources, want_failure=True, limit=2)
        quality_source_priors = _select_quality_source_priors(quality_source_sources, limit=2)

        def _summarize_source_type_priors(candidates: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
            """Split source-type priors into global backoff vs local episode priors.

            The prompt should expose global source-type transfer only as a weak
            backoff, while local exact/source relations stay in the current-state
            section below.
            """
            if not candidates:
                return [], []
            global_by_base: dict[str, dict[str, Any]] = {}
            local_by_base: dict[str, dict[str, Any]] = {}
            for entry in candidates:
                if str(entry.get("pattern_kind", "") or "") != "source_type_prior":
                    continue
                base = _base_location(str(entry.get("base", "") or ""))
                if not base:
                    continue
                try:
                    score = float(entry.get("score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
                bucket = global_by_base if str(entry.get("source_scope", "") or "") == "global" else local_by_base
                previous = bucket.get(base)
                if previous is None or score > float(previous.get("score", 0.0) or 0.0):
                    bucket[base] = dict(entry)

            def _render(entry: dict[str, Any]) -> str:
                base = _display_location(str(entry.get("base", "") or ""))
                support = int(float(entry.get("support", 0.0) or 0.0))
                confidence = float(entry.get("confidence", 0.0) or 0.0)
                scope = str(entry.get("source_scope", "") or "local")
                transfer_level = str(entry.get("transfer_level", "") or "exact")
                suffix = "global backoff" if scope == "global" else "local prior"
                if transfer_level == "role":
                    suffix = f"{suffix}, role-level"
                return f"{base} ({suffix}; support={support}, confidence={confidence:.2f})"

            global_lines = [
                _render(entry)
                for entry in sorted(
                    global_by_base.values(), key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True
                )
            ]
            local_lines = [
                _render(entry)
                for entry in sorted(
                    local_by_base.values(), key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True
                )
            ]
            return global_lines[:3], local_lines[:3]

        quality_search_cautions = _select_quality_search_cautions(quality_source_priors)
        quality_slot_constraints = _select_quality_slot_constraints(quality_source_priors)
        quality_slot_binding = _quality_slot_binding_line()
        global_source_type_priors, local_source_type_priors = _summarize_source_type_priors(
            getattr(self, "_memco_core_quality_source_prior_candidates", []) or []
        )

        if persistent_skeleton:
            global_skeleton = _dedupe(list(persistent_skeleton[:2]) + global_skeleton, global_limit)

        conflict_notes: list[str] = []
        if suppressed_global:
            conflict_notes.append(
                f"Suppressed {suppressed_global} global concrete/location-specific hint(s); global memory is used as task skeleton only."
            )
        if suppressed_local_grounding or suppressed_local_domain:
            if quality_enabled:
                conflict_notes.append(
                    "Kept only quality-filtered local source priors; concrete global locations and already-checked local locations remain suppressed."
                )
            elif sourcehint_enabled:
                conflict_notes.append(
                    "Suppressed local evidence with weak task/current-state relevance; keep target-matching local source evidence when it is not already checked."
                )
            else:
                conflict_notes.append(
                    "Suppressed weak or concrete local scene-location hints; use local evidence only for current-state consistency."
                )
        if global_skeleton and (local_grounding or local_domain):
            if quality_enabled:
                conflict_notes.append(
                    "Use global memory for abstract order, local source priors for search priority, and current observations to override both."
                )
            elif sourcehint_enabled:
                conflict_notes.append(
                    "Use global memory for abstract order, and local graph evidence for target-specific source/location grounding."
                )
            else:
                conflict_notes.append(
                    "Use global memory for abstract order and local graph evidence only for current-state grounding."
                )

        query_parts = [
            f"progress={getattr(query, 'progress_state', '') or 'unknown'}",
            f"stage={getattr(query, 'current_stage', '') or 'unknown'}",
            f"location={getattr(query, 'location', '') or 'unknown'}",
        ]
        for key in ("object", "destination", "tool"):
            value = goal_roles.get(key)
            if value:
                query_parts.append(f"{key}={value}")

        if candidate_mode:
            routing_intro = (
                "Use graph memory as phase-conditioned candidate context: global graph patterns propose transferable candidate roles, "
                "and local graph/current admissible actions ground them into executable next-step options."
            )
        elif quality_enabled:
            routing_intro = (
                "Use local graph grounding for current-state consistency; use global transfer as an abstract task skeleton; "
                "convert target-matching local graph source priors into current-state admissible action priorities."
            )
        elif sourcehint_enabled:
            routing_intro = (
                "Use local graph grounding for current-state consistency and target-specific source hints; "
                "use global transfer as an abstract task skeleton, not as a scene-location instruction."
            )
        else:
            routing_intro = (
                "Use local graph grounding for current-state consistency; use global transfer as an abstract task skeleton, "
                "not as a scene-location instruction."
            )

        lines = [
            "Graph policy memory routing.",
            routing_intro,
            (
                "Routing weights: "
                f"local_graph={local_graph_weight:.2f}, "
                f"local_domain={local_domain_weight:.2f}, "
                f"global_transfer={global_transfer_weight:.2f}, "
                f"persistent_global={persistent_global_weight:.2f}."
            ),
            "Current graph query: " + "; ".join(query_parts) + ".",
        ]
        if persistent_skeleton:
            lines.append(
                "Persistent global task plan (episode-level; keep across steps unless the current observation contradicts it):"
            )
            lines.extend(f"- {item}" for item in persistent_skeleton[:2])
            if global_transfer_weight <= 0.0:
                lines.append(
                    "Routing note: step-level global transfer is weak or absent, but the episode-level global task plan remains active."
                )
        if self._memco_core_is_two_object_second_search(query):
            lines.append(
                "Two-object stage: one target has already been delivered. Continue searching for another target instance; do not return to the destination until holding it."
            )
        if visible or held or exhausted:
            context_parts: list[str] = []
            if held:
                context_parts.append("held=" + ", ".join(held[:3]))
            if visible:
                context_parts.append("visible=" + ", ".join(visible[:5]))
            if exhausted:
                context_parts.append("already_checked=" + ", ".join(exhausted[:5]))
            lines.append("Current graph state: " + "; ".join(context_parts) + ".")
        if quality_slot_constraints:
            lines.append("Slot-level graph constraints (role/slot policy; reusable across object types):")
            lines.extend(f"- {item}" for item in quality_slot_constraints)
        if quality_slot_binding:
            lines.append(quality_slot_binding)
        if global_skeleton:
            lines.append("Global task skeleton (abstract transfer only; no scene-specific locations):")
            lines.extend(f"- {_instantiate_global_template(item)}" for item in global_skeleton)
        if local_grounding:
            lines.append("Local current-state grounding (trust only when aligned with observation/admissible actions):")
            lines.extend(f"- {item}" for item in local_grounding)
        if quality_success_routes:
            lines.append(
                "Quality-matched successful continuations (same goal pattern; higher priority than weak source priors):"
            )
            lines.extend(f"- {item}" for item in quality_success_routes)
        if quality_failure_avoids:
            lines.append("Quality-matched failed routes to avoid:")
            lines.extend(f"- {item}" for item in quality_failure_avoids)
        if local_domain:
            lines.append("Local domain guidance:")
            lines.extend(f"- {item}" for item in local_domain)
        if global_source_type_priors:
            if candidate_mode:
                lines.append("Global phase-conditioned candidate prior (transfer source roles; no exact locations):")
            else:
                lines.append("Global source-type transfer prior (weak backoff; no exact locations):")
            lines.extend(f"- {item}" for item in global_source_type_priors)
        if local_source_type_priors:
            if candidate_mode:
                lines.append("Local phase-conditioned candidate prior (episode-local grounding):")
            else:
                lines.append("Local source-type prior (episode-local):")
            lines.extend(f"- {item}" for item in local_source_type_priors)
        if quality_source_priors:
            if candidate_mode:
                lines.append("Current executable candidate priority (global/local graph mapped to admissible actions):")
            else:
                lines.append("Current-state graph priority (admissible source actions from local graph):")
            lines.extend(f"- {item}" for item in quality_source_priors)
        if quality_search_cautions:
            lines.append("Search caution (from local graph failure/search statistics):")
            lines.extend(f"- {item}" for item in quality_search_cautions)
        if local_warnings:
            lines.append("Local cautions:")
            lines.extend(f"- {item}" for item in local_warnings)
        if conflict_notes:
            lines.append("Memory arbitration:")
            lines.extend(f"- {item}" for item in _dedupe(conflict_notes, 3))
        elif len(lines) <= 4:
            return ""
        return "\n".join(lines)

    def _render_action_grounded_memory_evidence(self, *, query: Any, bundle: Any, env_ref: Any) -> str:
        """Render a small, action-grounded MemCo view for repair-only action mode.

        This keeps the memory bank large but exposes only a few pieces that are
        tied to the current admissible action set. Global memory is treated as a
        tie-breaker; prototype/template-like evidence is filtered from prompt
        text because it has been noisy in nvdamas AutoGen prompts.
        """
        admissible = [
            str(cmd).strip() for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()
        ]
        if not admissible:
            return ""

        import re

        def _clean(summary: str) -> str:
            cleaned = str(summary or "").strip()
            for prefix in (
                "Scene relation: ",
                "Plan: ",
                "Workflow pattern: ",
                "Workflow: ",
                "Reflection: ",
                "Closure: ",
                "Local state fact: ",
                "Prototype: ",
                "Precondition: ",
                "Blocked: ",
            ):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
            return cleaned

        def _norm(value: str) -> str:
            return re.sub(r"\s+", " ", str(value or "").replace("_", " ").replace("-", " ").strip().lower())

        admissible_norm = [_norm(cmd) for cmd in admissible]
        admissible_verbs = {cmd.split(" ", 1)[0] for cmd in admissible_norm if cmd}
        goal_roles = getattr(query, "goal_roles", {}) or {}
        goal_terms = {
            _norm(value)
            for value in (
                goal_roles.get("object"),
                goal_roles.get("destination"),
                goal_roles.get("tool"),
                getattr(query, "location", ""),
            )
            if str(value or "").strip()
        }
        progress_state = str(getattr(query, "progress_state", "") or "").strip()

        def _is_prototype(item: Any, text: str) -> bool:
            dynamic = getattr(item, "dynamic", {}) or {}
            return str(dynamic.get("artifact_kind", "")).lower() == "prototype" or str(text or "").lower().startswith(
                "prototype:"
            )

        def _is_template_like(text: str) -> bool:
            lowered = _norm(text)
            return any(
                marker in lowered
                for marker in (
                    "target object",
                    "goal destination",
                    "support surface",
                    "container=container",
                    "object=target object",
                    "destination=goal destination",
                    "think:",
                )
            )

        def _is_generic_search_open(text: str) -> bool:
            lowered = _norm(text)
            return any(
                marker in lowered
                for marker in (
                    "within search open advances",
                    "within search open supports",
                    "within carry target open advances",
                    "within carry target open supports",
                )
            )

        def _action_overlap(item: Any, text: str) -> float:
            lowered = _norm(text)
            if any(action and action in lowered for action in admissible_norm):
                return 3.0
            verbs = set()
            for pattern in getattr(item, "action_patterns", ()) or ():
                match = re.match(r"\s*([a-z_]+)\s*(?:\(|$)", str(pattern).strip().lower())
                if match:
                    verbs.add(match.group(1).replace("_", " "))
            for verb in ("open", "take", "move", "go", "use", "examine", "clean", "cool", "heat"):
                if re.search(rf"\b{verb}\b", lowered):
                    verbs.add(verb)
            return 1.0 if verbs & admissible_verbs else 0.0

        def _goal_overlap(text: str) -> float:
            lowered = _norm(text)
            return float(sum(1 for term in goal_terms if term and term in lowered))

        def _score_item(item: Any, route: str) -> tuple[float, str] | None:
            text = _clean(getattr(item, "summary", ""))
            if not text:
                return None
            if _is_generic_search_open(text):
                return None
            prototype = _is_prototype(item, text)
            action_score = _action_overlap(item, text)
            goal_score = _goal_overlap(text)
            template_like = _is_template_like(text)

            if route == "global" and (prototype or action_score <= 0.0):
                return None
            if template_like and goal_score <= 0.0 and route != "local_graph":
                return None

            score = float(getattr(item, "score", 0.0) or 0.0)
            score += 2.5 * action_score
            score += 1.4 * goal_score
            score += 0.7 * float(getattr(item, "goal_relevance", 0.0) or 0.0)
            score += 0.3 * float(getattr(item, "state_relevance", 0.0) or 0.0)
            if route == "local_graph":
                score += 0.8
            elif route == "local":
                score += 0.35
            elif route == "global":
                score -= 0.35
            if prototype:
                score -= 1.8
            if template_like:
                score -= 0.7
            if str(progress_state).startswith("search") and "holding target" in _norm(text):
                score -= 3.0
            return score, text

        def _rank(route: str, items: list[Any], *, limit: int, floor: float) -> list[str]:
            scored: list[tuple[float, str]] = []
            for item in items:
                packed = _score_item(item, route)
                if packed is None:
                    continue
                scored.append(packed)
            scored.sort(key=lambda pair: pair[0], reverse=True)
            picked: list[str] = []
            for score, text in scored:
                if score < floor:
                    continue
                if text not in picked:
                    picked.append(text)
                if len(picked) >= limit:
                    break
            return picked

        grounding = _rank(
            "local_graph",
            list(getattr(bundle, "local_graph_contribution", []) or [])
            + list(getattr(bundle, "fact_items", []) or [])
            + list(getattr(bundle, "relation_items", []) or []),
            limit=2,
            floor=0.8,
        )
        local_guidance = _rank(
            "local",
            list(getattr(bundle, "local_promoted_contribution", []) or []),
            limit=1,
            floor=1.2,
        )
        global_tiebreak = _rank(
            "global",
            list(getattr(bundle, "global_promoted_contribution", []) or []),
            limit=1,
            floor=2.0,
        )

        if not grounding and not local_guidance and not global_tiebreak:
            return ""

        lines = [
            "Memory evidence is supplementary context only.",
            "Use it only when it matches the current observation and admissible actions.",
            "Do not treat memory as a direct action recommendation.",
        ]
        if grounding:
            lines.append("Local state-grounded evidence:")
            lines.extend(f"- {item}" for item in grounding)
        if local_guidance:
            lines.append("Local domain guidance:")
            lines.extend(f"- {item}" for item in local_guidance)
        if global_tiebreak:
            lines.append("Global transferable tie-breaker:")
            lines.extend(f"- {item}" for item in global_tiebreak)
        return "\n".join(lines)

    @staticmethod
    def _normalize_action_text(value: str) -> str:
        import re

        return re.sub(r"\s+", " ", str(value or "").strip().rstrip(".。").lower())

    @staticmethod
    def _is_concrete_alfworld_action(value: str) -> bool:
        import re

        text = re.sub(r"\s+", " ", str(value or "").strip().rstrip(".。").lower())
        if not text or text.startswith("think:"):
            return False
        return bool(
            re.match(
                r"^(take|go to|open|put|move|clean|heat|cool|use|examine|look|close)\b",
                text,
            )
        )

    def _memco_core_quality_action_override(
        self, *, processed_action: str, admissible_actions: list[str]
    ) -> str | None:
        """Use quality-mode graph source priors only as a narrow search repair.

        This never runs for the stable graph_policy modes. It only redirects a
        broad cabinet/drawer sweep toward a concrete graph-supported admissible
        search action that was rendered for the current step.
        """
        if self._external_retrieval_mode != "graph_policy_quality_experimental":
            return None
        candidates = [
            str(action).strip()
            for action in (self._memco_core_quality_last_action_candidates or [])
            if str(action).strip() in set(admissible_actions)
        ]
        if not candidates:
            return None

        def _action_target(action: str) -> tuple[str, str]:
            lowered = self._normalize_action_text(action)
            for prefix in ("go to ", "open ", "examine "):
                if lowered.startswith(prefix):
                    target = lowered[len(prefix) :].strip()
                    return prefix.strip(), self._memco_core_base_token(target)
            return "", ""

        verb, base = _action_target(processed_action)
        if not verb or base not in {"cabinet", "drawer"}:
            return None
        for candidate in candidates:
            candidate_verb, candidate_base = _action_target(candidate)
            if candidate_verb and candidate_base and candidate_base not in {"cabinet", "drawer"}:
                return candidate
        return None

    def _soft_phasee_policy_lines(
        self,
        *,
        query: Any,
        rules_text: str,
        suggested: list[str],
        warnings: list[str],
        include_suggestions: bool = True,
    ) -> list[str]:
        lines: list[str] = []
        for line in str(rules_text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if (
                lowered.startswith("state-derived hard rules")
                or lowered.startswith("state-blocked actions")
                or lowered.startswith("task-specific processing rules")
                or lowered.startswith("- do not")
                or lowered.startswith("- if target is visible")
                or lowered.startswith("- only choose")
                or lowered.startswith("- while holding")
            ):
                continue
            if stripped.startswith("- "):
                continue
            if lowered.startswith(
                (
                    "target object:",
                    "destination:",
                    "goal tool:",
                    "current phase:",
                    "held relevant count:",
                    "placed relevant count:",
                    "task requirement:",
                    "processing tool:",
                )
            ):
                lines.append(stripped)
            if len(lines) >= 8:
                break
        if include_suggestions and suggested:
            lines.append("Possible next admissible actions to consider: " + " | ".join(suggested[:3]))
        soft_warnings = [
            item
            for item in warnings
            if "do_not_prioritize" not in item.lower() and "avoid_repeating" not in item.lower()
        ]
        if soft_warnings:
            lines.append("State hints: " + " | ".join(soft_warnings[:3]))
        progress = str(getattr(query, "progress_state", "") or "").strip()
        if progress:
            lines.append(f"Estimated progress state: {progress}")
        return lines[:10]

    def _phasee_policy_full_lines(
        self,
        *,
        rules_text: str,
        fused: dict[str, object],
        policy: Any,
        admissible: list[str],
    ) -> list[str]:
        lines: list[str] = []
        evidence = [str(item).strip() for item in (fused.get("evidence", []) or []) if str(item).strip()]
        failure_reflections = [
            str(item).strip() for item in (fused.get("failure_reflections", []) or []) if str(item).strip()
        ]
        state_facts = [str(item).strip() for item in (fused.get("state_facts", []) or []) if str(item).strip()]
        suggested = [str(item).strip() for item in (fused.get("suggested_actions", []) or []) if str(item).strip()]
        blocked = [str(item).strip() for item in (fused.get("blocked_actions", []) or []) if str(item).strip()]
        warnings = [str(item).strip() for item in (fused.get("warnings", []) or []) if str(item).strip()]

        if evidence:
            lines.append("Routed graph evidence:")
            lines.extend(f"- {item}" for item in evidence[:5])
        if failure_reflections:
            lines.append("Failure/caution evidence:")
            lines.extend(f"- {item}" for item in failure_reflections[:3])
        if state_facts:
            lines.append("State facts:")
            lines.extend(f"- {item}" for item in state_facts[:5])
        if rules_text:
            lines.append("State policy:")
            lines.extend(f"- {line.strip()}" for line in str(rules_text).splitlines() if line.strip()[:200])
        if suggested:
            lines.append("PhaseE suggested admissible actions:")
            lines.extend(f"- {item}" for item in suggested[:5])
        if blocked:
            lines.append("PhaseE risky/blocked actions:")
            lines.extend(f"- {item}" for item in blocked[:5])
        if warnings:
            lines.append("PhaseE warnings:")
            lines.extend(f"- {item}" for item in warnings[:5])

        score_map = getattr(policy, "action_scores", {}) or {}
        scored = []
        for action in admissible:
            try:
                scored.append((action, float(score_map.get(action, 0.0))))
            except Exception:
                scored.append((action, 0.0))
        scored.sort(key=lambda item: item[1], reverse=True)
        top_scored = [f"{action} ({score:.1f})" for action, score in scored[:5] if score != 0.0]
        if top_scored:
            lines.append("PhaseE action-score priors, for tie-breaking only:")
            lines.append("- " + " | ".join(top_scored))
        return lines[:36]

    def _record_phasee_quality_feedback_items(self, bundle: Any) -> None:
        """Record the evidence exposed by graph_policy_quality for feedback.

        This keeps quality routing close to the original PhaseE idea: the
        memory items that influence the policy prompt are the ones that receive
        later success/failure feedback. It does not alter the solver loop.
        """
        if not self._memco_core_feedback_enabled():
            return
        groups = (
            ("quality_policy_local_graph", ("fact_items", "relation_items", "local_graph_contribution"), 3),
            ("quality_policy_local_domain", ("plan_items", "workflow_items", "precondition_items", "closure_items"), 3),
            ("quality_policy_global", ("global_task_plan_items", "global_promoted_contribution", "global_items"), 2),
            ("quality_policy_failure", ("repair_items", "reflection_items"), 2),
        )
        seen: set[str] = set()
        for slot, attrs, limit in groups:
            count = 0
            for attr in attrs:
                for item in list(getattr(bundle, attr, []) or []):
                    text = str(getattr(item, "summary", "") or "").strip()
                    key = text.lower()
                    if not text or key in seen:
                        continue
                    if self._memco_core_feedback_item_state(item) == "quarantined":
                        continue
                    seen.add(key)
                    self._memco_core_record_feedback_item(item, slot=slot, rendered_text=text)
                    count += 1
                    if count >= limit:
                        break
                if count >= limit:
                    break

    def set_global_retriever(self, global_retriever: Any) -> None:
        """Attach a shared MemCo global memory while keeping this instance's local memory."""
        external_global = getattr(global_retriever, "_external_global_memory", None)
        if external_global is not None:
            self._external_global_memory = external_global
            self._external_enabled = True

    def _infer_external_domain(self, task_config: dict | None = None, env_ref: Any = None) -> str:
        task_config = task_config or {}
        candidates = [
            str(getattr(env_ref, "memco_domain", "") or ""),
            str(task_config.get("memco_domain", "") or ""),
            str(task_config.get("env_name", "") or ""),
            str(self.global_config.get("task_name", "") or ""),
            str(task_config.get("task_name", "") or ""),
        ]
        if getattr(env_ref, "game_name", None) or task_config.get("game_name"):
            candidates.append("pddl")
        for value in candidates:
            token = value.strip().lower()
            if not token:
                continue
            if token.startswith("pddl"):
                return "pddl"
            if token.startswith("fever"):
                return "fever"
            if token.startswith("alfworld"):
                return "alfworld"
        return "alfworld"

    def _resolve_external_adapter(self, task_config: dict | None = None, env_ref: Any = None) -> Any:
        adapters = self._external_adapters or {}
        domain = self._infer_external_domain(task_config, env_ref)
        return adapters.get(domain) or self._external_adapter

    def _build_external_query(self, **kargs):
        env_ref = kargs.get("env_ref")
        task_config = kargs.get("task_config") or {}
        if env_ref is None:
            self._external_error = "missing env_ref for external MemCo retrieval"
            return None
        adapter = self._resolve_external_adapter(task_config, env_ref)
        if adapter is not None and hasattr(adapter, "build_query"):
            try:
                return adapter.build_query(
                    env_ref=env_ref,
                    task_config=task_config,
                    query_task=str(kargs.get("query_task") or ""),
                    MemoryQuery=self._external_memory_query_type,
                    CandidateType=self._external_candidate_type,
                )
            except Exception as exc:
                self._external_error = f"external GraphMemory query adapter failed: {type(exc).__name__}: {exc}"
                return None
        goal = str(getattr(env_ref, "goal_instruction", "") or "").strip() or str(kargs.get("query_task") or "").strip()
        if not goal:
            self._external_error = "missing goal for external MemCo retrieval"
            return None

        gamefile = (
            str(getattr(env_ref, "resolved_gamefile", "") or "").strip()
            or str(getattr(env_ref, "gamefile", "") or "").strip()
        )
        scene_id = adapter.derive_scene_id(gamefile) if gamefile else "alfworld:unknown"
        history = list(getattr(env_ref, "current_history", []) or [])
        current = history[-1] if history else {}
        observation = str(current.get("Observation") or getattr(env_ref, "initial_observation", "") or "")
        admissible = list(getattr(env_ref, "last_admissible_commands", []) or [])
        recent_actions = list(getattr(self.episode_builder.state, "recent_actions", []) if self.episode_builder else [])
        last_action = adapter.canonicalize_action(recent_actions[-1]) if recent_actions else None
        belief = self._replay_external_episode_belief(
            adapter=adapter,
            history=history,
            goal=goal,
            scene_id=scene_id,
        )
        state = adapter.build_state(
            scene_id=scene_id,
            observation=observation,
            admissible_commands=admissible,
            held_objects=set(belief.get("held_objects", [])),
            searched_locations=set(belief.get("exhausted_locations", [])),
            goal=goal,
            last_action=last_action,
        )
        if state.location is None and belief.get("current_location"):
            state.location = str(belief["current_location"])
        progress = adapter.summarize_goal_progress(
            state,
            goal,
            belief={
                "placed_relevant_count_est": int(belief.get("placed_relevant_count_est", 0) or 0),
                "delivered_instances": list(belief.get("delivered_instances", []) or []),
            },
        )
        CandidateType = self._external_candidate_type
        desired_types = [CandidateType.PRECONDITION, CandidateType.WORKFLOW]
        if int(progress["remaining_relevant_count"]) <= 0 or str(progress["progress_state"]) in {
            "carry_target",
            "finalize",
        }:
            desired_types.append(CandidateType.FAILURE)
        progress_hint = ""
        if int(progress["required_count"]) > 1 and int(progress["remaining_relevant_count"]) > 0:
            if int(progress["held_relevant_count"]) > 0 or state.workflow_stage == "acquire_target":
                progress_hint = "multi_object_mid"
        MemoryQuery = self._external_memory_query_type
        return MemoryQuery(
            goal=goal,
            scene_id=scene_id,
            current_stage=state.workflow_stage,
            location=state.location,
            progress_hint=progress_hint,
            progress_state=str(progress["progress_state"]),
            task_family=adapter.infer_task_family(goal),
            goal_roles=dict(progress["goal_roles"]),
            required_count=int(progress["required_count"]),
            held_relevant_count=int(progress["held_relevant_count"]),
            placed_relevant_count=int(progress["placed_relevant_count"]),
            remaining_relevant_count=int(progress["remaining_relevant_count"]),
            destination_reached=bool(progress["destination_reached"]),
            goal_object_matches_visible=bool(progress["goal_object_matches_visible"]),
            admissible_actions=tuple(adapter.canonicalize_action(cmd) for cmd in admissible),
            desired_types=tuple(desired_types),
            failure_label=str(belief.get("last_failure") or "") or None,
            keywords=(),
            belief={
                "held_objects": list(belief.get("held_objects", []) or []),
                "searched_locations": list(belief.get("searched_locations", []) or []),
                "exhausted_locations": list(belief.get("exhausted_locations", []) or []),
                "search_attempt_counts": dict(belief.get("search_attempt_counts", {}) or {}),
                "delivered_instances": list(belief.get("delivered_instances", []) or []),
                "placed_relevant_count_est": int(belief.get("placed_relevant_count_est", 0) or 0),
                "last_failure": str(belief.get("last_failure") or ""),
            },
            dynamic_context={
                "visible_objects": list(state.visible_objects),
                "held_objects": list(state.held_objects),
                "searched_locations": list(state.searched_locations),
                "visited_locations": list(belief.get("visited_locations", []) or []),
                "inspected_locations": list(belief.get("inspected_locations", []) or []),
                "exhausted_locations": list(belief.get("exhausted_locations", []) or []),
                "search_attempt_counts": dict(belief.get("search_attempt_counts", {}) or {}),
                "delivered_instances": list(belief.get("delivered_instances", []) or []),
                "last_failure": str(belief.get("last_failure") or ""),
                "layout_id": adapter.derive_layout_id(gamefile) if gamefile else "",
                "task_config_env_name": str(task_config.get("env_name", "")),
            },
        )

    def _replay_external_episode_belief(
        self,
        *,
        adapter: Any,
        history: list[Any],
        goal: str,
        scene_id: str,
    ) -> dict[str, Any]:
        """Infer lightweight online belief from the current ALFWorld history.

        This mirrors the vendored adapter's history parser enough for retrieval
        and prompt-only PhaseE state policy. It does not change the environment
        action loop.
        """
        held_objects: set[str] = set()
        searched_locations: set[str] = set()
        exhausted_locations: set[str] = set()
        visited_locations: set[str] = set()
        inspected_locations: set[str] = set()
        delivered_instances: set[str] = set()
        search_attempt_counts: dict[str, int] = {}
        current_location: str | None = None
        last_failure: str | None = None

        if not history:
            return {
                "held_objects": [],
                "searched_locations": [],
                "exhausted_locations": [],
                "visited_locations": [],
                "inspected_locations": [],
                "search_attempt_counts": {},
                "delivered_instances": [],
                "placed_relevant_count_est": 0,
                "current_location": None,
                "last_failure": None,
            }

        initial = history[0] if isinstance(history[0], dict) else {}
        previous_admissible = list(initial.get("Admissible Commands", []) or [])
        prev_state = adapter.build_state(
            scene_id=scene_id,
            observation=str(initial.get("Observation", "") or ""),
            admissible_commands=previous_admissible,
            held_objects=held_objects,
            searched_locations=exhausted_locations,
            goal=goal,
        )
        current_location = prev_state.location
        if current_location:
            visited_locations.add(str(current_location).lower())

        goal_slots = adapter.goal_slots(goal)
        target_obj = str(goal_slots.get("object", "") or "")
        goal_destination = str(goal_slots.get("destination", "") or "")
        target_base = self._memco_core_base_token(target_obj)
        destination_base = self._memco_core_base_token(goal_destination)

        for record in history[1:]:
            if not isinstance(record, dict):
                continue
            action_text = str(record.get("Action", "") or "").strip()
            if not action_text:
                continue
            action = adapter.canonicalize_action(action_text)
            observation = str(record.get("Observation", "") or "")
            failure_label = adapter.detect_failure(observation, action, previous_admissible)
            success = failure_label is None
            last_failure = failure_label

            if success and action.verb == "take" and "pick up" in observation.lower():
                obj = str(action.slots.get("object", "unknown") or "unknown")
                src = str(action.slots.get("source", "") or "")
                held_objects.add(obj)
                if (
                    obj
                    and target_base
                    and self._memco_core_base_token(obj) == target_base
                    and destination_base
                    and self._memco_core_base_token(src) == destination_base
                ):
                    delivered_instances.discard(obj)

            if success and action.verb == "move" and "move" in observation.lower():
                obj = str(action.slots.get("object", "unknown") or "unknown")
                dst = str(action.slots.get("destination", "") or "")
                held_objects.discard(obj)
                if (
                    obj
                    and target_base
                    and self._memco_core_base_token(obj) == target_base
                    and destination_base
                    and self._memco_core_base_token(dst) == destination_base
                ):
                    delivered_instances.add(obj)

            next_admissible = list(record.get("Admissible Commands", []) or [])
            next_state = adapter.build_state(
                scene_id=scene_id,
                observation=observation,
                admissible_commands=next_admissible,
                held_objects=held_objects,
                searched_locations=exhausted_locations,
                goal=goal,
                last_action=action,
            )
            if next_state.location is None:
                next_state.location = prev_state.location
            if next_state.location:
                current_location = str(next_state.location)
                visited_locations.add(current_location.lower())

            if success:
                visible_target = False
                if target_base:
                    visible_target = any(
                        self._memco_core_base_token(str(item)) == target_base
                        for item in (next_state.visible_objects or ())
                    )
                search_ref = adapter.search_reference_for_action(action, prev_state)
                if (
                    action.verb == "go"
                    and next_state.location
                    and self._memco_core_location_role(next_state.location) != "container"
                ):
                    search_ref = next_state.location
                if action.verb in {"open", "examine", "look", "go"} and search_ref:
                    inspected_locations.add(str(search_ref).lower())
                if search_ref and not visible_target:
                    normalized_ref = str(search_ref).lower()
                    searched_locations.add(normalized_ref)
                    exhausted_locations.add(normalized_ref)
                    search_attempt_counts[normalized_ref] = search_attempt_counts.get(normalized_ref, 0) + 1

            prev_state = next_state
            previous_admissible = next_admissible

        return {
            "held_objects": sorted(held_objects),
            "searched_locations": sorted(searched_locations),
            "exhausted_locations": sorted(exhausted_locations),
            "visited_locations": sorted(visited_locations),
            "inspected_locations": sorted(inspected_locations),
            "search_attempt_counts": dict(sorted(search_attempt_counts.items())),
            "delivered_instances": sorted(delivered_instances),
            "placed_relevant_count_est": len(delivered_instances),
            "current_location": current_location,
            "last_failure": last_failure,
        }

    @staticmethod
    def _memco_core_base_token(value: str) -> str:
        import re

        token = re.sub(r"\s+", "_", str(value or "").strip().lower())
        for prefix in ("a_", "an_", "the_", "some_"):
            if token.startswith(prefix):
                token = token[len(prefix) :]
        return re.sub(r"_[0-9]+$", "", token)

    @classmethod
    def _memco_core_location_role(cls, value: str) -> str:
        return (
            "container"
            if cls._memco_core_base_token(value) in {"drawer", "cabinet", "safe", "fridge", "microwave"}
            else "support_surface"
        )

    def _resolve_external_owner_scene(self, task_config: dict | None = None, env_ref: Any = None) -> str:
        configured = str(self._graph_config_value("owner_scene", "") or "").strip()
        if configured:
            return configured
        task_config = task_config or {}
        configured = str(task_config.get("owner_scene", "") or "").strip()
        if configured:
            return configured
        configured = str(getattr(env_ref, "env_name", "") or "").strip()
        if configured:
            return configured
        configured = str(task_config.get("env_name", "") or "").strip()
        if configured:
            return configured
        if len(self._external_local_memories) == 1:
            return next(iter(self._external_local_memories))
        if "kitchen" in self._external_local_memories:
            return "kitchen"
        if self._external_local_memories:
            return next(iter(sorted(self._external_local_memories)))
        return "default"

    def _memco_core_feedback_enabled(self) -> bool:
        return bool(self._external_retrieval_mode in {"graph_policy_feedback", "graph_policy_quality"})

    def _load_memco_core_feedback_stats(self) -> dict[str, Any]:
        path = self._memco_core_feedback_path or (Path(self.persist_dir) / "feedback_stats.json")
        if not path.exists():
            return {"version": 1, "items": {}}
        try:
            with path.open("r", encoding="utf-8") as reader:
                data = json.load(reader)
        except Exception:
            return {"version": 1, "items": {}}
        if not isinstance(data, dict):
            return {"version": 1, "items": {}}
        if not isinstance(data.get("items"), dict):
            data["items"] = {}
        data.setdefault("version", 1)
        return data

    def _save_memco_core_feedback_stats(self) -> None:
        if self._memco_core_feedback_path is None:
            self._memco_core_feedback_path = Path(self.persist_dir) / "feedback_stats.json"
        self._memco_core_feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._memco_core_feedback_path.open("w", encoding="utf-8") as writer:
            json.dump(
                self._memco_core_feedback_stats or {"version": 1, "items": {}}, writer, ensure_ascii=False, indent=2
            )

    @staticmethod
    def _memco_core_item_key(item: Any, rendered_text: str = "") -> str:
        source = str(getattr(item, "source", "") or "unknown")
        candidate_id = str(getattr(item, "candidate_id", "") or "")
        ctype = str(getattr(item, "candidate_type", "") or "")
        summary = str(rendered_text or getattr(item, "summary", "") or "")
        digest = hashlib.sha1(summary.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{source}|{ctype}|{candidate_id or digest}|{digest}"

    def _memco_core_feedback_item_state(self, item: Any) -> str:
        if not self._memco_core_feedback_enabled():
            return "active"
        key = self._memco_core_item_key(item)
        stats = ((self._memco_core_feedback_stats or {}).get("items") or {}).get(key, {})
        state = str(stats.get("state", "") or "active")
        return state if state in {"active", "demoted", "quarantined"} else "active"

    def _memco_core_feedback_adjustment(self, item: Any) -> float:
        if not self._memco_core_feedback_enabled():
            return 0.0
        key = self._memco_core_item_key(item)
        stats = ((self._memco_core_feedback_stats or {}).get("items") or {}).get(key, {})
        if not stats:
            return 0.0
        state = str(stats.get("state", "") or "active")
        utility = float(stats.get("utility", 0.0) or 0.0)
        if state == "quarantined":
            return -100.0
        if state == "demoted":
            return min(utility, 0.0) - 0.75
        return max(min(utility, 0.45), -0.45)

    def _memco_core_record_feedback_item(self, item: Any, *, slot: str, rendered_text: str) -> None:
        if not self._memco_core_feedback_enabled():
            return
        key = self._memco_core_item_key(item, rendered_text)
        if key in self._memco_core_episode_feedback_items:
            return
        self._memco_core_episode_feedback_items[key] = {
            "source": str(getattr(item, "source", "") or ""),
            "candidate_id": str(getattr(item, "candidate_id", "") or ""),
            "candidate_type": str(getattr(item, "candidate_type", "") or ""),
            "pattern_kind": str(getattr(item, "pattern_kind", "") or ""),
            "slot": str(slot or ""),
            "summary": str(rendered_text or getattr(item, "summary", "") or "")[:500],
        }

    def _update_memco_core_feedback_from_episode(self, *, label: bool, **kargs) -> None:
        if not self._memco_core_feedback_enabled() or self.freeze_memory:
            self._memco_core_episode_feedback_items = {}
            return
        if not self._memco_core_episode_feedback_items:
            return
        stats_root = self._memco_core_feedback_stats or {"version": 1, "items": {}}
        items = stats_root.setdefault("items", {})
        env_ref = kargs.get("env_ref")
        max_trials = int(getattr(env_ref, "max_trials", 0) or 0) if env_ref is not None else 0
        step_count = int(getattr(self.episode_builder.state, "step_count", 0) or 0) if self.episode_builder else 0
        stalled = bool((not label) and max_trials and step_count >= max_trials)
        for key, meta in self._memco_core_episode_feedback_items.items():
            item_stats = items.setdefault(
                key,
                {
                    "source": meta.get("source", ""),
                    "candidate_id": meta.get("candidate_id", ""),
                    "candidate_type": meta.get("candidate_type", ""),
                    "pattern_kind": meta.get("pattern_kind", ""),
                    "slot": meta.get("slot", ""),
                    "summary": meta.get("summary", ""),
                    "use_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "stalled_count": 0,
                    "utility": 0.0,
                    "state": "active",
                },
            )
            item_stats["use_count"] = int(item_stats.get("use_count", 0) or 0) + 1
            if label:
                item_stats["success_count"] = int(item_stats.get("success_count", 0) or 0) + 1
            else:
                item_stats["failure_count"] = int(item_stats.get("failure_count", 0) or 0) + 1
            if stalled:
                item_stats["stalled_count"] = int(item_stats.get("stalled_count", 0) or 0) + 1
            use_count = max(int(item_stats.get("use_count", 0) or 0), 1)
            success_count = int(item_stats.get("success_count", 0) or 0)
            failure_count = int(item_stats.get("failure_count", 0) or 0)
            stalled_count = int(item_stats.get("stalled_count", 0) or 0)
            success_rate = success_count / use_count
            failure_rate = failure_count / use_count
            stalled_rate = stalled_count / use_count
            utility = success_rate - 0.75 * failure_rate - 0.35 * stalled_rate
            item_stats["utility"] = round(utility, 4)
            item_stats["last_label"] = bool(label)
            item_stats["last_step_count"] = step_count
            if use_count >= 5 and (failure_rate >= 0.75 or stalled_rate >= 0.6):
                item_stats["state"] = "quarantined"
            elif use_count >= 3 and (failure_rate >= 0.5 or stalled_rate >= 0.4):
                item_stats["state"] = "demoted"
            else:
                item_stats["state"] = "active"
        self._memco_core_feedback_stats = stats_root
        self._save_memco_core_feedback_stats()
        self._memco_core_episode_feedback_items = {}

    def summarize(self, **kargs) -> str:
        base = super().summarize(**kargs)
        if self.enable_overlay and self.episode_builder is not None:
            notes = self.episode_builder.planner_notes()
            if notes:
                return base + "\n\n" + "\n".join(notes)
        return base

    def add_memory(self, mas_message: MASMessage):
        if self.freeze_memory:
            return
        self.committed_messages.append(mas_message)
        self._append_record(mas_message)
        derived = self._derive_insights_from_message(mas_message)
        if derived:
            self.insight_bank = _dedupe_keep_order(derived + self.insight_bank)[:50]
            self._save_insights()

    def save_task_context(self, label: bool, feedback: str = None, **kargs) -> MASMessage:
        saved = super().save_task_context(label=label, feedback=feedback, **kargs)
        if self.episode_builder is not None:
            saved.add_extra_field("memco_step_count", self.episode_builder.state.step_count)
            saved.add_extra_field("memco_recent_actions", list(self.episode_builder.state.recent_actions))
        self._update_memco_core_feedback_from_episode(label=label, **kargs)
        if self._dynamic_graph_enabled and not self.freeze_memory:
            self._update_dynamic_graph_from_env(label=label, **kargs)
        return saved

    def _update_dynamic_graph_from_env(self, *, label: bool, **kargs) -> None:
        env_ref = kargs.get("env_ref")
        export_history = getattr(env_ref, "export_memco_history", None)
        if env_ref is None or not callable(export_history):
            self._external_error = "dynamic MemCo graph update skipped: missing exportable env_ref"
            return
        scene = self._resolve_external_owner_scene(kargs.get("task_config"), env_ref)
        local_memory = self._external_local_memories.get(scene)
        if local_memory is None:
            from .memco_backend.graph_types import LocalGraphMemory

            local_memory = LocalGraphMemory(agent_id=f"agent_{scene}")
            self._external_local_memories[scene] = local_memory

        history_dir = Path(self.persist_dir) / "dynamic_histories"
        status = "success" if label else "fail"
        model_id = str((kargs.get("task_config") or {}).get("model_type", "") or "")
        try:
            history_path = export_history(
                str(history_dir),
                model_id=model_id,
                status_override=status,
            )
            if not history_path:
                return
            adapter = self._resolve_external_adapter(kargs.get("task_config"), env_ref)
            episode = adapter.episode_from_history(str(history_path), agent_id=local_memory.agent_id)
            episode_graph = self._external_builder.build(episode)
            self._external_maintainer.update(local_memory, episode_graph, episode)
            self._external_maintainer.refine_memory(local_memory)
            self._refresh_shared_global_memory()
            if self.promotion_policy == "wilson":
                # A Wilson snapshot is a pure function of all local evidence
                # currently available.  Starting empty also removes records
                # that cease to satisfy the lower-bound criterion.
                promotion_base = self._external_empty_global()
                promotion_locals = self._wilson_promotion_locals(scene, local_memory)
            else:
                promotion_base = (
                    self._external_global_memory or self._external_empty_global()
                    if self._external_shared_global_dir is not None
                    else self._external_empty_global()
                )
                promotion_locals = (
                    [local_memory]
                    if self._external_shared_global_dir is not None
                    else list(self._external_local_memories.values())
                )
            self._external_global_memory = self._external_promoter.promote(
                promotion_base,
                promotion_locals,
                batch_name=f"dynamic_{scene}_{len(local_memory.episode_ids)}",
            )
            self._persist_shared_global_memory()
            self._persist_dynamic_graph_memory()
        except Exception as exc:
            self._external_error = f"dynamic MemCo graph update failed: {type(exc).__name__}: {exc}"

    def _persist_dynamic_graph_memory(self) -> None:
        if self._external_artifact_dir is None:
            return
        self._external_artifact_dir.mkdir(parents=True, exist_ok=True)
        local_artifact_scene = self._memco_core_local_artifact_scene(self._external_artifact_dir)
        local_items = self._external_local_memories.items()
        if local_artifact_scene:
            local_memory = self._external_local_memories.get(local_artifact_scene)
            local_items = [(local_artifact_scene, local_memory)] if local_memory is not None else []
        for scene, memory in local_items:
            path = self._external_artifact_dir / f"local_{scene}.json"
            with path.open("w", encoding="utf-8") as writer:
                json.dump(
                    self._external_local_to_dict(memory, include_topology=True), writer, ensure_ascii=False, indent=2
                )
        global_path = self._external_artifact_dir / "global_memory.json"
        promoted_batches = self._dedupe_promoted_batches(
            list(getattr(self._external_global_memory, "promoted_batches", []) or [])
        )
        try:
            self._external_global_memory.promoted_batches = promoted_batches
        except Exception:
            pass
        with global_path.open("w", encoding="utf-8") as writer:
            json.dump(self._external_global_to_dict(self._external_global_memory), writer, ensure_ascii=False, indent=2)
        self._persist_wilson_promotion_outputs(self._external_artifact_dir)
        summary = {
            "mode": "dynamic_graph",
            "locals": {
                scene: {
                    "episodes": len(memory.episode_ids),
                    "candidates": len(memory.candidates),
                    "rules": len(memory.rules_by_id),
                    "artifacts": len(memory.artifacts_by_id),
                    "nodes": len(memory.nodes_by_signature),
                    "edges": len(memory.edges_by_signature),
                }
                for scene, memory in self._external_local_memories.items()
            },
            "global": {
                "candidate_count": len(getattr(self._external_global_memory, "candidates", {})),
                "rule_count": len(getattr(self._external_global_memory, "rules_by_id", {})),
                "artifact_count": len(getattr(self._external_global_memory, "artifacts_by_id", {})),
                "promoted_batches": promoted_batches,
            },
        }
        with (self._external_artifact_dir / "summary.json").open("w", encoding="utf-8") as writer:
            json.dump(summary, writer, ensure_ascii=False, indent=2)

    def _overwrite_lightweight_memory_files(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        with open(self._records_path, "w", encoding="utf-8"):
            pass
        with open(self._insights_path, "w", encoding="utf-8") as writer:
            json.dump([], writer, ensure_ascii=False, indent=2)

    def _append_record(self, mas_message: MASMessage) -> None:
        record = MASMessage.to_dict(mas_message)
        with open(self._records_path, "a", encoding="utf-8") as writer:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_messages(self) -> list[MASMessage]:
        records: list[MASMessage] = []
        if not os.path.exists(self._records_path):
            return records
        with open(self._records_path, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(MASMessage.from_dict(json.loads(line)))
                except Exception:
                    continue
        return records

    def _load_insights(self) -> list[str]:
        if not os.path.exists(self._insights_path):
            return []
        try:
            with open(self._insights_path, "r", encoding="utf-8") as reader:
                data = json.load(reader)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [str(item) for item in data if str(item).strip()]

    def _save_insights(self) -> None:
        with open(self._insights_path, "w", encoding="utf-8") as writer:
            json.dump(self.insight_bank, writer, ensure_ascii=False, indent=2)

    def _derive_insights_from_message(self, mas_message: MASMessage) -> list[str]:
        hints: list[str] = []
        trajectory = str(mas_message.task_trajectory or "")
        task_main = str(mas_message.task_main or "")
        task_description = str(mas_message.task_description or "")
        is_fever = (
            task_main.strip().lower().startswith("claim:")
            or task_description.strip().lower().startswith("claim:")
            or "Search[" in trajectory
            or "Lookup[" in trajectory
            or "Finish[" in trajectory
        )
        if is_fever:
            if bool(mas_message.label):
                hints.append(
                    "[MemCo-FEVER] Treat memory as search workflow only; decide SUPPORTS/REFUTES/NOT ENOUGH INFO from current evidence."
                )
            if "No Results" in trajectory:
                hints.append(
                    "[MemCo-FEVER] After No Results, reformulate the current claim query; do not repeat an old entity-specific lookup."
                )
            return hints
        if "Nothing happens." in trajectory:
            hints.append("[MemCo] Avoid repeating actions that already returned 'Nothing happens.'.")
        if bool(mas_message.label):
            hints.append(
                "[MemCo] Prefer action sequences that preserve forward progress and update plans from the latest observation."
            )
        if "open" in trajectory.lower():
            hints.append(
                "[MemCo] Check receptacle state before interacting; opening a container is often a useful precondition."
            )
        return hints


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered
