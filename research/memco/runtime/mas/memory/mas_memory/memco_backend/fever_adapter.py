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

import json
import os
import re
from pathlib import Path
from typing import Any

from .graph_types import ActionFamily, CanonicalAction, Domain, EpisodeRecord, EpisodeStep, StateSummary, StepFeedback


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


def _short_text(text: str, *, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())[:limit]


def _extract_bracket_arg(action: str) -> tuple[str, str]:
    match = re.match(r"^\s*([A-Za-z_]+)\[(.*)\]\s*$", str(action or "").strip())
    if not match:
        return "", ""
    return match.group(1), match.group(2).strip()


def _memco_fever_policy_enabled() -> bool:
    return str(os.environ.get("NV_MEMCO_FEVER_POLICY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "adaptive_cache",
    }


class FeverAdapter:
    """GraphMemory adapter for FEVER search/lookup/finish trajectories."""

    domain_name = "fever"

    @staticmethod
    def _is_search_miss_text(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            marker in lowered
            for marker in (
                "could not find",
                "cannot find",
                "searcherrors",
                "jsondecodeerror",
                "pageerror",
                "disambiguationerror",
            )
        )

    def canonicalize_action(self, raw_action: str) -> CanonicalAction:
        surface = str(raw_action or "").strip()
        action_type, argument = _extract_bracket_arg(surface)
        lowered = surface.lower()
        if "thought" in lowered and not action_type:
            return CanonicalAction(Domain.FEVER, ActionFamily.OTHER, "thought", surface_form=surface)
        if action_type.lower() == "search":
            return CanonicalAction(
                Domain.FEVER,
                ActionFamily.SEARCH,
                "search",
                {"query": _normalize(argument)},
                surface_form=surface,
            )
        if action_type.lower() == "lookup":
            return CanonicalAction(
                Domain.FEVER,
                ActionFamily.INSPECT,
                "lookup",
                {"keyword": _normalize(argument)},
                surface_form=surface,
            )
        if action_type.lower() == "finish":
            return CanonicalAction(
                Domain.FEVER,
                ActionFamily.CONTROL,
                "finish",
                {"label": _normalize(argument)},
                surface_form=surface,
            )
        verb = _normalize(surface.split(" ", 1)[0] if surface else "unknown") or "unknown"
        return CanonicalAction(Domain.FEVER, ActionFamily.OTHER, verb, surface_form=surface)

    def derive_scene_id(self, domain: str | None = None, history_path: str | None = None) -> str:
        if domain:
            return f"fever:{_normalize(domain)}"
        if history_path:
            stem = Path(history_path).stem
            match = re.search(r"history_fever_([^_]+)_", stem)
            if match:
                return f"fever:{_normalize(match.group(1))}"
        return "fever:default"

    def derive_layout_id(self, domain: str | None = None, task_id: Any = None, history_path: str | None = None) -> str:
        return f"{self.derive_scene_id(domain, history_path)}:{task_id if task_id is not None else 'unknown'}"

    def infer_task_family(self, claim: str, domain: str | None = None) -> str:
        if _memco_fever_policy_enabled():
            if domain:
                return f"fever:{_normalize(domain)}"
            return "fever:claim_verification"
        # ALFWorld uses task_family for the abstract task class, while scene_id
        # carries the concrete domain/layout.  FEVER should follow the same
        # split: scene_id remains A_film_tv/B_music, and task_family is the
        # claim-verification pattern used by global artifacts/routing.
        claim_type = str(self._claim_profile(claim).get("claim_type", "") or "general_fact")
        return f"fever:{_normalize(claim_type)}"

    def goal_slots(self, goal: str) -> dict[str, str]:
        entity = self._claim_anchor(goal)
        return {"object": entity} if entity else {}

    def required_goal_count(self, goal: str) -> int:
        return 1

    def build_state(
        self,
        *,
        scene_id: str,
        claim: str,
        observation: str,
        admissible_commands: list[str] | tuple[str, ...] = (),
        stage: str = "",
    ) -> StateSummary:
        evidence_tokens = tuple(
            dict.fromkeys(_normalize(x) for x in self._evidence_phrases(observation) if _normalize(x))
        )[:16]
        verbs = tuple(
            dict.fromkeys(self.canonicalize_action(cmd).verb for cmd in admissible_commands if str(cmd).strip())
        )
        return StateSummary(
            domain=Domain.FEVER,
            scene_id=scene_id,
            visible_objects=evidence_tokens,
            admissible_verbs=verbs,
            workflow_stage=stage or "need_search",
            raw_observation=_short_text(observation, limit=600),
        )

    def build_query(self, *, env_ref: Any, task_config: dict, query_task: str, MemoryQuery: Any, CandidateType: Any):
        claim = str(
            getattr(env_ref, "claim", "")
            or getattr(env_ref, "current_task", "")
            or task_config.get("task", "")
            or query_task
            or ""
        ).strip()
        claim = claim.removeprefix("Claim:").strip()
        if not claim:
            return None
        domain = str(task_config.get("ab_domain", "") or getattr(env_ref, "ab_domain", "") or "default")
        scene_id = self.derive_scene_id(domain)
        history = [row for row in (getattr(env_ref, "current_history", []) or []) if isinstance(row, dict)]
        current = history[-1] if history else {}
        observation = str(current.get("Observation") or "")
        admissible = [str(cmd) for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()]
        progress_state = self._progress_state(history, done=bool(current.get("Done", False)))
        state = self.build_state(
            scene_id=scene_id,
            claim=claim,
            observation=observation,
            admissible_commands=admissible,
            stage=progress_state,
        )
        env_config = getattr(env_ref, "config", {}) if hasattr(env_ref, "config") else {}
        label = str(task_config.get("answer", "") or env_config.get("answer", "") or "")
        desired_types = [CandidateType.PRECONDITION, CandidateType.WORKFLOW]
        if progress_state in {"search_failed", "invalid_action"}:
            desired_types.append(CandidateType.FAILURE)
        if _memco_fever_policy_enabled():
            entity = self._claim_anchor(claim)
            keywords = tuple(
                dict.fromkeys(
                    [
                        f"domain={domain}",
                        f"progress={progress_state}",
                        f"label={_normalize(label)}" if label else "",
                        *(_normalize(x) for x in self._claim_keywords(claim)[:10]),
                        *(_normalize(x) for x in self._evidence_phrases(observation)[:8]),
                    ]
                )
            )
            return MemoryQuery(
                goal=f"Verify claim: {claim}",
                scene_id=scene_id,
                current_stage=state.workflow_stage,
                progress_state=progress_state,
                task_family=self.infer_task_family(claim, domain),
                goal_roles={"object": entity} if entity else {},
                required_count=1,
                placed_relevant_count=1 if progress_state in {"ready_finish", "done"} else 0,
                remaining_relevant_count=0 if progress_state == "done" else 1,
                destination_reached=progress_state == "done",
                goal_object_matches_visible=progress_state in {"need_lookup_or_finish", "ready_finish", "done"},
                admissible_actions=tuple(self.canonicalize_action(cmd) for cmd in admissible),
                desired_types=tuple(desired_types),
                failure_label=self._last_failure(history),
                keywords=tuple(k for k in keywords if k),
                belief={
                    "claim": claim,
                    "answer": label,
                    "history_len": len(history),
                    "last_action": str(current.get("Action", "") or ""),
                },
                dynamic_context={
                    "visible_objects": list(state.visible_objects),
                    "layout_id": self.derive_layout_id(domain, task_config.get("task_id", "")),
                    "task_config_env_name": str(task_config.get("env_name", "")),
                    "memco_domain": self.domain_name,
                },
            )
        profile = self._claim_profile(claim)
        entity = str(profile.get("primary_entity", "") or self._claim_anchor(claim))
        relation_keywords = [str(x) for x in profile.get("relation_keywords", []) if str(x).strip()]
        claim_type = str(profile.get("claim_type", "") or "general_fact")
        secondary_entity = str(profile.get("secondary_entity", "") or "")
        query_variants = [str(x) for x in profile.get("query_variants", []) if str(x).strip()]
        claim_polarity = str(profile.get("claim_polarity", "") or "positive")
        keywords = tuple(
            dict.fromkeys(
                [
                    f"domain={domain}",
                    f"progress={progress_state}",
                    f"claim_type={claim_type}",
                    f"claim_polarity={claim_polarity}",
                    f"relation={relation_keywords[0]}" if relation_keywords else "",
                    f"secondary={secondary_entity}" if secondary_entity else "",
                    f"label={_normalize(label)}" if label else "",
                    *(_normalize(x) for x in query_variants[:4]),
                    *(_normalize(x) for x in self._claim_keywords(claim)[:10]),
                    *(_normalize(x) for x in relation_keywords[:6]),
                    *(_normalize(x) for x in self._evidence_phrases(observation)[:8]),
                ]
            )
        )
        goal_roles = {
            "claim_type": claim_type,
            "object": entity,
            "claim_polarity": claim_polarity,
        }
        if relation_keywords:
            goal_roles["relation"] = _normalize(relation_keywords[0])
        if secondary_entity:
            goal_roles["secondary"] = _normalize(secondary_entity)
        return MemoryQuery(
            goal=f"Verify claim: {claim}",
            scene_id=scene_id,
            current_stage=state.workflow_stage,
            progress_state=progress_state,
            task_family=self.infer_task_family(claim, domain),
            goal_roles={key: value for key, value in goal_roles.items() if value},
            required_count=1,
            placed_relevant_count=1 if progress_state in {"ready_finish", "done"} else 0,
            remaining_relevant_count=0 if progress_state == "done" else 1,
            destination_reached=progress_state == "done",
            goal_object_matches_visible=progress_state in {"need_lookup_or_finish", "ready_finish", "done"},
            admissible_actions=tuple(self.canonicalize_action(cmd) for cmd in admissible),
            desired_types=tuple(desired_types),
            failure_label=self._last_failure(history),
            keywords=tuple(k for k in keywords if k),
            belief={
                "claim": claim,
                "answer": label,
                "claim_type": claim_type,
                "primary_entity": entity,
                "secondary_entity": secondary_entity,
                "relation_keywords": relation_keywords,
                "query_variants": query_variants,
                "claim_polarity": claim_polarity,
                "history_len": len(history),
                "last_action": str(current.get("Action", "") or ""),
                "fever_health": self._history_health(history),
            },
            dynamic_context={
                "visible_objects": list(state.visible_objects),
                "layout_id": self.derive_layout_id(domain, task_config.get("task_id", "")),
                "task_config_env_name": str(task_config.get("env_name", "")),
                "memco_domain": self.domain_name,
                "query_variants": query_variants,
                "fever_health": self._history_health(history),
            },
        )

    def episode_from_history(self, history_path: str, agent_id: str) -> EpisodeRecord:
        with open(history_path, "r", encoding="utf-8") as reader:
            payload = json.load(reader)
        claim = str(payload.get("claim") or payload.get("game_task") or "").removeprefix("Claim:").strip()
        domain = str(payload.get("ab_domain") or payload.get("domain") or "default")
        task_id = str(payload.get("task_id") or payload.get("game_index") or Path(history_path).stem)
        scene_id = self.derive_scene_id(domain, history_path)
        history = [row for row in payload.get("history", []) if isinstance(row, dict)]
        episode = EpisodeRecord(
            agent_id=agent_id,
            scene_id=scene_id,
            task_id=task_id,
            goal=f"Verify claim: {claim}",
            metadata={
                "status": payload.get("status", ""),
                "final_score": float(payload.get("final_score", 0.0) or 0.0),
                "layout_id": self.derive_layout_id(domain, task_id, history_path),
                "memco_domain": self.domain_name,
                "task_family": self.infer_task_family(claim, domain),
            },
        )
        previous = history[0] if history else {}
        prev_state = self._state_from_record(previous, scene_id=scene_id, claim=claim, history_prefix=history[:1])
        for idx, record in enumerate(history[1:], start=1):
            action_text = str(record.get("Action") or "").strip()
            if not action_text:
                continue
            next_state = self._state_from_record(
                record, scene_id=scene_id, claim=claim, history_prefix=history[: idx + 1]
            )
            failure_label = self._failure_label(record)
            success = failure_label is None and not str(record.get("Action Type", "")).lower() == "thought"
            episode.steps.append(
                EpisodeStep(
                    step_idx=len(episode.steps),
                    state=prev_state,
                    action=self.canonicalize_action(action_text),
                    next_state=next_state,
                    feedback=StepFeedback(
                        success=bool(success),
                        score=float(record.get("Score", 0.0) or 0.0),
                        done=bool(record.get("Done", False)),
                        failure_label=failure_label,
                        state_delta=self._state_delta(prev_state, next_state),
                    ),
                    subgoal=str(next_state.workflow_stage or ""),
                )
            )
            prev_state = next_state
        return episode

    def _state_from_record(
        self,
        record: dict[str, Any],
        *,
        scene_id: str,
        claim: str,
        history_prefix: list[dict[str, Any]],
    ) -> StateSummary:
        return self.build_state(
            scene_id=scene_id,
            claim=claim,
            observation=str(record.get("Observation") or ""),
            admissible_commands=[str(x) for x in (record.get("Admissible Commands") or [])],
            stage=self._progress_state(history_prefix, done=bool(record.get("Done", False))),
        )

    @staticmethod
    def _progress_state(history: list[dict[str, Any]], *, done: bool = False) -> str:
        if done:
            return "done"
        actions = [str(row.get("Action", "") or "") for row in history if str(row.get("Action", "") or "").strip()]
        observations = [
            str(row.get("Observation", "") or "") for row in history if str(row.get("Observation", "") or "").strip()
        ]
        if any("invalid action" in obs.lower() for obs in observations[-2:]):
            return "invalid_action"
        if _memco_fever_policy_enabled():
            if any(action.lower().startswith("lookup[") for action in actions):
                return "ready_finish"
            if any(action.lower().startswith("search[") for action in actions):
                if observations and "cannot find" in observations[-1].lower():
                    return "search_failed"
                return "need_lookup_or_finish"
            return "need_search"
        last_action = actions[-1].lower() if actions else ""
        if last_action.startswith("lookup["):
            return "ready_finish"
        if last_action.startswith("search["):
            if observations and FeverAdapter._is_search_miss_text(observations[-1]):
                return "search_failed"
            return "need_lookup_or_finish"
        if last_action.startswith("finish["):
            return "done"
        return "need_search"

    @staticmethod
    def _history_health(history: list[dict[str, Any]]) -> dict[str, int]:
        search_misses = 0
        search_recoveries = 0
        finish_after_search_miss = 0
        nei_after_single_failed_search = 0
        last_search_was_miss = False
        total_searches = 0
        for row in history:
            action = str(row.get("Action", "") or "")
            obs = str(row.get("Observation", "") or "")
            action_lower = action.lower()
            if action_lower.startswith("search["):
                total_searches += 1
                is_miss = FeverAdapter._is_search_miss_text(obs)
                if is_miss:
                    search_misses += 1
                elif last_search_was_miss:
                    search_recoveries += 1
                last_search_was_miss = is_miss
                continue
            if action_lower.startswith("finish["):
                if last_search_was_miss:
                    finish_after_search_miss += 1
                if last_search_was_miss and total_searches <= 1 and "not enough info" in action_lower:
                    nei_after_single_failed_search += 1
        return {
            "search_miss_count": search_misses,
            "search_recovery_attempt_count": search_recoveries,
            "finish_after_search_error_count": finish_after_search_miss,
            "nei_after_single_failed_search_count": nei_after_single_failed_search,
        }

    @staticmethod
    def _last_failure(history: list[dict[str, Any]]) -> str | None:
        for row in reversed(history):
            failure = FeverAdapter._failure_label(row)
            if failure:
                return failure
        return None

    @staticmethod
    def _failure_label(record: dict[str, Any]) -> str | None:
        obs = str(record.get("Observation") or "").lower()
        if "invalid action" in obs:
            return "invalid_action"
        if _memco_fever_policy_enabled():
            if "cannot find" in obs:
                return "search_not_found"
            if "last page searched was not found" in obs:
                return "lookup_without_page"
            if bool(record.get("Done", False)) and float(record.get("Score", 0.0) or 0.0) <= 0:
                return "wrong_finish_label"
            return None
        if FeverAdapter._is_search_miss_text(obs):
            return "search_not_found"
        if "last page searched was not found" in obs:
            return "lookup_without_page"
        if bool(record.get("Done", False)) and float(record.get("Score", 0.0) or 0.0) <= 0:
            return "wrong_finish_label"
        return None

    @staticmethod
    def _state_delta(prev_state: StateSummary, next_state: StateSummary) -> tuple[str, ...]:
        prev = set(prev_state.visible_objects)
        nxt = set(next_state.visible_objects)
        deltas = [f"evidence+={item}" for item in sorted(nxt - prev)[:6]]
        if prev_state.workflow_stage != next_state.workflow_stage:
            deltas.append(f"stage={next_state.workflow_stage}")
        return tuple(deltas)

    @staticmethod
    def _claim_anchor(claim: str) -> str:
        text = str(claim or "").strip()
        text = re.sub(r"^\s*(?:verify\s+claim|claim)\s*:\s*", "", text, flags=re.I).strip()
        if _memco_fever_policy_enabled():
            predicate = re.search(
                r"\b(?:is|are|was|were|has|have|had|does|do|did|worked|appeared|released|formed|created|directed|starred)\b",
                text,
                flags=re.I,
            )
            if predicate:
                subject = text[: predicate.start()].strip(" .,:;\"'")
                if 1 <= len(re.findall(r"[A-Za-z0-9]+", subject)) <= 8:
                    return _normalize(subject)
            candidates = re.findall(
                r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]\.)(?:\s+(?:the|of|and|in|on|for|to|a|an|[A-Z][A-Za-z0-9]*|[A-Z]\.)){0,5}",
                text,
            )
            if candidates:
                return _normalize(candidates[0])
            words = FeverAdapter._claim_keywords(claim)
            return _normalize(words[0]) if words else ""
        # FEVER search should usually start from the claim subject.  The older
        # title-case regex collapsed names with lowercase connectors, e.g.
        # "Off the Wall" -> "Off", which produces weak Search hints.
        predicate = re.search(
            r"\b(?:is|are|was|were|has|have|had|does|do|did|worked|appeared|released|formed|created|directed|stars|starred|starring|features|featured|includes|included)\b",
            text,
            flags=re.I,
        )
        called = re.search(r"\bthere\s+(?:is|are|was|were)\b.*?\bcalled\s+(.+?)(?:[.;]|$)", text, flags=re.I)
        if called:
            candidate = called.group(1).strip(" .,:;\"'")
            if candidate:
                return _normalize(candidate)
        if predicate:
            subject = text[: predicate.start()].strip(" .,:;\"'")
            subject = re.sub(r"\b(?:only|also|just)\s*$", "", subject, flags=re.I).strip(" .,:;\"'")
            if 1 <= len(re.findall(r"[A-Za-z0-9]+", subject)) <= 8:
                return _normalize(subject)
        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]\.)(?:\s+(?:(?:the|of|and|in|on|for|to|a|an)\b|[A-Z][A-Za-z0-9]*|[A-Z]\.)){0,5}",
            text,
        )
        if candidates:
            return _normalize(candidates[0])
        words = FeverAdapter._claim_keywords(claim)
        return _normalize(words[0]) if words else ""

    @staticmethod
    def _claim_profile(claim: str) -> dict[str, Any]:
        """Cheap structural routing for FEVER claims.

        The graph should transfer evidence workflows ("cast membership",
        "album track membership", etc.) instead of memorizing labels or old
        entity names.  These heuristics intentionally stay deterministic so
        they add almost no runtime overhead.
        """
        text = str(claim or "").strip()
        norm = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
        primary = FeverAdapter._claim_anchor(text)
        secondary = FeverAdapter._secondary_entity_hint(text, primary)
        claim_polarity = "negative" if re.search(r"\b(?:not|never|no|without|incapable)\b", norm) else "positive"

        claim_type = "general_fact"
        relation_keywords: list[str] = []
        if any(marker in norm for marker in (" soundtrack", " track", " song ", " album", " single")):
            claim_type = "album_track_membership"
            relation_keywords = ["soundtrack", "track listing", "album", "song"]
        elif any(
            marker in norm
            for marker in (
                " cast",
                " stars ",
                " starring",
                " starred",
                " appeared in",
                " actor",
                " actress",
                " played ",
            )
        ):
            claim_type = "cast_membership"
            relation_keywords = ["cast", "starring", "appeared", "played by"]
        elif any(
            marker in norm
            for marker in (" directed", " director", " created", " creator", " producer", " wrote", " written")
        ):
            claim_type = "creative_role"
            relation_keywords = ["director", "creator", "producer", "writer"]
        elif any(
            marker in norm for marker in (" released", " release date", " premiered", " publication", " published")
        ):
            claim_type = "release_date"
            relation_keywords = ["release date", "released", "premiered", "published"]
        elif any(
            marker in norm for marker in (" genre", " style", " type of music", " rock", " pop", " hip hop", " jazz")
        ):
            claim_type = "genre"
            relation_keywords = ["genre", "style", "music genre"]
        elif any(marker in norm for marker in (" award", " won", " nominated", " grammy", " oscar", " emmy")):
            claim_type = "award"
            relation_keywords = ["awards", "nominations", "won"]
        elif any(
            marker in norm
            for marker in (" singer", " musician", " band", " guitarist", " vocalist", " rapper", " composer")
        ):
            claim_type = "music_role"
            relation_keywords = ["occupation", "members", "associated acts", "music career"]
        elif any(marker in norm for marker in (" born", " nationality", " country", " city", " located", " from ")):
            claim_type = "nationality_location"
            relation_keywords = ["born", "nationality", "origin", "location"]
        elif any(marker in norm for marker in (" occupation", " profession", " known for", " worked as")):
            claim_type = "profession"
            relation_keywords = ["occupation", "known for", "career"]
        elif any(marker in norm for marker in (" spouse", " married", " member of", " part of", " belongs to")):
            claim_type = "identity_relation"
            relation_keywords = ["relationship", "member", "spouse"]

        if not relation_keywords:
            relation_keywords = FeverAdapter._claim_relation_keywords(text, primary)
        query_variants = FeverAdapter._claim_query_variants(
            text,
            primary_entity=primary,
            secondary_entity=secondary,
            claim_type=claim_type,
            relation_keywords=relation_keywords,
        )
        return {
            "claim_type": claim_type,
            "primary_entity": primary,
            "secondary_entity": secondary,
            "relation_keywords": relation_keywords[:5],
            "query_variants": query_variants[:6],
            "claim_polarity": claim_polarity,
        }

    @staticmethod
    def _claim_query_variants(
        claim: str,
        *,
        primary_entity: str,
        secondary_entity: str = "",
        claim_type: str = "general_fact",
        relation_keywords: list[str] | None = None,
    ) -> list[str]:
        """Current-claim search routes ordered from most relation-specific to broad.

        These are not memory labels.  They are deterministic query candidates
        used to avoid over-constraining strong models to Search[primary entity]
        when the claim contains a second entity or a specific relation.
        """
        primary = FeverAdapter._display_query(primary_entity)
        secondary = FeverAdapter._display_query(secondary_entity)
        relation_keywords = [FeverAdapter._display_query(x) for x in (relation_keywords or []) if str(x).strip()]
        claim_text = str(claim or "").strip()
        claim_norm = re.sub(r"[^a-z0-9\s]+", " ", claim_text.lower())
        variants: list[str] = []

        def add(value: str) -> None:
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;\"'")
            if cleaned and _normalize(cleaned) not in {_normalize(x) for x in variants}:
                variants.append(cleaned)

        if primary and secondary:
            add(f"{primary} {secondary}")
        if primary and "film career" in claim_norm:
            add(f"{primary} filmography")
            add(f"{primary} film career")
        if primary and claim_type in {"creative_role", "cast_membership", "identity_relation"} and relation_keywords:
            # For relation claims, this preserves Gemini-style combined search
            # (e.g. "Ashton Kutcher Ivan Reitman") before falling back to
            # separate entity pages.
            if secondary:
                add(f"{primary} {secondary}")
            add(f"{primary} {relation_keywords[0]}")
        if primary and claim_type == "album_track_membership":
            add(f"{primary} album")
            add(f"{primary} track listing")
        if primary and claim_type in {"cast_membership", "creative_role", "general_fact"}:
            if any(word in claim_norm for word in (" film", " movie", " drama", " directed", " stars", " starred")):
                add(f"{primary} film")
                add(f"{primary} movie")
            if any(word in claim_norm for word in (" tv ", " television", " series", " show")):
                add(f"{primary} TV series")
        if primary and relation_keywords:
            add(f"{primary} {relation_keywords[0]}")
        if primary:
            add(primary)
        if not variants:
            keywords = FeverAdapter._claim_keywords(claim_text)
            add(" ".join(keywords[:5]))
        return variants

    @staticmethod
    def _display_query(value: str) -> str:
        text = re.sub(r"[_\s]+", " ", str(value or "")).strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _claim_relation_keywords(claim: str, primary_entity: str = "") -> list[str]:
        stop = {
            "claim",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "of",
            "in",
            "on",
            "and",
            "or",
            "to",
            "by",
            "for",
            "with",
            "has",
            "have",
            "had",
            "not",
            "no",
        }
        text = re.sub(r"[^a-z0-9\s]+", " ", str(claim or "").lower())
        primary_tokens = set(re.findall(r"[a-z0-9]+", primary_entity.replace("_", " ")))
        words = [word for word in text.split() if len(word) > 2 and word not in stop and word not in primary_tokens]
        if not words:
            return ["claim relation"]
        # Preserve short adjacent phrases because Lookup often benefits from a
        # relation phrase ("track listing") more than a single leftover token.
        phrases: list[str] = []
        for idx in range(len(words) - 1):
            phrases.append(f"{words[idx]} {words[idx + 1]}")
            if len(phrases) >= 2:
                break
        return phrases + words[:3]

    @staticmethod
    def _secondary_entity_hint(claim: str, primary_entity: str = "") -> str:
        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]\.)(?:\s+(?:(?:the|of|and|in|on|for|to|a|an)\b|[A-Z][A-Za-z0-9]*|[A-Z]\.)){0,5}",
            str(claim or ""),
        )
        primary_norm = _normalize(primary_entity)
        for candidate in candidates[1:]:
            normalized = _normalize(candidate)
            if normalized and normalized != primary_norm:
                return normalized
        return ""

    @staticmethod
    def _claim_keywords(claim: str) -> list[str]:
        stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "and", "or", "to", "by", "for"}
        return [
            word for word in re.findall(r"[A-Za-z0-9]+", str(claim or "").lower()) if word not in stop and len(word) > 2
        ]

    @staticmethod
    def _evidence_phrases(observation: str) -> list[str]:
        text = str(observation or "")
        if not text:
            return []
        pieces = re.split(r"\.\s+|;\s+|\n+", text)
        return [piece.strip(" .") for piece in pieces if piece.strip(" .")][:12]
