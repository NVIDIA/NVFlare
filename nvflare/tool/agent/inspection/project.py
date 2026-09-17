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

"""Forward-only project containment for integration and job guards."""

from __future__ import annotations

from pathlib import Path

from nvflare.tool.agent.inspection.types import SourceScan

SECONDARY_PARTS = {
    "archive",
    "archives",
    "archived",
    "fixture",
    "fixtures",
    "test",
    "tests",
    "vendor",
    "vendors",
    "vendored",
}


class LocalImportGraph:
    def __init__(self, scan: SourceScan):
        self.edges = {path: set() for path in scan.facts}
        self.local_nvflare_shadow_files: set[str] = set()
        self._closures: dict[str, set[str]] = {}
        for path, facts in scan.facts.items():
            for module, level, names in facts.local_imports:
                resolved = _resolve_local_import(scan, path, module, level, names)
                self.edges[path].update(resolved)
                if (
                    level == 0
                    and module
                    and _is_nvflare_module(module)
                    and (resolved or _has_local_nvflare_root(scan, path))
                ):
                    self.local_nvflare_shadow_files.add(path)

    def closure(self, start: str) -> set[str]:
        if start not in self._closures:
            result: set[str] = set()
            stack = [start]
            while stack:
                path = stack.pop()
                if path in result:
                    continue
                result.add(path)
                stack.extend(self.edges.get(path, ()))
            self._closures[start] = result
        return set(self._closures[start])


def integration(scan: SourceScan, ownership: dict, graph: LocalImportGraph) -> dict:
    evidence_files = {
        path
        for path, facts in scan.facts.items()
        if path not in graph.local_nvflare_shadow_files and (facts.client_calls or facts.possible_client_calls)
    }
    scopes, ambiguous = _authority_scopes(scan, ownership, graph, evidence_files)
    credible = False
    possible = not scan.complete or ambiguous
    evidence: list[dict] = []
    for scope in scopes:
        scope_receive = False
        scope_send = False
        for path in sorted(scope):
            facts = scan.facts[path]
            if path in graph.local_nvflare_shadow_files:
                continue
            calls = {kind for kind, _ in facts.client_calls}
            direct = bool(calls & {"FLModel", "patch"}) or {"receive", "send"} <= calls
            credible |= direct
            scope_receive |= "receive" in calls
            scope_send |= "send" in calls
            possible |= bool(facts.possible_client_calls)
            possible |= bool(calls & {"receive", "send"}) and not direct
            evidence.extend({"file": path, "line": line, "kind": "client_api"} for _, line in facts.client_calls)
            evidence.extend({"file": path, "line": line, "kind": "client_api"} for line in facts.possible_client_calls)
        possible |= scope_receive and scope_send and not credible
    if ambiguous:
        for path in sorted(evidence_files):
            facts = scan.facts[path]
            evidence.extend({"file": path, "line": line, "kind": "client_api"} for _, line in facts.client_calls)
            evidence.extend({"file": path, "line": line, "kind": "client_api"} for line in facts.possible_client_calls)
    complete = scan.complete and not ambiguous
    if credible:
        state, reason = "converted", "direct_client_api"
    elif possible:
        state, reason = "possible", "incomplete_scan" if not scan.complete else "ambiguous_client_api"
    else:
        state, reason = "none", "no_client_api"
    return {"state": state, "complete": complete, "reason": reason, "evidence": evidence}


def owner_scopes(scan: SourceScan, ownership: dict, graph: LocalImportGraph) -> list[set[str]]:
    if ownership["state"] != "clear":
        return []
    owner_files = {
        path
        for path, facts in scan.facts.items()
        if any(framework == ownership["framework"] for framework, _, _ in facts.owners)
    }
    scopes = [graph.closure(path) for path in sorted(owner_files)]
    if scan.target.is_dir():
        for path in sorted(scan.facts):
            if _is_root(path) and not is_secondary(path) and graph.closure(path) & owner_files:
                scopes.append(graph.closure(path))
    return _deduplicate_scopes(scopes)


def candidate_scopes(
    scan: SourceScan, graph: LocalImportGraph, evidence_files: set[str]
) -> tuple[list[set[str]], bool]:
    if not evidence_files:
        return [], False
    if scan.target.is_file():
        return [{"."}], False
    roots = [
        path
        for path in sorted(scan.facts)
        if _is_root(path) and not is_secondary(path) and graph.closure(path) & evidence_files
    ]
    independent_roots = [
        path for path in roots if not any(path in graph.closure(other) for other in roots if other != path)
    ]
    if len(independent_roots) == 1:
        return [graph.closure(independent_roots[0])], False
    if len(independent_roots) > 1:
        return [], True
    active = sorted(path for path in evidence_files if not is_secondary(path))
    if len(active) == 1:
        return [graph.closure(active[0])], False
    return ([], bool(active))


def is_secondary(path: str) -> bool:
    return bool(set(Path(path).parts[:-1]) & SECONDARY_PARTS)


def _authority_scopes(
    scan: SourceScan, ownership: dict, graph: LocalImportGraph, evidence_files: set[str]
) -> tuple[list[set[str]], bool]:
    scopes = owner_scopes(scan, ownership, graph)
    if scopes:
        return scopes, False
    return candidate_scopes(scan, graph, evidence_files)


def _resolve_local_import(
    scan: SourceScan, source: str, module: str | None, level: int, names: tuple[str, ...]
) -> set[str]:
    source_parent = Path(source).parent if source != "." else Path()
    module_parts = tuple(part for part in (module or "").split(".") if part)
    if level:
        base = source_parent
        for _ in range(level - 1):
            base = base.parent
        bases = [base / Path(*module_parts)]
    else:
        imported = Path(*module_parts) if module_parts else Path()
        bases = [imported, source_parent / imported, Path("src") / imported]
    resolved_candidates: set[Path] = set()
    for base in bases:
        exact = {base.with_suffix(".py"), base / "__init__.py"}
        for name in names:
            exact.update({base / f"{name}.py", base / name / "__init__.py"})
        resolved_candidates.update(candidate for candidate in exact if candidate.as_posix() in scan.facts)
    resolved_candidates = _prefer_importer_packaging_root(resolved_candidates, source)
    candidates = set(resolved_candidates)
    for candidate in resolved_candidates:
        parent = candidate.parent
        while parent != Path(".") and parent.parts != ("src",):
            candidates.add(parent / "__init__.py")
            parent = parent.parent
    return {path.as_posix() for path in candidates if path.as_posix() in scan.facts}


def _prefer_importer_packaging_root(candidates: set[Path], source: str) -> set[Path]:
    if len(candidates) <= 1:
        return candidates
    source_root = _packaging_root(Path(source))
    same_root = {candidate for candidate in candidates if _packaging_root(candidate) == source_root}
    return same_root or candidates


def _packaging_root(path: Path) -> str:
    return "src" if path.parts and path.parts[0] == "src" else ""


def _is_nvflare_module(module: str) -> bool:
    return module == "nvflare" or module.startswith("nvflare.")


def _has_local_nvflare_root(scan: SourceScan, source: str) -> bool:
    source_parent = Path(source).parent if source != "." else Path()
    package_root = Path(_packaging_root(Path(source)))
    roots = {package_root / "nvflare.py", package_root / "nvflare/__init__.py"}
    roots.update({source_parent / "nvflare.py", source_parent / "nvflare/__init__.py"})
    if any(candidate.as_posix() in scan.files_seen for candidate in roots):
        return True
    scan_root = scan.target.parent if scan.target.is_file() else scan.target
    return any((scan_root / candidate).is_file() or (scan_root / candidate).is_symlink() for candidate in roots)


def _deduplicate_scopes(scopes: list[set[str]]) -> list[set[str]]:
    result: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for scope in scopes:
        key = frozenset(scope)
        if key not in seen:
            seen.add(key)
            result.append(scope)
    return result


def _is_root(path: str) -> bool:
    return path != "." and Path(path).parent == Path(".")
