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

"""Local-import and project-authority analysis for static inspection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nvflare.tool.agent.inspection.models import InspectionFacts

_PACKAGE_ROOT_DIR_NAMES = {"src"}
_SECONDARY_PROJECT_DIR_NAMES = {
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


@dataclass
class LocalImportGraph:
    """Directed local-import graph with one cached traversal per source file."""

    edges: dict[str, frozenset[str]]
    _reachable_cache: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    @property
    def files(self) -> frozenset[str]:
        return frozenset(self.edges)

    def reachable_from(self, source_file: str) -> frozenset[str]:
        cached = self._reachable_cache.get(source_file)
        if cached is not None:
            return cached

        pending = list(self.edges.get(source_file, ()))
        reachable = set()
        while pending:
            imported_file = pending.pop()
            if imported_file == source_file or imported_file in reachable:
                continue
            reachable.add(imported_file)
            pending.extend(self.edges.get(imported_file, ()))

        result = frozenset(reachable)
        self._reachable_cache[source_file] = result
        return result

    def component_from(self, anchor_file: str) -> frozenset[str]:
        return frozenset({anchor_file}) | self.reachable_from(anchor_file)


@dataclass(frozen=True)
class ProjectComponent:
    anchor_file: str
    files: frozenset[str]

    @property
    def depth(self) -> int:
        return len(Path(self.anchor_file).parts)

    @property
    def directory(self) -> str:
        parent = Path(self.anchor_file).parent.as_posix()
        return "." if parent == "." else parent


@dataclass(frozen=True)
class ProjectScope:
    active_components: tuple[ProjectComponent, ...]
    ambiguous: bool = False

    def component_for(self, file_path: str) -> Optional[ProjectComponent]:
        if self.ambiguous:
            return None
        return next((component for component in self.active_components if file_path in component.files), None)

    def authorizes(self, file_path: str) -> bool:
        return self.component_for(file_path) is not None


@dataclass(frozen=True)
class SourceJobCandidate:
    source_file: str
    flare_import_files: frozenset[str]
    project_component: Optional[ProjectComponent]
    export_supported: bool

    @property
    def authoritative(self) -> bool:
        return self.project_component is not None and bool(self.flare_import_files)


def build_local_import_graph(facts: InspectionFacts) -> LocalImportGraph:
    # Store resolved files, not module names. Resolution first disambiguates
    # root and src-layout copies in the importing file's packaging context.
    local_files_by_module = _local_files_by_module(facts)
    edges = {}
    for source_file, imports in facts.file_imports.items():
        imported_files = set()
        for import_name in imports:
            imported_files.update(_local_files_for_import(import_name, source_file, local_files_by_module))
        edges[source_file] = frozenset(imported_files)
    return LocalImportGraph(edges)


def build_project_scope(facts: InspectionFacts, graph: LocalImportGraph) -> ProjectScope:
    if facts.root.is_file():
        inspected_file = facts.root.name
        component = ProjectComponent(inspected_file, graph.component_from(inspected_file))
        return ProjectScope(active_components=(component,))

    entry_point_files = {entry["path"] for entry in facts.entry_points}
    framework_files = {item["file"] for evidence in facts.framework_evidence.values() for item in evidence}
    anchor_files = entry_point_files | framework_files
    components = tuple(
        ProjectComponent(anchor_file, graph.component_from(anchor_file)) for anchor_file in sorted(anchor_files)
    )
    if not components:
        return ProjectScope(active_components=(ProjectComponent(".", graph.files),))

    project_groups = _component_groups(components)
    group_depths = tuple(min(component.depth for component in group) for group in project_groups)
    nearest_depth = min(group_depths)
    active_groups = tuple(group for group, depth in zip(project_groups, group_depths) if depth == nearest_depth)

    entry_point_groups = tuple(
        group for group in active_groups if any(component.anchor_file in entry_point_files for component in group)
    )
    has_secondary_entry_point = any(
        component.anchor_file in entry_point_files and _is_secondary_project_entry_point(component.anchor_file)
        for group in entry_point_groups
        for component in group
    )
    if entry_point_groups and not has_secondary_entry_point:
        active_groups = entry_point_groups

    active_components = tuple(component for group in active_groups for component in group)
    return ProjectScope(active_components=active_components, ambiguous=len(active_groups) > 1)


def authoritative_source_job_candidate(
    facts: InspectionFacts,
    project_scope: ProjectScope,
    graph: LocalImportGraph,
) -> Optional[SourceJobCandidate]:
    candidates = [
        candidate for candidate in _source_job_candidates(facts, project_scope, graph) if candidate.authoritative
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (len(Path(candidate.source_file).parts), candidate.source_file))


def evidence_tied_to_entry_context(
    facts: InspectionFacts,
    evidence: list[dict],
    graph: Optional[LocalImportGraph] = None,
) -> bool:
    if _evidence_tied_to_inspected_file_or_entry_point(facts, evidence):
        return True
    if facts.root.is_file():
        return False
    import_graph = graph or build_local_import_graph(facts)
    return any(_entry_point_imports_file(facts, item["file"], import_graph) for item in evidence)


def authorizes_flare_evidence(project_scope: ProjectScope, file_path: str) -> bool:
    return project_scope.authorizes(file_path) and not _is_secondary_project_entry_point(file_path)


def _source_job_candidates(
    facts: InspectionFacts,
    project_scope: ProjectScope,
    graph: LocalImportGraph,
) -> tuple[SourceJobCandidate, ...]:
    flare_import_files = frozenset(item["file"] for item in facts.flare_imports)
    source_files = set(facts.job_py_paths) | facts.sim_env_files
    candidates = []
    for source_file in sorted(source_files):
        component_files = graph.component_from(source_file)
        candidates.append(
            SourceJobCandidate(
                source_file=source_file,
                flare_import_files=flare_import_files & component_files,
                project_component=(
                    project_scope.component_for(source_file)
                    if authorizes_flare_evidence(project_scope, source_file)
                    else None
                ),
                export_supported=source_file in facts.export_support_files,
            )
        )
    return tuple(candidates)


def _component_groups(components: tuple[ProjectComponent, ...]) -> tuple[tuple[ProjectComponent, ...], ...]:
    remaining = set(range(len(components)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        group_indices = []
        while pending:
            current_index = pending.pop()
            group_indices.append(current_index)
            current = components[current_index]
            connected = sorted(
                candidate_index
                for candidate_index in remaining
                if _components_share_project(current, components[candidate_index])
            )
            remaining.difference_update(connected)
            pending.extend(connected)
        groups.append(tuple(components[index] for index in sorted(group_indices)))
    return tuple(groups)


def _components_share_project(left: ProjectComponent, right: ProjectComponent) -> bool:
    return left.directory == right.directory or left.anchor_file in right.files or right.anchor_file in left.files


def _is_secondary_project_entry_point(file_path: str) -> bool:
    return any(part.lower() in _SECONDARY_PROJECT_DIR_NAMES for part in Path(file_path).parent.parts)


def _evidence_tied_to_inspected_file_or_entry_point(facts: InspectionFacts, evidence: list[dict]) -> bool:
    if facts.root.is_file():
        return any(item["file"] == facts.root.name for item in evidence)
    entry_point_paths = {entry["path"] for entry in facts.entry_points}
    return any(item["file"] in entry_point_paths for item in evidence)


def _entry_point_imports_file(
    facts: InspectionFacts,
    evidence_file: str,
    graph: Optional[LocalImportGraph] = None,
) -> bool:
    if not _module_names_for_file(evidence_file):
        return False
    import_graph = graph or build_local_import_graph(facts)
    return any(evidence_file in import_graph.reachable_from(entry["path"]) for entry in facts.entry_points)


def _local_files_by_module(facts: InspectionFacts) -> dict[str, set[str]]:
    # A source file can have both its full module name and a src-root-stripped
    # name. Collisions are resolved per import by _prefer_shared_packaging_root.
    files_by_module: dict[str, set[str]] = {}
    for file_path in facts.file_imports:
        for module_name in _module_names_for_file(file_path):
            files_by_module.setdefault(module_name, set()).add(file_path)
    return files_by_module


def _packaging_root_of(file_path: str) -> str:
    parts = Path(file_path).parts
    if parts and parts[0] in _PACKAGE_ROOT_DIR_NAMES:
        return parts[0]
    return ""


def _prefer_shared_packaging_root(files: set[str], importing_file: str) -> set[str]:
    if len(files) <= 1:
        return files
    importing_root = _packaging_root_of(importing_file)
    same_root = {file_path for file_path in files if _packaging_root_of(file_path) == importing_root}
    return same_root or files


def _local_files_for_import(
    import_name: str,
    importing_file: str,
    local_files_by_module: dict[str, set[str]],
) -> set[str]:
    files = set()
    exact_candidates = _exact_module_candidates_for_import(import_name, importing_file, local_files_by_module)
    resolved_modules = set()
    for module_name in exact_candidates:
        module_files = local_files_by_module.get(module_name, set())
        if module_files:
            resolved_modules.add(module_name)
            files.update(module_files)
    if not files:
        return files
    for module_name in _package_module_prefix_candidates_for_resolved(resolved_modules, exact_candidates):
        files.update(
            file_path
            for file_path in local_files_by_module.get(module_name, set())
            if _is_package_module_file(file_path)
        )
    return _prefer_shared_packaging_root(files, importing_file)


def _exact_module_candidates_for_import(
    import_name: str,
    importing_file: str,
    local_files_by_module: dict[str, set[str]],
) -> set[str]:
    candidates = {import_name} if import_name else set()
    context_prefix = _import_context_prefix(importing_file)
    if context_prefix:
        for module_name in list(candidates):
            prefixed = f"{context_prefix}.{module_name}"
            if _is_single_segment_import(module_name) or prefixed in local_files_by_module:
                candidates.add(prefixed)
    return candidates


def _package_module_prefix_candidates_for_resolved(
    resolved_modules: set[str],
    exact_candidates: set[str],
) -> set[str]:
    candidates: set[str] = set()
    for module_name in resolved_modules:
        candidates.update(_module_name_prefixes(module_name))
    candidates.difference_update(exact_candidates)
    return candidates


def _is_single_segment_import(import_name: str) -> bool:
    return bool(import_name) and "." not in import_name


def _module_name_prefixes(module_name: str) -> set[str]:
    parts = [part for part in module_name.split(".") if part]
    return {".".join(parts[:index]) for index in range(1, len(parts) + 1)}


def _is_package_module_file(file_path: str) -> bool:
    return Path(file_path).name == "__init__.py"


def _file_module_parts(file_path: str) -> Optional[tuple[str, ...]]:
    if not file_path.endswith(".py"):
        return None
    path = Path(file_path)
    parts = path.parent.parts if path.name == "__init__.py" else path.with_suffix("").parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return parts


def _module_names_for_file(file_path: str) -> set[str]:
    parts = _file_module_parts(file_path)
    if not parts:
        return set()
    names = {".".join(parts)}
    if len(parts) > 1 and parts[0] in _PACKAGE_ROOT_DIR_NAMES:
        names.add(".".join(parts[1:]))
    return names


def _import_context_prefix(file_path: str) -> str:
    if not file_path.endswith(".py"):
        return ""
    parts = Path(file_path).parent.parts
    if any(part in {"", ".", ".."} for part in parts):
        return ""
    return ".".join(parts)


def _resolve_import_from_module(importing_file: str, module: str, level: int) -> str:
    if level <= 0:
        return module
    package_parts = Path(importing_file).parent.parts
    keep = max(0, len(package_parts) - level + 1)
    parts = list(package_parts[:keep])
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part)
