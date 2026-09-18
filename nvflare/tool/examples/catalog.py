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

import json
import re
import unicodedata
from pathlib import Path, PurePosixPath

_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
PROVENANCE_FILE = ".nvflare-example.json"
_PROVENANCE_KEY = unicodedata.normalize("NFC", PROVENANCE_FILE.casefold())


class _CatalogObject(dict):
    def __init__(self, pairs):
        super().__init__()
        self.duplicate_keys = []
        for key, value in pairs:
            if key in self:
                self.duplicate_keys.append(key)
            else:
                self[key] = value


def load_catalog(path=None):
    path = Path(path or Path(__file__).with_name("catalog.json"))
    definitions = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_CatalogObject)
    if not isinstance(definitions, _CatalogObject) or not definitions:
        raise ValueError("the example catalog must be a non-empty object")
    if definitions.duplicate_keys:
        raise ValueError(f"duplicate catalog short name: {definitions.duplicate_keys[0]}")

    catalog = {}
    source_paths = set()
    for name, entry in definitions.items():
        error = None
        if not isinstance(name, str) or not _NAME.fullmatch(name) or not isinstance(entry, _CatalogObject):
            error = "must map a lowercase short name to a catalog entry"
        elif entry.duplicate_keys:
            error = f"contains duplicate key {entry.duplicate_keys[0]}"
        elif not {"category", "source_path"} <= set(entry) or set(entry) - {
            "category",
            "dependencies",
            "source_path",
            "destination_path",
        }:
            error = "must contain category and source_path, with optional dependencies and destination_path"
        elif not isinstance(entry["category"], str) or not _NAME.fullmatch(entry["category"]):
            error = "category must be a lowercase name"
        if not error:
            source_path = entry["source_path"]
            path_value = PurePosixPath(source_path) if isinstance(source_path, str) else None
            normalized_path = path_value.as_posix() if path_value else None
            parts = path_value.parts if path_value else ()
            if (
                len(parts) < 2
                or parts[0] != "examples"
                or "\x00" in source_path
                or any(part in {"", ".", ".."} for part in parts)
                or normalized_path != source_path
            ):
                error = "source_path must be a normalized path below examples/"
            elif source_path in source_paths:
                error = f"duplicates source_path {source_path}"
        if not error and "destination_path" in entry:
            destination_path = entry["destination_path"]
            path_value = PurePosixPath(destination_path) if isinstance(destination_path, str) else None
            normalized_path = path_value.as_posix() if path_value else None
            parts = path_value.parts if path_value else ()
            if (
                not parts
                or path_value.is_absolute()
                or "\x00" in destination_path
                or any(part in {"", ".", ".."} for part in parts)
                or normalized_path != destination_path
            ):
                error = "destination_path must be a normalized relative path"
            elif unicodedata.normalize("NFC", parts[0].casefold()) == _PROVENANCE_KEY:
                error = f"destination_path cannot use the reserved name {PROVENANCE_FILE}"
        if not error and "dependencies" in entry:
            dependencies = entry["dependencies"]
            if (
                not isinstance(dependencies, list)
                or not dependencies
                or any(
                    not isinstance(dependency, str) or not _NAME.fullmatch(dependency) for dependency in dependencies
                )
                or len(dependencies) != len(set(dependencies))
            ):
                error = "dependencies must be a non-empty list of unique catalog short names"
        if error:
            raise ValueError(f"invalid catalog entry {name!r}: {error}")
        source_paths.add(source_path)
        catalog[name] = dict(entry)

    resolved_dependencies = {}

    def resolve_dependencies(name, visiting):
        if name in resolved_dependencies:
            return resolved_dependencies[name]
        if name in visiting:
            raise ValueError(f"catalog dependency cycle includes {name!r}")
        visiting.add(name)
        resolved = {name}
        for dependency in catalog[name].get("dependencies", []):
            if dependency not in catalog:
                raise ValueError(f"catalog entry {name!r} depends on unknown example {dependency!r}")
            resolved.update(resolve_dependencies(dependency, visiting))
        visiting.remove(name)
        resolved_dependencies[name] = resolved
        return resolved

    for name in catalog:
        components = resolve_dependencies(name, set())
        if len(components) == 1:
            continue
        for component in components:
            if "destination_path" in catalog[component]:
                raise ValueError(
                    f"catalog dependency group for {name!r} cannot include destination_path on {component!r}"
                )
        component_paths = {
            component: tuple(
                unicodedata.normalize("NFC", part.casefold())
                for part in PurePosixPath(catalog[component]["source_path"]).parts
            )
            for component in components
        }
        for component, parts in component_paths.items():
            for other, other_parts in component_paths.items():
                if component != other and len(parts) <= len(other_parts) and other_parts[: len(parts)] == parts:
                    raise ValueError(
                        f"catalog dependency group for {name!r} contains overlapping source paths for "
                        f"{component!r} and {other!r}"
                    )
    return catalog
