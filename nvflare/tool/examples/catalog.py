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
from pathlib import Path, PurePosixPath

_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


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
    errors = []
    source_paths = set()
    for name, entry in definitions.items():
        error = None
        if not isinstance(name, str) or not _NAME.fullmatch(name) or not isinstance(entry, _CatalogObject):
            error = "must map a lowercase short name to a catalog entry"
        elif entry.duplicate_keys:
            error = f"contains duplicate key {entry.duplicate_keys[0]}"
        elif not {"category", "source_path"} <= set(entry) or set(entry) - {
            "category",
            "source_path",
            "destination_path",
        }:
            error = "must contain category and source_path, with optional destination_path"
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
                or any(part in {"", ".", ".."} for part in parts)
                or normalized_path != destination_path
            ):
                error = "destination_path must be a normalized relative path"
        if error:
            errors.append({"name": name, "error": error})
            continue
        source_paths.add(source_path)
        catalog[name] = entry
    return catalog, errors
