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

_SHORT_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class _JsonObject(list):
    pass


def _duplicate_key(pairs):
    keys = set()
    for key, value in pairs:
        if key in keys:
            return key
        keys.add(key)
    return None


def load_catalog(path=None):
    path = Path(path or Path(__file__).with_name("catalog.json"))
    definitions = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_JsonObject)
    if not isinstance(definitions, _JsonObject) or not definitions:
        raise ValueError("the example catalog must be a non-empty object")
    duplicate_name = _duplicate_key(definitions)
    if duplicate_name is not None:
        raise ValueError(f"duplicate catalog short name: {duplicate_name}")

    catalog = {}
    errors = []
    source_paths = set()
    for name, entry_pairs in definitions:
        error = None
        entry = None
        if not isinstance(name, str) or not _SHORT_NAME.fullmatch(name) or not isinstance(entry_pairs, _JsonObject):
            error = "must map a lowercase short name to one source_path"
        else:
            duplicate_key = _duplicate_key(entry_pairs)
            if duplicate_key is not None:
                error = f"contains duplicate key {duplicate_key}"
            entry = dict(entry_pairs)
        if not error and set(entry) != {"source_path"}:
            error = "must contain only source_path"
        if not error:
            source_path = entry["source_path"]
            normalized_path = PurePosixPath(source_path).as_posix() if isinstance(source_path, str) else None
            parts = PurePosixPath(source_path).parts if isinstance(source_path, str) else ()
            if (
                not parts
                or parts[0] != "examples"
                or any(part in {"", ".", ".."} for part in parts)
                or normalized_path != source_path
            ):
                error = "source_path must be a normalized path below examples/"
            elif source_path in source_paths:
                error = f"duplicates source_path {source_path}"
        if error:
            errors.append({"name": name, "error": error})
            continue
        source_paths.add(source_path)
        catalog[name] = entry
    return catalog, errors
