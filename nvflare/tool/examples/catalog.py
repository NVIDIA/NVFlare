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
from collections import Counter
from pathlib import Path, PurePosixPath

_SHORT_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class _JsonObject(list):
    pass


def load_catalog(path=None):
    path = Path(path or Path(__file__).with_name("catalog.json"))
    definitions = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_JsonObject)
    if not isinstance(definitions, _JsonObject) or not definitions:
        raise ValueError("the example catalog must be a non-empty object")

    catalog = {}
    errors = []
    source_paths = set()
    duplicate_names = {name for name, count in Counter(name for name, _ in definitions).items() if count > 1}
    reported_duplicate_names = set()
    for name, entry in definitions:
        error = None
        if name in duplicate_names:
            if name not in reported_duplicate_names:
                errors.append({"name": name, "error": "duplicates short name"})
                reported_duplicate_names.add(name)
            continue
        if not isinstance(name, str) or not _SHORT_NAME.fullmatch(name) or not isinstance(entry, _JsonObject):
            error = "must map a lowercase short name to one source_path"
        elif len(entry) != 1 or entry[0][0] != "source_path":
            error = "must contain only source_path"
        else:
            source_path = entry[0][1]
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
        catalog[name] = {"source_path": source_path}
    return catalog, errors
