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


def load_catalog(path=None):
    path = Path(path or Path(__file__).with_name("catalog.json"))
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("the example catalog must be a non-empty object")

    source_paths = set()
    for name, entry in catalog.items():
        if not isinstance(name, str) or not _SHORT_NAME.fullmatch(name) or not isinstance(entry, dict):
            raise ValueError("each example must map a short name to one source_path")
        if set(entry) != {"source_path"}:
            raise ValueError("each example must map a short name to one source_path")
        source_path = entry["source_path"]
        parts = PurePosixPath(source_path).parts if isinstance(source_path, str) else ()
        if not parts or parts[0] != "examples" or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"invalid source path for {name}")
        if source_path in source_paths:
            raise ValueError(f"duplicate example source path: {source_path}")
        source_paths.add(source_path)
    return catalog
