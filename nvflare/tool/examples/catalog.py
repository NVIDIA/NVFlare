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
import subprocess
from pathlib import Path


def load_catalog(path=None):
    path = Path(path or Path(__file__).with_name("catalog.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def source_files(repository_root, source_path):
    repository_root = Path(repository_root)
    source = repository_root / source_path
    if not (repository_root / ".git").exists():
        return tuple(path.relative_to(source).as_posix() for path in sorted(source.rglob("*")) if path.is_file())

    result = subprocess.run(
        ["git", "ls-files", "-z", "--", source_path],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    paths = result.stdout.decode("utf-8").split("\0")
    return tuple(Path(path).relative_to(source_path).as_posix() for path in paths if path)
