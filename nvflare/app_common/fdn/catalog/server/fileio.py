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

"""Atomic file replacement shared by the catalog controller and the server store."""

import json
import os


def write_atomically(path: str, data: dict) -> None:
    """Write JSON to a temporary file, fsync it, then replace the target. Readable by the owner only: the files hold
    a job's key, withheld dataset names or stored catalogs."""
    tmp = f"{path}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        # Context files and stored catalogs may contain private data, even after a partial write.
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
