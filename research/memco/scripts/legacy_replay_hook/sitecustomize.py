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

"""Opt-in OpenAI recorder for an unmodified legacy MemCo process."""

import fcntl
import hashlib
import json
import os
from pathlib import Path


def _canonical_request(request: dict) -> str:
    return json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _append(path: Path, request: dict, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        index = sum(1 for line in stream if line.strip())
        entry = {
            "index": index,
            "request_sha256": hashlib.sha256(_canonical_request(request).encode("utf-8")).hexdigest(),
            "request": request,
            "response": response,
        }
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _install() -> None:
    path_value = str(os.environ.get("MEMCO_LEGACY_RECORD_PATH", "") or "").strip()
    if not path_value:
        return
    from openai.resources.chat.completions import Completions

    original = Completions.create

    def recorded_create(self, *args, **kwargs):
        if args:
            raise RuntimeError("Legacy replay recorder requires keyword OpenAI arguments")
        result = original(self, **kwargs)
        answer = result.choices[0].message.content
        if answer is not None:
            _append(Path(path_value), kwargs, str(answer).strip())
        return result

    Completions.create = recorded_create


_install()
