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

"""Strict sequential recording and replay for deterministic LLM parity tests."""

import hashlib
import json
from pathlib import Path


def _canonical_request(request: dict) -> str:
    return json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class LLMReplayTape:
    def __init__(self, path: Path, mode: str):
        if mode not in {"record", "replay"}:
            raise ValueError("LLM replay mode must be record or replay")
        self.path = Path(path)
        self.mode = mode
        self.index = 0
        self.entries = []
        if mode == "record":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        else:
            with self.path.open("r", encoding="utf-8") as stream:
                self.entries = [json.loads(line) for line in stream if line.strip()]

    @staticmethod
    def request_hash(request: dict) -> str:
        return hashlib.sha256(_canonical_request(request).encode("utf-8")).hexdigest()

    def record(self, request: dict, response: str) -> None:
        if self.mode != "record":
            raise RuntimeError("Cannot record to a replay-mode tape")
        entry = {
            "index": self.index,
            "request_sha256": self.request_hash(request),
            "request": request,
            "response": response,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        self.index += 1

    def replay(self, request: dict) -> str:
        if self.mode != "replay":
            raise RuntimeError("Cannot replay a record-mode tape")
        if self.index >= len(self.entries):
            raise RuntimeError(f"LLM replay exhausted after {self.index} requests")
        entry = self.entries[self.index]
        actual_hash = self.request_hash(request)
        expected_hash = entry.get("request_sha256")
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"LLM replay request mismatch at index {self.index}: expected {expected_hash}, got {actual_hash}"
            )
        self.index += 1
        return str(entry["response"])

    def assert_consumed(self) -> None:
        if self.mode == "replay" and self.index != len(self.entries):
            raise RuntimeError(f"LLM replay left {len(self.entries) - self.index} unconsumed responses")
