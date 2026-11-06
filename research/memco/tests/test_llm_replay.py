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

import sys
import tempfile
import unittest
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent.parent / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from mas.llm_replay import LLMReplayTape  # noqa: E402


class LLMReplayTapeTest(unittest.TestCase):
    def test_record_and_replay_exact_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            request = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "prompt"}]}
            recorder = LLMReplayTape(path, "record")
            recorder.record(request, "answer")

            replay = LLMReplayTape(path, "replay")

            self.assertEqual("answer", replay.replay(request))

    def test_replay_rejects_changed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            recorder = LLMReplayTape(path, "record")
            recorder.record({"messages": [{"content": "original"}]}, "answer")
            replay = LLMReplayTape(path, "replay")

            with self.assertRaisesRegex(RuntimeError, "request mismatch at index 0"):
                replay.replay({"messages": [{"content": "changed"}]})

    def test_replay_rejects_exhausted_tape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            LLMReplayTape(path, "record")
            replay = LLMReplayTape(path, "replay")

            with self.assertRaisesRegex(RuntimeError, "replay exhausted"):
                replay.replay({"messages": []})

    def test_replay_rejects_unconsumed_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            recorder = LLMReplayTape(path, "record")
            recorder.record({"messages": []}, "answer")
            replay = LLMReplayTape(path, "replay")

            with self.assertRaisesRegex(RuntimeError, "left 1 unconsumed"):
                replay.assert_consumed()


if __name__ == "__main__":
    unittest.main()
