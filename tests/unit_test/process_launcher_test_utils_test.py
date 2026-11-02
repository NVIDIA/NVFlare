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

"""Readiness checks shared by real-process integration scenarios."""

from tests import process_launcher_test_utils
from tests.timing_utils import isolate_time


def test_wait_for_file_does_not_accept_an_empty_pid_file(tmp_path, monkeypatch):
    pid_file = tmp_path / "worker.pid"
    pid_file.touch()
    local_time = isolate_time(monkeypatch, process_launcher_test_utils)
    sleeps = []

    def finish_pid_write(interval):
        sleeps.append(interval)
        pid_file.write_text("12345")

    monkeypatch.setattr(local_time, "sleep", finish_pid_write)
    process_launcher_test_utils._wait_for_file(pid_file)

    assert sleeps == [0.01]
    assert int(pid_file.read_text()) == 12345
