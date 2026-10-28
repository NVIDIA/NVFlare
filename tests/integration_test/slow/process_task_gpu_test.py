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

"""Real-CUDA visibility, cleanup and reuse on explicitly assigned devices.

NVFLARE_PROCESS_GPU_DEVICES names a JSON file with the approved gpu_devices list.
The fixture does not reserve GPUs; schedule it only on administrator-assigned idle devices.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from tests.integration_test.tools.process_gpu.cuda_probe import launch_probe as _launch
from tests.integration_test.tools.process_gpu.cuda_probe import wait_for_probe as _ready

pytestmark = pytest.mark.skipif(
    not os.environ.get("NVFLARE_PROCESS_GPU_DEVICES"), reason="requires explicitly scheduled real GPU capacity"
)


@pytest.fixture
def gpu_launcher():
    config = json.loads(Path(os.environ["NVFLARE_PROCESS_GPU_DEVICES"]).read_text())
    return ProcessTaskLauncher(gpu_devices=config["gpu_devices"])


def _no_live_pid(pid):
    try:
        assert psutil.Process(pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        pass


@pytest.mark.timeout(120)
def test_real_cuda_reuse_while_test_publisher_holds_previous_result(gpu_launcher, tmp_path):
    # Process-scope test: this thread is a test publisher, not a CJ/server upload.
    first = _launch(gpu_launcher, tmp_path / "first.json")
    try:
        assert first.wait_for_settlement(60).succeeded
        first_facts = _ready(tmp_path / "first.json")
        saved = (tmp_path / "first.pt").read_bytes()
        publishing, finish_publication = threading.Event(), threading.Event()

        def publish_saved_result():
            publishing.set()
            assert finish_publication.wait(60)
            assert (tmp_path / "first.pt").read_bytes() == saved
            return time.time()

        with ThreadPoolExecutor(max_workers=1) as executor:
            upload = executor.submit(publish_saved_result)
            assert publishing.wait(5)
            second = None
            try:
                second = _launch(gpu_launcher, tmp_path / "second.json")
                assert second.wait_for_settlement(60).succeeded
                assert not upload.done()
                second_facts = _ready(tmp_path / "second.json")
                assert second_facts["visible"] == first_facts["visible"]
            finally:
                finish_publication.set()
                if second:
                    second.cancel()
            published_at = upload.result(5)
        assert second_facts["computed_at"] < published_at
        # The first immutable result remains reusable after another execution.
        assert (tmp_path / "first.pt").read_bytes() == saved
    finally:
        first.cancel()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("timeout_first", [False, True], ids=["cancel", "wait-timeout"])
def test_real_cuda_descendant_cleanup_allows_a_fresh_worker(gpu_launcher, tmp_path, timeout_first):
    handle = _launch(gpu_launcher, tmp_path / "held.json", "child")
    try:
        facts = _ready(tmp_path / "held.json")
        if timeout_first:
            with pytest.raises(TimeoutError):
                handle.wait_for_settlement(0.1)
        assert handle.cancel().settled
        _no_live_pid(facts["pid"])
        _no_live_pid(facts["descendant_pid"])
        replacement = _launch(gpu_launcher, tmp_path / "replacement.json")
        try:
            assert replacement.wait_for_settlement(60).succeeded
        finally:
            replacement.cancel()
    finally:
        handle.cancel()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", ["exception", "abrupt", "oom"])
def test_real_cuda_failure_settles_and_a_fresh_worker_succeeds(gpu_launcher, tmp_path, mode):
    handle = _launch(gpu_launcher, tmp_path / "failure.json", mode)
    try:
        status = handle.wait_for_settlement(60)
        assert status.settled and not status.succeeded
        facts = _ready(tmp_path / "failure.json")
        _no_live_pid(facts["pid"])
        if mode == "oom":
            error = json.loads((tmp_path / "failure.failure.json").read_text())
            assert error["error_type"] == "OutOfMemoryError"
        healthy = _launch(gpu_launcher, tmp_path / "after-failure.json")
        try:
            assert healthy.wait_for_settlement(60).succeeded
        finally:
            healthy.cancel()
    finally:
        handle.cancel()
