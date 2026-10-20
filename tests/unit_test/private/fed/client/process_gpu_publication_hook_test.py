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
import runpy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.private.fed.client.client_runner import _TASK_CHECK_RESULT_OK, _TASK_CHECK_RESULT_TRY_AGAIN, ClientRunner


def test_scheduled_publication_hook_delays_once_and_recovers_receipt_observation(tmp_path, monkeypatch):
    def sent(self, result, task_id, context):
        context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
        return True

    def try_sent(self, result, task_id, context):
        sent(self, result, task_id, context)
        return _TASK_CHECK_RESULT_OK

    monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_ROOT", str(tmp_path))
    monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_SITE", "site-1")
    monkeypatch.setenv("NVFLARE_PROCESS_GPU_RETRY_ONCE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(ClientRunner, "_send_task_result", sent)
    monkeypatch.setattr(ClientRunner, "_try_send_result_once", try_sent)
    hook = Path(__file__).resolve().parents[5] / "tests/integration_test/data/process_gpu/cj_hooks/sitecustomize.py"
    runpy.run_path(str(hook))
    context = FLContext()
    for key, value in (
        (ReservedKey.RUN_NUM, "job"),
        (ReservedKey.IDENTITY_NAME, "site-1"),
        (FLContextKey.TASK_ATTEMPT_ID, "attempt"),
    ):
        context.set_prop(key, value, private=True, sticky=True)
    runner = SimpleNamespace(run_abort_signal=Signal())
    with ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(ClientRunner._send_task_result, runner, Shareable(), "task", context)
        deadline = time.monotonic() + 5
        while not list(tmp_path.glob("pause-*.json")):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        marker = json.loads(next(tmp_path.glob("pause-*.json")).read_text())
        assert marker["identity"] == ["job", "site-1", "task", "attempt"]
        assert not publication.done()
        Path(marker["resume"]).touch()
        assert publication.result(5)
    assert ClientRunner._try_send_result_once(runner, Shareable(), "task", context) == _TASK_CHECK_RESULT_TRY_AGAIN
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) is None
    assert ClientRunner._try_send_result_once(runner, Shareable(), "task", context) == _TASK_CHECK_RESULT_OK
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
