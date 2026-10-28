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

"""Explicit test-only faults in the real CJ publication/receipt path.

Install only for a scheduled owned fixture via its PYTHONPATH, with a trace
directory and selected site. Do not install globally or into a startup kit.
"""

import hashlib
import json
import os
import time
from pathlib import Path

_root = os.environ.get("NVFLARE_PROCESS_GPU_DELAY_ROOT")
_site = os.environ.get("NVFLARE_PROCESS_GPU_DELAY_SITE")

if _root and _site:
    from nvflare.apis.fl_constant import FLContextKey
    from nvflare.private.fed.client.client_runner import (
        _TASK_CHECK_RESULT_OK,
        _TASK_CHECK_RESULT_TRY_AGAIN,
        ClientRunner,
    )

    _send = ClientRunner._send_task_result
    _try_send = ClientRunner._try_send_result_once

    def _identity(task_id, context):
        return [
            context.get_job_id(),
            context.get_identity_name(),
            task_id,
            context.get_prop(FLContextKey.TASK_ATTEMPT_ID),
        ]

    def _record(event, identity, **facts):
        root = Path(_root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        record = {"event": event, "identity": identity, "pid": os.getpid(), "time": time.time(), **facts}
        with (root / f"{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def _delayed_send(self, result, task_id, context):
        identity = _identity(task_id, context)
        if context.get_identity_name() == _site and not getattr(self, "_qualification_paused", False):
            self._qualification_paused = True
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("qualification CJ must have empty CUDA visibility")
            token = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            marker = Path(_root) / ("pause-" + token + ".json")
            resume = Path(_root) / ("continue-" + token)
            record = _record("cj_publication_blocked", identity, resume=str(resume))
            pending = marker.with_suffix(".pending")
            pending.write_text(json.dumps(record, sort_keys=True))
            pending.replace(marker)
            deadline = time.monotonic() + 180
            while not resume.exists():
                if self.run_abort_signal.triggered:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("scheduled CJ publication was not released within 180 seconds")
                time.sleep(0.05)
            _record("cj_publication_resumed", identity)
        success = _send(self, result, task_id, context)
        if context.get_identity_name() == _site:
            _record(
                "cj_publication_finished",
                identity,
                success=success,
                receipt=context.get_prop(FLContextKey.TASK_RESULT_RECEIPT),
            )
        return success

    def _lost_receipt_once(self, result, task_id, context):
        response = _try_send(self, result, task_id, context)
        if (
            context.get_identity_name() == _site
            and os.environ.get("NVFLARE_PROCESS_GPU_RETRY_ONCE") == "1"
            and not getattr(self, "_qualification_receipt_lost", False)
            and response == _TASK_CHECK_RESULT_OK
        ):
            self._qualification_receipt_lost = True
            # The server already owns the result. Drop only this observation;
            # the next real readiness/receipt exchange must recover it.
            context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, None, private=True, sticky=False)
            _record("receipt_observation_dropped", _identity(task_id, context))
            return _TASK_CHECK_RESULT_TRY_AGAIN
        return response

    ClientRunner._send_task_result = _delayed_send
    ClientRunner._try_send_result_once = _lost_receipt_once
