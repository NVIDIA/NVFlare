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

"""Trusted worker wiring for the shared Client API, progress and analytics."""

import copy
import json
import math
import os
import time
from functools import partial

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.executors.client_api.backend_spec import CLIENT_API_BACKEND_FACTORY
from nvflare.app_common.executors.client_api.task_in_process_backend import TaskInProcessBackend
from nvflare.app_common.executors.client_api_executor import FED_ANALYTIC_EVENT_TYPE
from nvflare.private.fed.task_worker.artifacts import require_eager_shareable
from nvflare.private.fed.task_worker.protocol import _atomic_publish, _open_regular

CLIENT_TASK_CONTEXT_KEYS = {CLIENT_API_BACKEND_FACTORY, FLContextKey.CLIENT_NAME, FLContextKey.PROCESS_TYPE}


def _mark_progress(store, identity, name):
    path = os.path.join(store.attempt_dir(identity), name)
    record = {"identity": identity.to_dict(), "monotonic": time.monotonic()}
    _atomic_publish(path, json.dumps(record, sort_keys=True).encode("utf-8"))


def _read_progress(store, identity, name):
    try:
        with _open_regular(os.path.join(store.attempt_dir(identity), name)) as stream:
            data = stream.read(65537)
    except FileNotFoundError:
        return None
    if len(data) > 65536:
        raise ValueError("Client API progress marker is too large")
    record = json.loads(data)
    if set(record) != {"identity", "monotonic"} or record["identity"] != identity.to_dict():
        raise ValueError("Client API progress marker has an invalid identity")
    value = record["monotonic"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Client API progress marker has an invalid timestamp")
    return value


def result_wait_deadline(store, identity, timeout, finalization_timeout=30.0):
    """Bound the complete result wait, including script exit and finalizers."""
    started = _read_progress(store, identity, "result_wait_started.json")
    sent = _read_progress(store, identity, "script_result_sent.json")
    deadlines = []
    if started is not None and timeout is not None:
        deadlines.append(started + timeout)
    if sent is not None:
        deadlines.append(sent + finalization_timeout)
    return min(deadlines) if deadlines else None


def _mark_result(store, identity, shareable):
    require_eager_shareable(shareable)
    _mark_progress(store, identity, "script_result_sent.json")


def bind_client_task_context(fl_ctx, identity, store, analytics):
    fl_ctx.set_prop(FLContextKey.CLIENT_NAME, identity.site_name, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.PROCESS_TYPE, "client_task_worker", private=True, sticky=True)
    fl_ctx.set_prop(
        CLIENT_API_BACKEND_FACTORY,
        partial(
            TaskInProcessBackend,
            on_started=partial(_mark_progress, store, identity, "result_wait_started.json"),
            on_result=partial(_mark_result, store, identity),
        ),
        private=True,
        sticky=True,
    )
    fl_ctx.get_engine().add_event_observer(partial(_capture_analytics, analytics))


def _capture_analytics(records, event_type, fl_ctx):
    if event_type not in (ANALYTIC_EVENT_TYPE, FED_ANALYTIC_EVENT_TYPE):
        return False
    data = fl_ctx.get_prop(FLContextKey.EVENT_DATA)
    if not isinstance(data, Shareable):
        return False
    try:
        dxo = from_shareable(data)
    except (ValueError, TypeError):
        return False
    if dxo.data_kind != DataKind.ANALYTIC:
        return False
    records.append(
        {
            "event_type": event_type,
            "data": copy.deepcopy(data),
            "federated": fl_ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION,
        }
    )
    return True
