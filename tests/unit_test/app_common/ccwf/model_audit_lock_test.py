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

import threading
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf import swarm_client_ctl
from nvflare.app_common.ccwf.client_controller_executor import ClientControllerExecutor
from nvflare.app_common.ccwf.common import Constant
from nvflare.fuel.sec import task_audit
from nvflare.fuel.sec.audit import AuditService
from tests.timing_utils import WAIT_TIMEOUT, ManualClock, ObservedLock, ThreadGroup
from tests.unit_test.fl_context_helper import make_fl_context


class _Persistor(ModelPersistor):
    def __init__(self):
        super().__init__()
        self.saved = []

    def load_model(self, fl_ctx):
        raise NotImplementedError

    def save_model(self, model, fl_ctx):
        self.saved.append(model)


def _make_operation(boundary, lock, fl_ctx, on_event, monkeypatch):
    if boundary in ("gather", "gather_timeout", "reject_non_trainer"):
        gatherer = swarm_client_ctl.Gatherer(
            task_data=Shareable(),
            fl_ctx=fl_ctx,
            for_round=1,
            executor=Mock(),
            aggregator=Mock(accept=Mock(return_value=True)),
            metric_comparator=Mock(),
            all_clients=["site"],
            trainers=["site"],
            min_responses_required=1,
            wait_time_after_min_resps_received=0,
            timeout=0,
        )
        gatherer.lock = lock
        gatherer.fire_event = on_event
        contribution = Shareable()
        contribution.set_header(AppConstants.CURRENT_ROUND, 1)

        if boundary == "gather_timeout":
            clock = ManualClock(gatherer.start_time)
            monkeypatch.setattr(swarm_client_ctl, "time", clock)
            gatherer.timeout = 1
            clock.advance(2)

            def poll():
                assert gatherer.is_done()

            return poll, EventType.WARNING_LOG_AVAILABLE

        if boundary == "reject_non_trainer":
            gatherer.perm_lock = lock

            def check_permission():
                assert (
                    gatherer.can_accept_submission("non-trainer", contribution, fl_ctx) == ReturnCode.MODEL_UNRECOGNIZED
                )

            return check_permission, EventType.ERROR_LOG_AVAILABLE

        def gather():
            assert gatherer.gather("site", contribution, fl_ctx).get_return_code() == ReturnCode.OK

        return gather, AppEventType.BEFORE_CONTRIBUTION_ACCEPT

    controller = (
        ClientControllerExecutor(["controller"], task_name_prefix="test")
        if boundary == "executor_finalize"
        else swarm_client_ctl.SwarmClientController()
    )
    if boundary != "set_learn_task":
        controller.finalize_lock = lock
        controller.fire_event = on_event

        def finalize():
            controller.finalize(fl_ctx)
            assert controller.workflow_done

        return finalize, Constant.EXECUTOR_FINALIZED

    controller.learn_task_lock = lock
    assert controller.set_learn_task(Shareable(), fl_ctx)
    original_task = controller.learn_task
    new_data = Shareable()

    def on_abort(event_type, fl_ctx):
        if event_type != EventType.ABORT_TASK:
            return
        on_event(event_type, fl_ctx)
        # The real learning loop clears this after responding to the abort signal.
        controller.learn_task = None

    def set_task():
        assert controller.set_learn_task(new_data, fl_ctx)
        assert original_task.abort_signal.triggered
        assert controller.learn_task.task_data is new_data

    controller.fire_event = on_abort
    return set_task, EventType.ABORT_TASK


@pytest.mark.parametrize(
    "boundary",
    ["gather", "gather_timeout", "reject_non_trainer", "controller_finalize", "executor_finalize", "set_learn_task"],
)
def test_model_observations_wait_for_client_workflow_lock_release(monkeypatch, boundary):
    lock = ObservedLock()
    entered = threading.Event()
    release = threading.Event()
    fl_ctx = make_fl_context(run_num="local-job")
    private_model = object()
    persistor = _Persistor()
    broken_ctx = Mock(spec=FLContext)
    broken_ctx.get_job_id.side_effect = RuntimeError("failed to capture job")
    observed = []

    def try_lock():
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        return acquired

    def write(**record):
        observed.append((record["action"], try_lock()))
        return "written"

    def diagnostic():
        observed.append(("diagnostic", try_lock()))

    def on_event(event_type, fl_ctx):
        if event_type != selected_event:
            return
        # Use actual save calls, including a capture failure, from the locked callback.
        persistor.save(private_model, fl_ctx)
        persistor.save(private_model, broken_ctx)
        entered.set()
        release.wait()

    monkeypatch.setattr(AuditService, "the_auditor", Mock(add_event=write))
    monkeypatch.setattr(task_audit, "recording_failed", diagnostic)
    operation, selected_event = _make_operation(boundary, lock, fl_ctx, on_event, monkeypatch)

    with ThreadGroup(release) as workers:
        workers.start(operation)
        assert entered.wait(WAIT_TIMEOUT)
        assert not try_lock()
        assert lock.blocked.is_set()
        assert persistor.saved == [private_model, private_model]
        assert observed == []

    actions = [action for action, unlocked in observed]
    assert "model.save.started" in actions
    assert "model.save.returned" in actions
    assert "diagnostic" in actions
    assert all(unlocked for _, unlocked in observed)
