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
from unittest.mock import MagicMock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.ccwf.client_ctl import ClientSideController, _LearnTask
from nvflare.app_common.ccwf.common import Constant


class _FailingClientSideController(ClientSideController):
    def __init__(self):
        super().__init__(
            task_name_prefix="test",
            learn_task_check_interval=0.01,
            learn_task_ack_timeout=1.0,
            learn_task_abort_timeout=1.0,
            final_result_ack_timeout=1.0,
        )

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        raise NotImplementedError

    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        self.asked_to_stop = True
        raise RuntimeError("learn failed")


class _AbortedClientSideController(_FailingClientSideController):
    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        self.asked_to_stop = True
        abort_signal.trigger(True)
        raise RuntimeError("aborted")


class _BlockingLogClientSideController(_FailingClientSideController):
    def __init__(self):
        super().__init__()
        self.exception_caught = threading.Event()
        self.continue_logging = threading.Event()

    def log_exception(self, fl_ctx: FLContext, msg: str):
        self.exception_caught.set()
        self.continue_logging.wait(timeout=5.0)


def test_do_learn_logs_exception_from_learn_task():
    ctl = _FailingClientSideController()
    ctl.logger = MagicMock()
    ctl.learn_task = _LearnTask("train", Shareable(), FLContext())

    ctl._do_learn()

    # log_exception logs the contextualized message, then the traceback
    assert ctl.logger.error.call_count == 2
    assert "exception from do_learn_task" in ctl.logger.error.call_args_list[0].args[0]
    assert ctl.learn_task is None
    # the failure must be recorded so the next status report tells the server
    # to end the job with an error status instead of FINISHED:COMPLETED
    assert ctl.current_status.error == ReturnCode.EXECUTION_EXCEPTION
    report = ctl._get_status_report()
    assert report is not None and report.error == ReturnCode.EXECUTION_EXCEPTION


def test_do_learn_does_not_record_error_for_aborted_task():
    # an aborted task (e.g. end-of-workflow teardown) is expected to raise;
    # it must not fail an otherwise-completed job
    ctl = _AbortedClientSideController()
    ctl.logger = MagicMock()
    ctl.learn_task = _LearnTask("train", Shareable(), FLContext())

    ctl._do_learn()

    assert ctl.learn_task is None
    assert not ctl.current_status.error


def test_do_learn_records_failure_when_abort_races_exception_logging():
    ctl = _BlockingLogClientSideController()
    ctl.learn_task = _LearnTask("train", Shareable(), FLContext())

    learn_thread = threading.Thread(target=ctl._do_learn)
    learn_thread.start()
    assert ctl.exception_caught.wait(timeout=5.0)

    # The task failed before it was aborted. Simulate end-workflow triggering
    # the abort signal while exception logging has released the GIL.
    ctl.learn_task.abort_signal.trigger(True)
    ctl.continue_logging.set()
    learn_thread.join(timeout=5.0)

    assert not learn_thread.is_alive()
    assert ctl.learn_task is None
    assert ctl.current_status.error == ReturnCode.EXECUTION_EXCEPTION


def test_no_report_after_workflow_done():
    # Once the workflow has ended, the client stops reporting status - the
    # server no longer reliably consumes reports, and a recorded error may be
    # a recoverable condition (e.g. a swarm straggler's dropped submission)
    # that must not fail an otherwise-successful job. A genuine failure is
    # instead delivered on the normal pull path while the workflow is live.
    ctl = _FailingClientSideController()
    ctl.logger = MagicMock()
    ctl.workflow_id = "swarm"
    ctl.workflow_done = True
    ctl.update_status(action="do_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)

    fl_ctx = FLContext()
    ctl.handle_event(EventType.BEFORE_PULL_TASK, fl_ctx)

    assert not fl_ctx.get_prop(Constant.STATUS_REPORTS)


def test_error_report_is_delivered_while_workflow_live():
    # a genuine learn failure recorded while the workflow is still running is
    # reported on the next pull so the job ends with an error status
    ctl = _FailingClientSideController()
    ctl.logger = MagicMock()
    ctl.workflow_id = "swarm"
    ctl.update_status(action="do_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)

    fl_ctx = FLContext()
    ctl.handle_event(EventType.BEFORE_PULL_TASK, fl_ctx)

    reports = fl_ctx.get_prop(Constant.STATUS_REPORTS)
    assert reports and reports["swarm"][Constant.ERROR] == ReturnCode.EXECUTION_EXCEPTION
