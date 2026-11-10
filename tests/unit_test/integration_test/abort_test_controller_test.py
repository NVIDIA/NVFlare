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

from unittest.mock import Mock

import pytest

from nvflare.apis.signal import Signal
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from tests.integration_test.data.apps.abort_test_job.app.custom.abort_test_controller import AbortTestController


def test_job_stays_active_until_explicit_release(monkeypatch):
    controller = AbortTestController()
    engine = Mock()
    collector = GroupInfoCollector()
    fl_ctx = Mock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_prop.return_value = collector
    controller.initialize(fl_ctx)
    topic, release = engine.register_app_command.call_args.args
    assert topic == "release_abort_test"
    waits = []

    def wait(interval):
        waits.append(interval)
        if len(waits) > 1000:
            pytest.fail("release command did not open the gate")
        controller.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, fl_ctx)
        assert collector.info[controller.name]["phase"] == "waiting_for_release"
        if len(waits) == 1000:
            assert release(topic, None, fl_ctx) == {"released": True}
        return controller._released.is_set()

    monkeypatch.setattr(controller._released, "wait", wait)

    controller.control_flow(Signal(), fl_ctx)

    assert len(waits) == 1000
    controller.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, fl_ctx)
    assert collector.info[controller.name]["phase"] == "released"


@pytest.mark.parametrize("abort_before_entry", [False, True])
def test_job_exits_on_abort_without_releasing_gate(abort_before_entry, monkeypatch):
    controller = AbortTestController()
    abort_signal = Signal()
    if abort_before_entry:
        abort_signal.trigger(True)

    def wait(interval):
        assert wait_mock.call_count == 1, "controller continued waiting after abort"
        abort_signal.trigger(True)
        return False

    wait_mock = Mock(side_effect=wait)
    monkeypatch.setattr(controller._released, "wait", wait_mock)

    controller.control_flow(abort_signal, Mock())

    assert controller._released.is_set() is False
    assert wait_mock.call_count == (0 if abort_before_entry else 1)
