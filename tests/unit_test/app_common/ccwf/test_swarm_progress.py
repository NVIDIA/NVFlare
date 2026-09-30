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

from unittest.mock import MagicMock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter


def test_swarm_progress_is_owned_by_the_selected_aggregation_client():
    aggregation_context = FLContext()
    aggregation_context.get_identity_name = lambda: "site-1"
    aggregation_context.set_prop(AppConstants.PROGRESS_OWNER, "site-1", private=True, sticky=False)
    aggregation_context.set_prop(AppConstants.PROGRESS_TITLE, "Swarm learning", private=True, sticky=False)
    other_context = FLContext()
    other_context.get_identity_name = lambda: "site-2"
    other_context.set_prop(AppConstants.PROGRESS_OWNER, "site-1", private=True, sticky=False)

    assert MetricsArtifactWriter._is_progress_owner(aggregation_context)
    assert not MetricsArtifactWriter._is_progress_owner(other_context)


def test_rotating_swarm_aggregator_keeps_global_round_number(caplog):
    writer = MetricsArtifactWriter()
    fl_ctx = FLContext()
    fl_ctx.get_identity_name = lambda: "site-2"
    fl_ctx.set_prop(AppConstants.PROGRESS_OWNER, "site-2", private=True, sticky=False)
    fl_ctx.set_prop(AppConstants.PROGRESS_TITLE, "Swarm learning", private=True, sticky=False)
    fl_ctx.set_prop(AppConstants.START_ROUND, 0, private=True, sticky=False)
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1, private=True, sticky=False)
    fl_ctx.set_prop(AppConstants.NUM_ROUNDS, 3, private=True, sticky=False)

    with caplog.at_level("INFO"):
        writer.handle_event(EventType.START_RUN, fl_ctx)
        writer.handle_event(AppEventType.ROUND_STARTED, fl_ctx)

    output = "\n".join(record.message for record in caplog.records)
    assert "ROUND 2 / 3" in output
    assert "Swarm learning" in output


def test_swarm_client_publishes_selected_progress_owner():
    class _RoundStarted(Exception):
        pass

    controller = SwarmClientController.__new__(SwarmClientController)
    controller.me = "site-1"
    controller.update_status = MagicMock()
    controller.log_error = MagicMock()
    controller.log_info = MagicMock()
    controller.log_debug = MagicMock()
    controller._stamp_result_upload_receiver_ids = MagicMock()
    controller._prepare_learn_task_data = MagicMock(side_effect=lambda data, _ctx: (data, data))
    controller.shareable_generator = MagicMock()
    controller.shareable_generator.shareable_to_learnable.return_value = Learnable()

    def get_config(key, default=None):
        return {
            Constant.START_ROUND: 0,
            AppConstants.NUM_ROUNDS: 3,
        }.get(key, default)

    controller.get_config_prop = MagicMock(side_effect=get_config)
    fl_ctx = FLContext()
    task_data = Shareable()
    task_data.set_header(AppConstants.CURRENT_ROUND, 1)
    task_data.set_header(Constant.AGGREGATOR, "site-2")

    def inspect_round_started(event_type, event_ctx):
        assert event_type == AppEventType.ROUND_STARTED
        assert event_ctx.get_prop(AppConstants.PROGRESS_OWNER) == "site-2"
        assert event_ctx.get_prop(AppConstants.CURRENT_ROUND) == 1
        assert event_ctx.get_prop(AppConstants.NUM_ROUNDS) == 3
        assert event_ctx.get_prop(AppConstants.PROGRESS_TITLE) == "Swarm learning"
        raise _RoundStarted

    controller.fire_event = MagicMock(side_effect=inspect_round_started)

    with pytest.raises(_RoundStarted):
        controller.do_learn_task("train", task_data, fl_ctx, Signal())
