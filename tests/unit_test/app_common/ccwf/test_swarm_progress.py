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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
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
