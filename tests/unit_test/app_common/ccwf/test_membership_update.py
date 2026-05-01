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

"""Tests for current client membership-update behavior.

Membership prune handling was removed from SwarmClientController, so there is
no membership update handler and no registration for the membership topic in
process_config().
"""

from unittest.mock import MagicMock

from nvflare.app_common.ccwf.common import Constant, topic_for_membership_update
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController


class TestMembershipUpdateBehaviorRemoved:
    def test_handler_method_not_present(self):
        """SwarmClientController no longer provides _handle_membership_update."""
        ctl = SwarmClientController.__new__(SwarmClientController)
        assert not hasattr(ctl, "_handle_membership_update")

    def test_topic_not_registered_in_process_config(self):
        """process_config() should not register a membership-update topic handler."""
        ctl = SwarmClientController.__new__(SwarmClientController)
        ctl.logger = MagicMock()
        ctl.log_info = MagicMock()
        ctl.log_warning = MagicMock()
        ctl.log_error = MagicMock()
        ctl.me = "site-1"
        ctl.workflow_id = "wf-42"
        ctl.request_to_submit_learn_result_task_name = "req_submit"
        ctl.topic_for_my_workflow = lambda base: f"{base}.wf-42"

        registered_topics = {}
        engine = MagicMock()
        engine.register_aux_message_handler.side_effect = lambda topic, message_handle_func: registered_topics.update(
            {topic: message_handle_func}
        )
        ctl.engine = engine

        def cfg(key, *default):
            return {
                Constant.CLIENTS: ["site-1", "site-2"],
                Constant.TRAIN_CLIENTS: ["site-1", "site-2"],
                Constant.AGGR_CLIENTS: ["site-1", "site-2"],
            }.get(key, default[0] if default else None)

        ctl.get_config_prop = cfg

        fl_ctx = MagicMock()
        ctl.process_config(fl_ctx)

        expected_membership_topic = topic_for_membership_update("wf-42")
        assert expected_membership_topic not in registered_topics
        assert set(registered_topics.keys()) == {"cwf.share_result.wf-42", "req_submit"}
        assert registered_topics["cwf.share_result.wf-42"] == ctl._process_share_result
        assert registered_topics["req_submit"] == ctl._process_submission_request
