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

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.impl.wf_comm_server import WFCommServer, _DeadClientStatus
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.server_engine_spec import ServerEngineSpec


def _make_wf_comm(clients, dead_names, min_sites=1, required_sites=None):
    """Build a WFCommServer with mocked engine, enrolled clients, and pre-declared dead clients."""
    mock_engine = Mock(spec=ServerEngineSpec)
    ctx_mgr = FLContextManager(
        engine=mock_engine,
        identity_name="__mock_server",
        job_id="job_1",
        public_stickers={},
        private_stickers={},
    )
    fl_ctx = ctx_mgr.new_context()
    fl_ctx.set_prop(
        FLContextKey.JOB_META,
        {
            JobMetaKey.MIN_CLIENTS: min_sites,
            JobMetaKey.MANDATORY_CLIENTS: required_sites or [],
        },
    )
    mock_engine.new_context.return_value = fl_ctx
    mock_engine.get_clients.return_value = clients

    wf = WFCommServer()
    wf._engine = mock_engine
    for name in dead_names:
        status = _DeadClientStatus()
        status.disconnect_time = 1.0  # non-None → deemed disconnected
        wf._dead_clients[name] = status
    return wf


class TestJobPolicyViolated:
    def test_alive_below_min_sites_aborts(self):
        """min_sites=2, 2 enrolled, 1 dead → alive=1 < min_sites=2 → abort."""
        clients = [Client("site-1", "tok-1"), Client("site-2", "tok-2")]
        wf = _make_wf_comm(clients, dead_names=["site-1"], min_sites=2)
        assert wf._job_policy_violated() is True

    def test_non_required_dead_client_no_abort(self):
        """min_sites=1, 2 enrolled, 1 non-required dead → alive=1 >= min_sites=1, not required → no abort."""
        clients = [Client("site-1", "tok-1"), Client("site-2", "tok-2")]
        wf = _make_wf_comm(clients, dead_names=["site-1"], min_sites=1, required_sites=[])
        assert wf._job_policy_violated() is False

    def test_required_dead_client_aborts(self):
        """min_sites=1, 2 enrolled, 1 required dead → required client is dead → abort."""
        clients = [Client("site-1", "tok-1"), Client("site-2", "tok-2")]
        wf = _make_wf_comm(clients, dead_names=["site-1"], min_sites=1, required_sites=["site-1"])
        assert wf._job_policy_violated() is True
