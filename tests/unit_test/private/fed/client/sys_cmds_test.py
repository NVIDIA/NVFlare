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

import nvflare as _nvflare_mod
from nvflare.private.admin_defs import Message
from nvflare.private.defs import SysCommandTopic
from nvflare.private.fed.client.client_req_processors import ClientRequestProcessors
from nvflare.private.fed.client.scheduler_cmds import ReportVersionProcessor


def test_report_version_processor_topic():
    processor = ReportVersionProcessor()
    assert processor.get_topics() == [SysCommandTopic.REPORT_VERSION]


def test_report_version_processor_returns_local_version(monkeypatch):
    monkeypatch.setattr(_nvflare_mod, "__version__", "2.8.0-test")
    processor = ReportVersionProcessor()

    reply = processor.process(Message(topic=SysCommandTopic.REPORT_VERSION, body=""), app_ctx=None)

    assert reply.topic == f"reply_{SysCommandTopic.REPORT_VERSION}"
    assert json.loads(reply.body) == {"version": "2.8.0-test"}


def test_client_request_processors_register_report_version_processor():
    assert any(isinstance(p, ReportVersionProcessor) for p in ClientRequestProcessors.request_processors)
