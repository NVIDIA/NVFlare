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

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.hci.base64_utils import b64str_to_str, str_to_b64str
from nvflare.fuel.hci.server.sess import Session


class _FakeIdAsserter:
    cert = "server-cert"

    @staticmethod
    def sign(data, return_str=True):
        assert return_str
        return "signature"


def test_session_token_round_trip_preserves_study():
    session = Session(
        sess_id="session-id",
        user_name="admin@nvidia.com",
        org="nvidia",
        role="lead",
        origin_fqcn="origin",
        active_study="cancer-research",
    )

    token = session.make_token(_FakeIdAsserter())
    restored = Session.decode_token(token)

    assert restored.active_study == "cancer-research"
    assert restored.user_name == "admin@nvidia.com"
    assert restored.user_org == "nvidia"
    assert restored.user_role == "lead"


def test_session_token_uses_study_field_name():
    session = Session(
        sess_id="session-id",
        user_name="admin@nvidia.com",
        org="nvidia",
        role="lead",
        origin_fqcn="origin",
        active_study="cancer-research",
    )

    token = session.make_token(_FakeIdAsserter())
    payload = json.loads(b64str_to_str(token.split(":")[0]))

    assert payload["study"] == "cancer-research"
    assert "t" not in payload


def test_decode_token_defaults_legacy_session_study():
    legacy_payload = json.dumps({"n": "admin@nvidia.com", "r": "lead", "o": "nvidia", "s": "session-id"})
    token = f"{str_to_b64str(legacy_payload)}:signature"

    restored = Session.decode_token(token)

    assert restored.active_study == DEFAULT_STUDY


def test_decode_token_accepts_legacy_t_study_field():
    legacy_payload = json.dumps(
        {"n": "admin@nvidia.com", "r": "lead", "o": "nvidia", "s": "session-id", "t": "legacy-study"}
    )
    token = f"{str_to_b64str(legacy_payload)}:signature"

    restored = Session.decode_token(token)

    assert restored.active_study == "legacy-study"
