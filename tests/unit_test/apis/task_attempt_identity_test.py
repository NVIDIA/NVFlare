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

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.wf_comm_spec import WFCommSpec


@pytest.mark.parametrize("placement", ["header", "cookie", "both"])
def test_task_attempt_identity_can_be_echoed_by_existing_cookie_jar_or_header(placement):
    data = Shareable()
    if placement in ("header", "both"):
        data.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "authority-attempt")
    if placement in ("cookie", "both"):
        data.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "authority-attempt")
    assert data.get_task_attempt_id() == "authority-attempt"


def test_conflicting_attempt_header_and_cookie_are_rejected():
    data = Shareable()
    data.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "authority-attempt")
    data.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "different-attempt")
    with pytest.raises(ValueError, match="conflicting"):
        data.get_task_attempt_id()


@pytest.mark.parametrize("jar", [[], "cookie", 123, True])
def test_task_attempt_identity_rejects_malformed_cookie_jar(jar):
    data = Shareable()
    data.set_cookie_jar(jar)
    with pytest.raises(ValueError, match="cookie jar"):
        data.get_task_attempt_id()


@pytest.mark.parametrize("attempt_id", ["", " ", "nul\x00attempt", 1, False, {}, []])
def test_attempt_identity_rejects_invalid_values(attempt_id):
    data = Shareable()
    data.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, attempt_id)
    with pytest.raises(ValueError, match="non-empty string"):
        data.get_task_attempt_id()


@pytest.mark.parametrize("placement", ["header", "cookie"])
@pytest.mark.parametrize("required", [True, "true", 1, ""])
def test_required_assignment_cannot_downgrade_to_unfenced_legacy(placement, required):
    data = Shareable()
    if placement == "header":
        data.set_header(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, required)
    else:
        data.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, required)
    with pytest.raises(ValueError):
        data.get_task_attempt_id()


def test_unfenced_legacy_protocol_remains_supported_but_not_assumed_for_fenced_backends():
    data = Shareable()
    assert data.get_task_attempt_id() is None
    communicator = WFCommSpec()
    fl_ctx = FLContext()
    assert communicator.check_submission(None, "legacy", "id", data, fl_ctx)
    assert communicator.claim_submission(None, "legacy", "id", data, fl_ctx)
    data.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "authority-attempt")
    assert not communicator.check_submission(None, "fenced", "id", data, fl_ctx)
    assert not communicator.claim_submission(None, "fenced", "id", data, fl_ctx)
