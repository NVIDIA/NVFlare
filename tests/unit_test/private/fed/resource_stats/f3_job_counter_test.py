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

from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
from nvflare.private.fed.resource_stats.f3_job_counter import (
    clear_job_f3_counter,
    get_job_f3_counter,
    start_job_f3_counter,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    clear_job_f3_counter()
    yield
    clear_job_f3_counter()


def test_no_counter_before_start():
    assert get_job_f3_counter() is None


def test_start_creates_a_counter_reachable_from_anywhere():
    started = start_job_f3_counter()

    assert isinstance(started, F3Counter)
    assert get_job_f3_counter() is started


def test_start_is_idempotent_and_preserves_prior_observations():
    first = start_job_f3_counter()
    admission = first.try_begin(F3TrafficClass.TASK_RESULT, "server")
    assert first.complete_remote_accepted(admission, 10)

    second = start_job_f3_counter()

    assert first is second
    assert get_job_f3_counter() is first
    assert second.freeze()["sent_to"] == [{"participant_name": "server", "payload_bytes": "10", "messages": "1"}]


def test_clear_drops_the_reference():
    start_job_f3_counter()

    clear_job_f3_counter()

    assert get_job_f3_counter() is None
