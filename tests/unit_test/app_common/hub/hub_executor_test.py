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

from nvflare.app_common.hub.hub_executor import HubExecutor


def test_hub_executor_preserves_task_read_wait_fast_fail_without_stream_progress():
    executor = HubExecutor(pipe_id="pipe", task_read_wait_time=10.0)

    assert executor.peer_read_timeout == 10.0
    assert executor.streaming_idle_timeout is None
    assert executor._get_task_send_peer_read_timeout() == 10.0
    assert (
        executor._should_continue_task_send_waiting(
            task_name="train",
            task_id="task-1",
            job_id=None,
            send_start_time=0.0,
            fl_ctx=MagicMock(),
        )
        is False
    )
