# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.task_exchanger import TaskExchanger


class HubExecutor(TaskExchanger):
    """
    This executor is to be used by Tier-1 (T1) clients.
    It exchanges task data/result with the Hub Controller of Tier-2 (T2) Server
    """

    def __init__(
        self,
        pipe_id: str,
        task_wait_time=None,
        result_poll_interval: float = 0.1,
        task_read_wait_time: float = 10.0,
        read_interval: float = 2.0,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        resend_interval: float = 2.0,
        max_resends=None,
    ):
        """Constructor of HubExecutor.

        Args:
            pipe_id (str): component id of pipe
            task_wait_time: how long to wait for result from T2. None means waits forever.
            result_poll_interval: polling interval for T2 result
            task_read_wait_time: how long to wait for T2 to read a task assignment
            read_interval: how often to read from pipe
            heartbeat_interval: how often to send heartbeat to peer
            heartbeat_timeout: max amount of time to allow missing heartbeats before treating peer as dead
            resend_interval: how often to resend a message when failing to send
            max_resends: max number of resends. None means no limit
        """
        super().__init__(
            pipe_id=pipe_id,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            resend_interval=resend_interval,
            max_resends=max_resends,
            peer_read_timeout=task_read_wait_time,
            task_wait_time=task_wait_time,
            result_poll_interval=result_poll_interval,
        )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        contrib_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        if contrib_round is None:
            self.log_warning(fl_ctx, "CONTRIBUTION_ROUND Not Set in task data!")

        return super().execute(task_name, shareable, fl_ctx, abort_signal)
