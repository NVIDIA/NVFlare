# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_constant import FLContextKey, RunProcessKey
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.simulator.simulator_const import SimulatorConstants


class SimulatorParentClientEngine(ClientEngine):
    def __init__(self, client, client_token, args, rank=0):
        super().__init__(client, client_token, args, rank)
        fl_ctx = self.new_context()
        fl_ctx.set_prop(FLContextKey.SIMULATE_MODE, True, private=True, sticky=True)

        self.client_executor.run_processes[SimulatorConstants.JOB_NAME] = {
            RunProcessKey.LISTEN_PORT: None,
            RunProcessKey.CONNECTION: None,
            RunProcessKey.CHILD_PROCESS: None,
            RunProcessKey.STATUS: ClientStatus.STARTED,
        }

    def get_all_job_ids(self):
        jobs = []
        for job in self.client_executor.run_processes.keys():
            jobs.append(job)
        return jobs

    def abort_app(self, job_id: str) -> str:
        return self.client_executor.run_processes.pop(job_id)
