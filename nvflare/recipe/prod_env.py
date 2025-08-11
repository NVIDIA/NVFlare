# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os.path
import tempfile

from nvflare.fuel.flare_api.flare_api import new_insecure_session
from nvflare.job_config.api import FedJob

from .spec import ExecEnv


class ProdEnv(ExecEnv):

    def __init__(
        self,
        startup_kit_dir: str,
        login_timeout: float = 5.0,
        monitor_job_duration: int = None,
    ):
        self.startup_kit_dir = startup_kit_dir
        self.login_timeout = login_timeout
        self.monitor_job_duration = monitor_job_duration

    def deploy(self, job: FedJob):
        sess = new_insecure_session(startup_kit_location=self.startup_kit_dir, timeout=self.login_timeout)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                job.export_job(temp_dir)
                job_path = os.path.join(temp_dir, job.name)
                job_id = sess.submit_job(job_path)
                print(f"submitted job: {job_id}")

            # monitor job until done.
            if self.monitor_job_duration:
                rc = sess.monitor_job(job_id, timeout=self.monitor_job_duration)
                print(f"job monitor done: {rc=}")

            return job_id
        finally:
            sess.close()
