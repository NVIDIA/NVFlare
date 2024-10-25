# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from abc import abstractmethod

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class JobHandleSpec:
    @abstractmethod
    def terminate(self):
        """To terminate the job run.

        Returns: the job run return code.

        """
        raise NotImplementedError()

    @abstractmethod
    def poll(self):
        """To get the return code of the job run.

        Returns: return_code

        """
        raise NotImplementedError()

    @abstractmethod
    def wait(self):
        """To wait until the job run complete.

        Returns: returns until the job run complete.

        """
        raise NotImplementedError()


class JobLauncherSpec(FLComponent):
    @abstractmethod
    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        """To launch a job run.

        Args:
            job_meta: job meta data
            fl_ctx: FLContext

        Returns: boolean to indicates the job launch success or fail.

        """
        raise NotImplementedError()
