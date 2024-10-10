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

from nvflare.apis.resource_manager_spec import ResourceManagerSpec


class JobLaunchSpec:
    @abstractmethod
    def launch_job(self,
                   client,
                   startup,
                   job_id,
                   args,
                   app_custom_folder,
                   target: str,
                   scheme: str,
                   timeout=None) -> bool:
        """To launch a job run.

        Args:
            timeout: the job needs to be started within this timeout. Otherwise failed the job launch.
                    None means no timeout limit.

        Returns: boolean to indicates the job launch success or fail.

        """
        raise NotImplemented

    @abstractmethod
    def terminate(self):
        """To terminate the job run.

        Returns: the job run return code.

        """
        raise NotImplemented

    @abstractmethod
    def return_code(self):
        """To get the return code of the job run.

        Returns: return_code

        """
        raise NotImplemented

    @abstractmethod
    def wait(self):
        """To wait until the job run complete.

        Returns: returns until the job run complete.

        """
        raise NotImplemented
