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
from typing import List, Optional

from pydantic import PositiveInt

from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv


class POCEnv(ExecEnv):

    num_clients: Optional[PositiveInt] = None
    clients: Optional[List[str]] = None
    num_threads: Optional[PositiveInt] = (None,)
    log_config: Optional[str] = (None,)

    def deploy(self, job: FedJob):
        if self.clients is None:
            self.num_clients = 2
        # TBD
        # first launch a POC system; if the POC is not yet started
        # then submit job to it.
        pass
