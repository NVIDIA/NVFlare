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
from typing import List, Optional

from nvflare.app_common.ccwf.ccwf_job import CCWFJob
from nvflare.job_config.pt.model import Wrap


class SwarmJob(CCWFJob):
    def __init__(
        self,
        initial_model,
        name,
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
    ):
        super().__init__(name, min_clients, mandatory_clients)
        self.comp_ids = self.to_server(Wrap(initial_model))
