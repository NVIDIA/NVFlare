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

import pytest

from nvflare import FedAvg
from nvflare.job_config.fed_job import FedJob


class TestFedJob:
    def test_validate_targets(self):
        job = FedJob()
        component = FedAvg()

        job.to(component, "server")
        job.to(component, "site-1")

        with pytest.raises(Exception):
            job.to(component, "site-/1", gpu=0)

    def test_non_empty_target(self):
        job = FedJob()
        component = FedAvg()
        with pytest.raises(Exception):
            job.to(component, None)
