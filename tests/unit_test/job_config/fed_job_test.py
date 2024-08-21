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

from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.executors.model_learner_executor import ModelLearnerExecutor
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.api import FedJob


class TestFedJob:
    def test_validate_targets(self):
        job = FedJob()
        controller = FedAvg()
        executor = ModelLearnerExecutor(learner_id=job.as_id(ModelLearner()))

        job.to(controller, "server")
        job.to(executor, "site-1")

        with pytest.raises(Exception):
            job.to(executor, "site-/1")

    def test_non_empty_target(self):
        job = FedJob()
        component = FedAvg()
        with pytest.raises(Exception):
            job.to(component, None)
