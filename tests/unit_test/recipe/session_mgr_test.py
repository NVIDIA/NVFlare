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

import pytest

from nvflare.fuel.utils.secret_utils import PotentialSecretWarning
from nvflare.job_config.api import FedJob
from nvflare.recipe.session_mgr import SessionManager


def test_submit_job_scans_generated_config_before_submission():
    job = FedJob(name="secret-submit-job", min_clients=1)
    job.to_server({"auth_token": "abcd1234efgh"})

    session = MagicMock()
    session.submit_job.return_value = "job-id"
    manager = SessionManager({})
    manager._get_session = MagicMock(return_value=session)

    with pytest.warns(PotentialSecretWarning, match="generated job file"):
        assert manager.submit_job(job) == "job-id"

    session.submit_job.assert_called_once()
    session.close.assert_called_once()
