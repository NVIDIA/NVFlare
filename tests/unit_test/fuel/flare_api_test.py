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

from nvflare.fuel.flare_api.api_spec import InvalidJobDefinition
from nvflare.fuel.flare_api.flare_api import Session
from nvflare.fuel.hci.client.api import ResultKey
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue


@pytest.mark.parametrize("job_name", ["fox training poc", ".hidden-job", "-flaggy-job"])
def test_submit_job_rejects_invalid_job_folder_names(tmp_path, job_name):
    job_dir = tmp_path / job_name
    job_dir.mkdir()

    session = Session.__new__(Session)
    session.upload_dir = str(tmp_path)
    session._do_command = MagicMock()

    with pytest.raises(InvalidJobDefinition, match=rf"job folder name '{job_name}'.*no spaces"):
        session.submit_job(str(job_dir))

    session._do_command.assert_not_called()


def test_submit_job_accepts_valid_job_folder_name(tmp_path):
    job_dir = tmp_path / "fox-training_poc.1"
    job_dir.mkdir()

    session = Session.__new__(Session)
    session.upload_dir = str(tmp_path)
    session._do_command = MagicMock(
        return_value={
            ResultKey.STATUS: "SUCCESS",
            ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK, MetaKey.JOB_ID: "job-1"},
        }
    )

    assert session.submit_job(str(job_dir)) == "job-1"
    session._do_command.assert_called_once()
