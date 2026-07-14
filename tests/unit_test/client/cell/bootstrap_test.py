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

"""Tests for the external_process bootstrap-config contract (nvflare/client/cell/bootstrap.py)."""

import json
import os

import pytest

from nvflare.client.cell.bootstrap import BootstrapKey, read_bootstrap_config, write_bootstrap_config

CONFIG = {
    BootstrapKey.CONNECT_URL: "tcp://127.0.0.1:56789",
    BootstrapKey.CJ_FQCN: "site-1.job-1",
    BootstrapKey.TRAINER_FQCN: "site-1.job-1.client_api_trainer_1",
    BootstrapKey.LAUNCH_TOKEN: "secret-token",
    BootstrapKey.PROTOCOL_VERSION: 1,
    BootstrapKey.JOB_ID: "job-1",
    BootstrapKey.SITE_NAME: "site-1",
    BootstrapKey.TASK_EXCHANGE: {"train_task_name": "train"},
}


class TestBootstrapConfig:
    def test_write_read_round_trip(self, tmp_path):
        path = str(tmp_path / "bootstrap.json")
        write_bootstrap_config(path, CONFIG)
        assert read_bootstrap_config(path) == CONFIG

    def test_file_is_owner_only(self, tmp_path):
        path = str(tmp_path / "bootstrap.json")
        write_bootstrap_config(path, CONFIG)
        # the launch token must never be readable by other local users
        assert os.stat(path).st_mode & 0o777 == 0o600

    def test_overwrite_tightens_wider_preexisting_mode(self, tmp_path):
        # launch_once=False overwrites the same path per launch; a pre-existing file with
        # a wider mode (e.g. left by other tooling) must come out 0600 as well
        path = str(tmp_path / "bootstrap.json")
        with open(path, "w") as f:
            f.write("{}")
        os.chmod(path, 0o644)

        write_bootstrap_config(path, CONFIG)

        assert os.stat(path).st_mode & 0o777 == 0o600
        assert read_bootstrap_config(path) == CONFIG

    def test_read_rejects_non_dict_content(self, tmp_path):
        path = str(tmp_path / "bootstrap.json")
        with open(path, "w") as f:
            json.dump(["not", "a", "dict"], f)
        with pytest.raises(ValueError, match="expect a JSON dict"):
            read_bootstrap_config(path)
