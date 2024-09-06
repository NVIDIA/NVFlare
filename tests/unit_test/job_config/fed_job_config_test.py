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
import os
import tempfile

from nvflare.job_config.fed_job_config import FedJobConfig


class TestFedJobConfig:
    def test_locate_imports(self):
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        cwd = os.path.dirname(__file__)
        source_file = os.path.join(cwd, "../data/job_config/sample_code.data")
        expected = [
            "from typing import Any, Dict, List",
            "from nvflare.fuel.f3.drivers.base_driver import BaseDriver",
            "from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo ",
            "from nvflare.fuel.f3.drivers.driver_params import DriverCap",
        ]
        with open(source_file, "r") as sf:
            with tempfile.NamedTemporaryFile(dir=cwd, suffix=".py") as dest_file:
                imports = list(job_config.locate_imports(sf, dest_file=dest_file.name))
        assert imports == expected

    def test_trim_whitespace(self):
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        expected = "site-0,site-1"
        assert expected == job_config._trim_whitespace("site-0,site-1")
        assert expected == job_config._trim_whitespace("site-0, site-1")
        assert expected == job_config._trim_whitespace(" site-0,site-1 ")
        assert expected == job_config._trim_whitespace(" site-0, site-1 ")
