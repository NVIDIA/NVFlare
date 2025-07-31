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

from unittest.mock import call, mock_open, patch

import pytest

from nvflare.lighter.cc_provision.impl.onprem_packager import OnPremPackager
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Project


@pytest.fixture
def basic_project():
    """Create a basic project with server and clients."""
    project = Project("test_project", "A testing project")
    props = {"cc_config": "test_config.yaml"}
    project.set_server("server", "orgC", props)
    project.add_client("client1", "orgA", props)
    project.add_client("client2", "orgB", props)
    return project


@pytest.fixture
def ctx(basic_project):
    """Create a basic provisioning context."""
    ctx = ProvisionContext("test_workspace", basic_project)
    ctx.get_result_location = lambda: "test_workspace"
    return ctx


@pytest.fixture
def packager():
    """Create a packager instance."""
    return OnPremPackager(cc_config_key="cc_config", build_image_cmd="custom_build_cmd.sh")


class TestOnPremPackager:
    def test_initialization(self, packager):
        """Test packager initialization."""
        assert packager.cc_config_key == "cc_config"
        assert packager.build_image_cmd == "custom_build_cmd.sh"

    @patch("shutil.copy")
    @patch("shutil.rmtree")
    @patch("os.mkdir")
    @patch("builtins.open", new_callable=mock_open, read_data="docker run {~~cvm_image_name~~}")
    @patch("nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager._build_cc_image")
    @patch("nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager._change_log_dir")
    def test_package_for_participant(
        self,
        mock_change_log_dir,
        mock_build_cc_image,
        mock_open,
        mock_makedirs,
        mock_rmtree,
        mock_copy,
        packager,
        basic_project,
        ctx,
    ):
        """Test packaging for a participant."""

        packager.package(basic_project, ctx)

        assert mock_build_cc_image.call_count == 3
        mock_build_cc_image.assert_has_calls(
            [call("test_config.yaml", p.name, "test_workspace") for p in basic_project.get_all_participants()],
            any_order=True,
        )
