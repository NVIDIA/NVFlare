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

import json
import tempfile
from unittest.mock import patch

import pytest

from nvflare.lighter.cc_provision.cc_constants import CC_AUTHORIZERS_KEY
from nvflare.lighter.cc_provision.impl.onprem_cvm import OnPremCVMBuilder
from nvflare.lighter.constants import PropKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Entity, Project


@pytest.fixture
def basic_project():
    """Create a basic project with server and clients and cc enabled."""
    project = Project("test_project", "A testing project")
    props = {
        PropKey.CC_CONFIG: "test_cc_config.yml",
        PropKey.CC_ENABLED: True,
        PropKey.CC_CONFIG_DICT: {
            "compute_env": "onprem_cvm",
            "cc_cpu_mechanism": "amd_sev_snp",
            "cc_gpu_mechanism": "nvidia_cc",
            "cc_issuer": "test_issuer",
            "cc_attestation": "test_attestation",
        },
    }
    project.set_server("server", "orgC", props)
    project.add_client("client1", "orgA", props)
    project.add_client("client2", "orgB", props)
    return project


@pytest.fixture
def ctx(basic_project):
    """Create a basic provisioning context."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield ProvisionContext(temp_dir, basic_project)


@pytest.fixture
def builder():
    """Create a builder instance."""
    # Create a test authorizers file
    test_authorizers = {"cc_authorizers": [{"id": "test_authorizer", "path": "test.path.TestAuthorizer", "args": {}}]}

    return OnPremCVMBuilder()


class TestOnPremCVMBuilder:
    """Test suite for OnPremCVMBuilder."""

    @patch("nvflare.lighter.utils.write")
    @patch("os.path.exists")
    @patch("nvflare.lighter.cc_provision.impl.onprem_cvm._change_log_dir")
    def test_build_resources(self, mock_change, mock_exists, mock_write, builder, basic_project, ctx):
        """Test building resources for an entity."""
        mock_exists.return_value = True
        server = basic_project.get_server()
        ctx[CC_AUTHORIZERS_KEY] = [{"id": "test_authorizer", "path": "test.path.TestAuthorizer", "args": {}}]

        builder._build_resources(server, ctx)

        # Verify resource file creation
        mock_write.assert_called_once()
        call_args = mock_write.call_args[0]
        assert call_args[0].endswith("test_authorizer__p_resources.json")

        # Verify resource content
        content = json.loads(call_args[1])
        assert "components" in content
        assert len(content["components"]) == 1
        assert content["components"][0]["id"] == "test_authorizer"
        assert content["components"][0]["path"] == "test.path.TestAuthorizer"
        assert content["components"][0]["args"] == {}

    def test_build(self, builder, basic_project, ctx):
        """Test the complete build process."""
        with patch.object(builder, "_build_resources") as mock_build:
            builder.build(basic_project, ctx)

            # Verify build calls
            assert mock_build.call_count == 3  # Server + 2 clients
            calls = mock_build.call_args_list

            # Verify server build
            server_call = calls[0]
            assert isinstance(server_call[0][0], Entity)
            assert server_call[0][0].name == "server"
            assert server_call[0][0].org == "orgC"

            # Verify client builds
            client1_call = calls[1]
            assert isinstance(client1_call[0][0], Entity)
            assert client1_call[0][0].name == "client1"
            assert client1_call[0][0].org == "orgA"

            client2_call = calls[2]
            assert isinstance(client2_call[0][0], Entity)
            assert client2_call[0][0].name == "client2"
            assert client2_call[0][0].org == "orgB"
