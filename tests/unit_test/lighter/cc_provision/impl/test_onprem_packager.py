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

from unittest.mock import patch

import pytest

from nvflare.lighter.cc_provision.impl.onprem_packager import OnPremPackager
from nvflare.lighter.constants import CtxKey, PropKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Project


@pytest.fixture
def packager():
    """Create a packager instance."""
    return OnPremPackager(cc_config_key="cc_config", build_image_cmd="custom_build_cmd.sh")


class TestOnPremPackager:
    def test_initialization(self, packager):
        """Test packager initialization."""
        assert packager.cc_config_key == "cc_config"
        assert packager.build_image_cmd == "custom_build_cmd.sh"

    def test_package_accepts_project_cc_config_mapping_with_file(self, packager, tmp_path):
        cc_config_file = tmp_path / "cc_site.yml"
        cc_config_file.write_text("compute_env: onprem_cvm\n")
        tar_file = tmp_path / "site.tgz"
        tar_file.write_text("package")
        result_dir = tmp_path / "prod"
        site_dir = result_dir / "server"
        site_dir.mkdir(parents=True)

        project = Project("test_project", "A testing project")
        project.set_server("server", "org", {PropKey.CC_CONFIG: {"file": str(cc_config_file)}})
        ctx = ProvisionContext(str(tmp_path), project)
        ctx[CtxKey.CURRENT_PROD_DIR] = str(result_dir)

        with patch.object(packager, "_build_cc_image", return_value=str(tar_file)):
            packager._package_for_participant(project.get_server(), ctx)

        assert (site_dir / "server.tgz").read_text() == "package"
