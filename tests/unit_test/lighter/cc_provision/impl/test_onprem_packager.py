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

import pytest

from nvflare.lighter.cc_provision.impl.onprem_packager import OnPremPackager


@pytest.fixture
def packager():
    """Create a packager instance."""
    return OnPremPackager(cc_config_key="cc_config", build_image_cmd="custom_build_cmd.sh")


class TestOnPremPackager:
    def test_initialization(self, packager):
        """Test packager initialization."""
        assert packager.cc_config_key == "cc_config"
        assert packager.build_image_cmd == "custom_build_cmd.sh"
