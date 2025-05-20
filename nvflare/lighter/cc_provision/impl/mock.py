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

from pathlib import Path

from .onprem_cvm import OnPremCVMBuilder

AUTHORIZERS_YAML_PATH = Path(__file__).parent.parent / "resources" / "mock" / "cc_authorizers.yml"


class MockBuilder(OnPremCVMBuilder):
    """Builder for Mock Confidential Virtual Machines (for testing)."""

    def __init__(
        self,
        token_expiration=3600,
        authorizers_yaml_path: str = AUTHORIZERS_YAML_PATH,
    ):
        """Initialize the mock builder.

        Args:
            token_expiration: Token expiration time in seconds
            authorizer_id: ID of the authorizer
            authorizer_path: Path to the authorizer class
        """
        super().__init__(
            token_expiration=token_expiration,
            authorizers_yaml_path=authorizers_yaml_path,
        )
