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

import pytest

from nvflare.apis.app_validation import AppValidationKey


class TestAppValidationKey:
    """Test AppValidationKey constants."""

    def test_flower_predeployed_constant_exists(self):
        """FLOWER_PREDEPLOYED constant is defined."""
        assert hasattr(AppValidationKey, 'FLOWER_PREDEPLOYED')

    def test_flower_predeployed_constant_value(self):
        """FLOWER_PREDEPLOYED constant has correct value."""
        assert AppValidationKey.FLOWER_PREDEPLOYED == "flower_predeployed"

    def test_byoc_constant_unchanged(self):
        """BYOC constant still exists and unchanged."""
        assert AppValidationKey.BYOC == "byoc"
