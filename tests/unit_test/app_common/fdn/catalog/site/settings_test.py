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

from nvflare.app_common.fdn.catalog.common.limits import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.site.settings import SettingsError, parse_settings


class TestParseSettings:
    def test_none_is_all_defaults(self):
        settings = parse_settings(None)
        assert settings.limits == ReadLimits(**DEFAULT_LIMITS)

    def test_valid_settings(self):
        settings = parse_settings(
            {
                "limits": {"max_records": 50},
            }
        )
        assert settings.limits.max_records == 50 and settings.limits.max_calls == DEFAULT_LIMITS["max_calls"]

    @pytest.mark.parametrize(
        "value",
        [
            [],
            {"unknown": 1},
            {"limits": {"max_records": 0}},
            {"limits": {"max_records": True}},
            {"limits": {"other": 1}},
            {"connectors": {"x": {"args": {}}}},
            {"installed_operations": "cohort.count"},
        ],
    )
    def test_invalid_settings(self, value):
        with pytest.raises(SettingsError):
            parse_settings(value)
