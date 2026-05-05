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

from nvflare.private.fed.app import utils


@pytest.mark.parametrize("version_info", [(3, 10, 0), (3, 13, 0), (3, 14, 0)])
def test_version_check_accepts_supported_python_versions(monkeypatch, version_info):
    monkeypatch.setattr(utils.sys, "version_info", version_info)

    utils.version_check()


@pytest.mark.parametrize(
    "version_info, expected_message",
    [
        ((3, 9, 18), "Python versions 3.9 and below are not supported"),
        ((3, 15, 0), "Python versions 3.15 and above are not yet supported"),
    ],
)
def test_version_check_rejects_unsupported_python_versions(monkeypatch, version_info, expected_message):
    monkeypatch.setattr(utils.sys, "version_info", version_info)

    with pytest.raises(RuntimeError, match=expected_message):
        utils.version_check()
