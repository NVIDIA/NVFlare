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

from nvflare.tool.examples import store
from tests.unit_test.tool.examples.helpers import Remote


@pytest.fixture
def remote():
    result = Remote()
    yield result
    result.source.close()


@pytest.fixture(autouse=True)
def reset_output_mode():
    from nvflare.tool.cli_output import set_output_format

    set_output_format("txt")
    yield
    set_output_format("txt")


@pytest.fixture
def cache(tmp_path, remote):
    return store.ExampleStore(tmp_path / "cache", remote.source)
