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

from nvflare.collab import collab
from nvflare.collab.api.context import set_call_context


@pytest.mark.parametrize(
    "accessor",
    [
        lambda: collab.context,
        lambda: collab.fl_ctx,
        lambda: collab.caller,
        lambda: collab.get_prop("missing"),
    ],
)
def test_facade_requires_active_call_context(accessor):
    set_call_context(None)

    with pytest.raises(RuntimeError, match="Collab context is only available"):
        accessor()
