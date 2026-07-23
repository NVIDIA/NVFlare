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

from unittest.mock import MagicMock

import pytest

from nvflare.collab.runtime.lifecycle import run_server


def test_run_server_finalizes_when_initialize_raises():
    server_app = MagicMock()
    server_ctx = MagicMock()
    server_app.new_context.return_value = server_ctx
    server_app.initialize.side_effect = ValueError("initialization failed")

    with pytest.raises(ValueError, match="initialization failed"):
        run_server(server_app, MagicMock())

    server_app.finalize.assert_called_once_with(server_ctx)
