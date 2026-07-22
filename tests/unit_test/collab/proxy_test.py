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

from nvflare.collab.api.proxy import Proxy


@pytest.mark.parametrize("optional, expected_handle_count", [(False, 1), (True, 0)])
def test_optional_call_failure_does_not_invoke_backend_exception_handler(optional, expected_handle_count):
    app = MagicMock()
    app.name = "server"
    context = MagicMock()
    context.__enter__.return_value = context
    app.new_context.return_value = context
    app.apply_outgoing_call_filters.side_effect = lambda _target, _func, kwargs, _ctx: kwargs
    backend = MagicMock()
    error = RuntimeError("remote call failed")
    backend.call_target.return_value = error
    proxy = Proxy(
        app=app,
        target_name="site-1",
        target_fqn="",
        backend=backend,
        target_interface={"train": []},
    )

    result = proxy(optional=optional).train()

    assert result is error
    assert backend.handle_exception.call_count == expected_handle_count
