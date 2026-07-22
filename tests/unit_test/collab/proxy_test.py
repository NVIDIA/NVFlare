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

from nvflare.collab.api.exceptions import CollabCallError
from nvflare.collab.api.proxy import Proxy


def _make_failing_proxy(error):
    app = MagicMock()
    app.name = "server"
    context = MagicMock()
    context.__enter__.return_value = context
    app.new_context.return_value = context
    app.apply_outgoing_call_filters.side_effect = lambda _target, _func, kwargs, _ctx: kwargs
    backend = MagicMock()
    backend.call_target.return_value = error
    proxy = Proxy(
        app=app,
        target_name="site-1",
        target_fqn="",
        backend=backend,
        target_interface={"train": []},
    )
    return proxy, backend


def test_required_call_failure_raises_without_panicking_immediately():
    error = RuntimeError("remote call failed")
    proxy, backend = _make_failing_proxy(error)

    with pytest.raises(CollabCallError, match="remote call failed") as exc_info:
        proxy.train()

    assert exc_info.value.site == "site-1"
    assert exc_info.value.func_name == "train"
    assert exc_info.value.cause is error
    backend.handle_exception.assert_not_called()


def test_optional_call_failure_returns_none_without_panicking():
    error = RuntimeError("remote call failed")
    proxy, backend = _make_failing_proxy(error)

    result = proxy(optional=True).train()

    assert result is None
    backend.handle_exception.assert_not_called()
