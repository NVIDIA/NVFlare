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

from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.context import set_call_context
from nvflare.collab.api.decorators import get_object_publish_interface, publish
from nvflare.collab.api.exceptions import CollabCallError, RunAborted
from nvflare.collab.api.proxy import Proxy
from nvflare.collab.api.proxy_list import ProxyList


def _make_failing_proxy(error):
    app = MagicMock()
    app.name = "server"
    context = MagicMock()
    context.__enter__.return_value = context
    app.new_context.return_value = context
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


class _BindingTarget:
    @publish
    def update(self, value, *, rounds):
        return value, rounds

    @publish
    def evaluate(self, model, /, *, metric):
        return model, metric


def _make_binding_proxy():
    app = MagicMock()
    app.name = "server"
    context = MagicMock()
    context.__enter__.return_value = context
    app.new_context.return_value = context
    backend = MagicMock()
    backend.call_target.return_value = "ok"
    proxy = Proxy(
        app=app,
        target_name="site-1",
        target_fqn="",
        backend=backend,
        target_interface=get_object_publish_interface(_BindingTarget()).to_dict(),
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


def test_proxy_rejects_duplicate_positional_and_keyword_argument():
    proxy, backend = _make_binding_proxy()

    with pytest.raises(CollabCallError, match="multiple values for argument 'value'") as exc_info:
        proxy.update(1, value=2, rounds=3)

    assert isinstance(exc_info.value.cause, TypeError)
    backend.call_target.assert_not_called()


def test_proxy_enforces_keyword_only_and_positional_only_arguments():
    proxy, backend = _make_binding_proxy()

    assert proxy.evaluate("model", metric="accuracy") == "ok"
    backend.call_target.assert_called_once()

    backend.reset_mock()
    with pytest.raises(CollabCallError, match="takes 1 positional arguments but 2 were given") as exc_info:
        proxy.evaluate("model", "accuracy")
    assert isinstance(exc_info.value.cause, TypeError)

    with pytest.raises(CollabCallError, match="positional-only argument passed as keyword") as exc_info:
        proxy.evaluate(model="model", metric="accuracy")
    assert isinstance(exc_info.value.cause, TypeError)
    backend.call_target.assert_not_called()


def test_optional_call_failure_returns_none_without_panicking():
    error = RuntimeError("remote call failed")
    proxy, backend = _make_failing_proxy(error)

    result = proxy(optional=True).train()

    assert result is None
    backend.handle_exception.assert_not_called()


def test_optional_call_propagates_run_abort():
    error = RunAborted("run is aborted")
    proxy, backend = _make_failing_proxy(error)

    with pytest.raises(RunAborted, match="run is aborted"):
        proxy(optional=True).train()

    backend.handle_exception.assert_not_called()


def test_proxy_variants_share_the_call_option_default_timeout():
    proxy, _ = _make_failing_proxy(None)
    context = MagicMock()
    context.app = proxy.app
    context.abort_signal = MagicMock()

    set_call_context(context)
    try:
        assert proxy().call_opt.timeout == 60.0
        assert ProxyList([proxy])()._call_opt.timeout == 60.0
    finally:
        set_call_context(None)


def test_proxy_rejects_missing_private_attributes():
    proxy, _ = _make_failing_proxy(None)

    with pytest.raises(AttributeError):
        getattr(proxy, "_missing")
    with pytest.raises(AttributeError):
        getattr(proxy(), "_missing")
    with pytest.raises(AttributeError):
        getattr(ProxyList([proxy]), "_missing")


def test_proxy_list_method_names_are_dispatched_instead_of_mutating_the_list():
    app = ClientApp(object())
    app.name = "server"
    backend = MagicMock()

    def complete(gcc, *_args, **_kwargs):
        gcc.set_result("remote-pop")
        gcc.call_completed()

    backend.call_target_in_group.side_effect = complete
    proxy = Proxy(
        app=app,
        target_name="site-1",
        target_fqn="",
        backend=backend,
        target_interface={"pop": []},
    )
    proxies = ProxyList([proxy])

    with app.new_context("server", "server"):
        assert list(proxies(blocking=False).pop()) == [("site-1", "remote-pop")]
    assert len(proxies) == 1
