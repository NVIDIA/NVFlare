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

import threading
from unittest.mock import MagicMock

import pytest

from nvflare.apis.signal import Signal
from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.group import Group
from nvflare.collab.api.proxy import Proxy


def _proxy(app, name, interface, backend=None):
    return Proxy(
        app=app,
        target_name=name,
        target_fqn=name,
        backend=backend or MagicMock(),
        target_interface=interface,
    )


def test_group_preflights_all_members_before_dispatch():
    app = ClientApp(object())
    first_backend = MagicMock()
    second_backend = MagicMock()
    proxies = [
        _proxy(app, "site-1", {"train": ["value"]}, first_backend),
        _proxy(app, "site-2", {"other": []}, second_backend),
    ]

    with pytest.raises(RuntimeError, match="does not have method 'train'"):
        Group(app, Signal(), proxies).train(1)

    first_backend.call_target_in_group.assert_not_called()
    second_backend.call_target_in_group.assert_not_called()


def test_nonblocking_group_returns_before_bounded_dispatch_completes():
    app = ClientApp(object())
    first_dispatched = threading.Event()
    second_dispatched = threading.Event()
    first_call = {}

    def dispatch_first(gcc, *_args, **_kwargs):
        first_call["gcc"] = gcc
        first_dispatched.set()

    def dispatch_second(gcc, *_args, **_kwargs):
        second_dispatched.set()
        gcc.set_result("second")
        gcc.call_completed()

    first_backend = MagicMock()
    first_backend.call_target_in_group.side_effect = dispatch_first
    second_backend = MagicMock()
    second_backend.call_target_in_group.side_effect = dispatch_second
    proxies = [
        _proxy(app, "site-1", {"train": []}, first_backend),
        _proxy(app, "site-2", {"train": []}, second_backend),
    ]
    group = Group(app, Signal(), proxies, CallOption(blocking=False, parallel=1))
    returned = {}

    call_returned = threading.Event()

    def invoke():
        returned["results"] = group.train()
        call_returned.set()

    thread = threading.Thread(target=invoke)
    thread.start()
    assert call_returned.wait(timeout=1.0)
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    assert first_dispatched.wait(timeout=1.0)
    assert not second_dispatched.wait(timeout=0.1)

    first_call["gcc"].set_result("first")
    first_call["gcc"].call_completed()

    assert second_dispatched.wait(timeout=1.0)
    assert sorted(returned["results"]) == [("site-1", "first"), ("site-2", "second")]
