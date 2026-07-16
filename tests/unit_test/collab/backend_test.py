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

from nvflare.apis.signal import Signal
from nvflare.collab.api.backend import Backend
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.context import Context


class _GroupOnlyBackend(Backend):
    def __init__(self, result):
        super().__init__(Signal())
        self.result = result
        self.group_context = None

    def call_target_in_group(self, gcc, func_name: str, *args, **kwargs):
        self.group_context = gcc
        gcc.set_result(self.result)


def test_call_target_uses_single_member_group():
    backend = _GroupOnlyBackend(result="result")
    context = Context(app=object(), caller="server", callee="site-1", abort_signal=backend.abort_signal)

    result = backend.call_target(
        context=context,
        target_name="site-1.trainer",
        call_opt=CallOption(),
        func_name="train",
        value=1,
    )

    assert result == "result"
    assert backend.group_context.target_name == "site-1.trainer"
    assert backend.group_context.context is context
    assert backend.group_context.waiter.sites == ["site-1.trainer"]


def test_call_target_returns_immediately_when_no_result_is_expected():
    backend = _GroupOnlyBackend(result="ignored")
    context = Context(app=object(), caller="server", callee="site-1", abort_signal=backend.abort_signal)

    result = backend.call_target(
        context=context,
        target_name="site-1.trainer",
        call_opt=CallOption(expect_result=False),
        func_name="stop",
    )

    assert result is None
