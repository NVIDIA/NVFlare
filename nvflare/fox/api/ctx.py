# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.signal import Signal

fox_context = threading.local()


class Context:

    def __init__(self, app, caller: str, callee: str, abort_signal: Signal, target_group=None):
        if not isinstance(caller, str):
            raise ValueError(f"caller must be str but got {type(caller)}")

        if not isinstance(callee, str):
            raise ValueError(f"callee must be str but got {type(callee)}")

        self.caller = caller
        self.callee = callee
        self.target_group = target_group
        self.abort_signal = abort_signal
        self.app = app
        self.props = {}
        self.parent_ctx = get_call_context()

    @property
    def backend(self):
        return self.app.backend

    @property
    def backend_type(self):
        return self.app.backend_type

    @property
    def clients(self):
        return self.app.client_proxies

    @property
    def server(self):
        return self.app.server_proxy

    @property
    def client_hierarchy(self):
        return self.app.client_hierarchy

    @property
    def workspace(self):
        return self.app.workspace

    @property
    def target_group_size(self):
        if self.target_group:
            return self.target_group.size
        else:
            return 1

    def set_prop(self, name: str, value):
        self.props[name] = value

    def get_prop(self, name: str, default=None):
        return self.props.get(name, default)

    def is_aborted(self):
        return self.abort_signal and self.abort_signal.triggered

    def __str__(self):
        return f"{self.app.name}:{self.caller}=>{self.callee}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parent_ctx:
            set_call_context(self.parent_ctx)
        else:
            set_call_context(None)


def get_call_context():
    if hasattr(fox_context, "call_ctx"):
        return fox_context.call_ctx
    else:
        return None


def set_call_context(ctx):
    fox_context.call_ctx = ctx
