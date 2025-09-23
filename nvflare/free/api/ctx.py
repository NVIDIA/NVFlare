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
from nvflare.apis.signal import Signal


class Context:

    def __init__(self, caller: str, callee: str, abort_signal: Signal, props: dict = None):
        self.caller = caller
        self.callee = callee
        self.abort_signal = abort_signal
        self.server = None
        self.clients = None
        self.app = None
        self.props = {}
        if props:
            self.props.update(props)

    def set_prop(self, name: str, value):
        self.props[name] = value

    def get_prop(self, name: str, default=None):
        return self.props.get(name, default)

    def is_aborted(self):
        return self.abort_signal and self.abort_signal.triggered
