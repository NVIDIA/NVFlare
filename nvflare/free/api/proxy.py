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
from .backend import Backend
from .constants import CollabMethodArgName
from .ctx import Context


class Proxy:

    def __init__(self, app, target_name, backend: Backend, caller_name: str):
        self.app = app
        self.target_name = target_name
        self.backend = backend
        self.caller_name = caller_name

    @property
    def name(self):
        return self.target_name

    def get_target(self, name: str):
        obj = getattr(self, name, None)
        if not obj:
            return None
        if isinstance(obj, Proxy):
            return obj
        else:
            return None

    def __getattr__(self, func_name):
        """
        This method is called when Python cannot find an invoked method func_name of this class.
        """

        def method(*args, **kwargs):
            ctx = Context(self.caller_name, self.name, self.backend.abort_signal)
            kwargs[CollabMethodArgName.CONTEXT] = ctx
            return self.backend.call_target(self.target_name, func_name, *args, **kwargs)

        return method
