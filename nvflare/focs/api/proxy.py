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
import copy

from .backend import Backend
from .constants import CollabMethodArgName


class Proxy:

    def __init__(self, app, target_name, backend: Backend, target_signature=None):
        """The Proxy represents a target in the App."""
        self.app = app
        self.target_name = target_name
        self.backend = backend
        self.caller_name = app.name
        self.target_signature = target_signature

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
            call_args = args
            call_kwargs = kwargs

            if self.target_signature:
                arg_names = self.target_signature.get(func_name)
                if arg_names:
                    # check args and turn them to kwargs
                    call_kwargs = copy.copy(kwargs)
                    call_args = []
                    for i, arg_value in enumerate(args):
                        call_kwargs[arg_names[i]] = arg_value

            ctx = self.app.new_context(self.caller_name, self.name)
            call_kwargs[CollabMethodArgName.CONTEXT] = ctx

            print(f"calling target {self.target_name} func {func_name}: {call_args=} {call_kwargs=}")
            return self.backend.call_target(self.target_name, func_name, *call_args, **call_kwargs)

        return method
