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
from abc import ABC, abstractmethod

from nvflare.apis.signal import Signal
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_traceback

from .call_opt import CallOption
from .ctx import Context
from .gcc import GroupCallContext


class Backend(ABC):
    """A FOX Backend implements remote object calls. This interface defines the required methods that a Backend
    must implement.
    """

    def __init__(self, abort_signal: Signal):
        self.abort_signal = abort_signal
        self.logger = get_obj_logger(self)

    @abstractmethod
    def call_target(self, context: Context, target_name: str, call_opt: CallOption, func_name: str, *args, **kwargs):
        """
        Call a target function with arguments and return a result.

        Args:
            context: the call context
            target_name: the fully qualified name of the target object to be called in the remote app.
            call_opt: call options.
            func_name: name of the function to be called in the remote app.
            *args: args to pass to the target function.
            **kwargs: kwargs to pass to the target function.

        Notes: the target name is fully qualified: <target_app_name>.<obj_name>

        Returns:

        """
        pass

    @abstractmethod
    def call_target_in_group(self, gcc: GroupCallContext, func_name: str, *args, **kwargs):
        """Call a remote object as part of a group.

        Args:
            gcc: contextual information about group call.
            func_name: name of the function to be called in the remote app.
            *args: args to pass to the target function.
            **kwargs: kwargs to pass to the target function.

        Returns:

        """
        pass

    def handle_exception(self, exception: Exception):
        self.logger.error(f"exception occurred: {secure_format_traceback()}")
        raise exception
