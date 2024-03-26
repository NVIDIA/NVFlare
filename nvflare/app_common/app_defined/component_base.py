# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_component import FLComponent


class ComponentBase(FLComponent):
    def __init__(self):
        FLComponent.__init__(self)
        self.fl_ctx = None

    def debug(self, msg: str):
        """Convenience method for logging an DEBUG message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_debug(self.fl_ctx, msg)

    def info(self, msg: str):
        """Convenience method for logging an INFO message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_info(self.fl_ctx, msg)

    def error(self, msg: str):
        """Convenience method for logging an ERROR message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_error(self.fl_ctx, msg)

    def warning(self, msg: str):
        """Convenience method for logging a WARNING message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_warning(self.fl_ctx, msg)

    def exception(self, msg: str):
        """Convenience method for logging an EXCEPTION message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_exception(self.fl_ctx, msg)

    def critical(self, msg: str):
        """Convenience method for logging a CRITICAL message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_critical(self.fl_ctx, msg)
