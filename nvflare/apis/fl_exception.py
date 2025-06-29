# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


class FLCommunicationError(Exception):
    """Base class for fed_learn communication exceptions."""

    def __init__(self, message, exception=None):
        """Init the FLCommunicationError.

        Args:
            exception: grpc.RpcError when trying to register gprc channel
        """
        super().__init__()
        # Copy all the exception properties into FLCommunicationError instance.
        if exception:
            self.__dict__.update(exception.__dict__)
        self.message = message

    def __str__(self):
        return self.message


class UnsafeJobError(Exception):
    """Raised when a job is detected to be unsafe"""

    pass


class NotAuthenticated(Exception):
    """Raised when client registration is not authenticated."""

    pass


class NotAuthorized(Exception):
    """Raised when a job is not authorized"""

    pass


class UnsafeComponentError(Exception):
    """Raised when a component in the configuration is detected to be unsafe"""

    pass


class TaskExecutionError(Exception):
    """Raised when a task execution failed"""

    pass


class NotReadyToEndRun(Exception):
    """Raised when a component is not ready to end run"""

    pass
