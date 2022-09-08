# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

    def __init__(self, exception, message):
        """Init the FLCommunicationError.

        Args:
            exception: grpc.RpcError when trying to register gprc channel
        """
        super().__init__()
        # Copy all the gRPC exception properties into FLCommunicationError instance.
        self.__dict__.update(exception.__dict__)
        self.message = message


class WorkflowError(Exception):
    """FL Workflow error to indicate not to continue workflow execution."""

    def __init__(self, *args: object) -> None:
        """Init the WorkflowError.

        Args:
            *args: variable number of arguments for Exception; usually is error message string
        """
        super().__init__(*args)
