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

from nvflare.apis.resource_manager_spec import ResourceConsumerSpec


class BEResourceConsumer(ResourceConsumerSpec):
    """A best-effort resource consumer that accepts any resource allocation without action.

    This implementation of ResourceConsumerSpec is a no-op consumer intended for use
    in environments where resource consumption tracking is not required or is handled
    externally. It silently accepts the provided resources and takes no further action.
    """

    def consume(self, resources: dict):
        """Consumes the given resources without performing any action.

        Args:
            resources: a dict of allocated resources to consume.
        """
        pass
