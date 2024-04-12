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

from cma_decomposer import register_decomposers

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.widgets.widget import Widget


class RegisterDecomposer(Widget):
    def __init__(self):
        """Handler to register CMA decomposers."""
        super().__init__()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            # We serialize CMAEvolutionStrategy object directly. This requires registering custom decomposers.
            register_decomposers()
