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
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.event import fire_event_to_components


def fire_event(event: str, handlers: list, ctx: FLContext):
    """Fires the specified event and invokes the list of handlers.

    Args:
        event: the event to be fired
        handlers: handlers to be invoked
        ctx: context for cross-component data sharing

    Returns: N/A

    """
    return fire_event_to_components(event, handlers, ctx)
