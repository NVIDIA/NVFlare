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
from enum import Enum

from nvflare.apis.fl_component import FLComponent


class Widget(FLComponent):
    """Pre-defined components that address specific needs.

    Some examples of such needs:
        - report current status
        - dynamically change its tunable parameters
        - record processing errors
        - stats recording

    Each widget is a singleton object that is registered with the Engine with a
    unique ID.

    All built-in widget IDs are documented in the WidgetID class.

    """

    def __init__(self):
        """Init the Widget."""
        FLComponent.__init__(self)


class WidgetID(str, Enum):

    INFO_COLLECTOR = "info_collector"
    COMPONENT_CALLER = "component_caller"
    FED_EVENT_RUNNER = "fed_event_runner"
