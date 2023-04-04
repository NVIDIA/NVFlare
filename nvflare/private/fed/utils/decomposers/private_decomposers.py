# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
"""Decomposers for objects used by NVFlare platform privately

This module contains all the decomposers used to run NVFlare.
The decomposers are registered at server/client startup.

"""

from nvflare.fuel.utils import fobs
from nvflare.private.admin_defs import Message
from nvflare.private.fed.server.run_info import RunInfo
from nvflare.private.fed.server.server_state import Cold2HotState, ColdState, Hot2ColdState, HotState, ShutdownState


def register():
    if register.registered:
        return

    fobs.register_data_classes(Message, RunInfo, HotState, ColdState, Hot2ColdState, Cold2HotState, ShutdownState)

    register.registered = True


register.registered = False
