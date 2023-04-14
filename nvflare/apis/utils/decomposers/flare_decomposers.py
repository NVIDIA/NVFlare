# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
"""Decomposers for objects used by NVFlare itself

This module contains all the decomposers used to run NVFlare.
The decomposers are registered at server/client startup.

"""
import os
from argparse import Namespace
from typing import Any

from nvflare.apis.client import Client
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import Decomposer, DictDecomposer


# The __init__ initializes logger so generic decomposers can't be used
class ContextDecomposer(Decomposer):
    def supported_type(self):
        return FLContext

    def decompose(self, target: FLContext) -> Any:
        return [target.model, target.props]

    def recompose(self, data: Any) -> FLContext:
        obj = FLContext()
        obj.model = data[0]
        obj.props = data[1]
        return obj


# Workspace does directory check so generic decomposer is not used
class WorkspaceDecomposer(Decomposer):
    def supported_type(self):
        return Workspace

    def decompose(self, target: Workspace) -> Any:
        return [target.root_dir, target.site_name, target.config_folder]

    def recompose(self, data: Any) -> Workspace:
        return Workspace(data[0], data[1], data[2])


def register():
    if register.registered:
        return

    fobs.register(DictDecomposer(Shareable))

    fobs.register_data_classes(DXO, Client, RunSnapshot, Signal, Namespace)

    fobs.register_folder(os.path.dirname(__file__), __package__)

    register.registered = True


register.registered = False
