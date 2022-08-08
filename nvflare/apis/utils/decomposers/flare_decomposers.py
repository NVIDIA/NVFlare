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
"""Decomposers for objects used by NVFlare itself

This module contains all the decomposers used to run NVFlare.
The decomposers are registered at server/client startup.

"""
import os
from argparse import Namespace
from typing import Any

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.client import Client
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import Decomposer


class ShareableDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return Shareable

    def decompose(self, target: Shareable) -> Any:
        return target.copy()

    def recompose(self, data: Any) -> Shareable:
        obj = Shareable()
        for k, v in data.items():
            obj[k] = v
        return obj


class ContextDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return FLContext

    def decompose(self, target: FLContext) -> Any:
        return [target.model, target.props]

    def recompose(self, data: Any) -> FLContext:
        obj = FLContext()
        obj.model = data[0]
        obj.props = data[1]
        return obj


class DxoDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return DXO

    def decompose(self, target: DXO) -> Any:
        return [target.data_kind, target.data, target.meta]

    def recompose(self, data: Any) -> DXO:
        return DXO(data[0], data[1], data[2])


class ClientDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return Client

    def decompose(self, target: Client) -> Any:
        return [target.name, target.token, target.last_connect_time, target.props]

    def recompose(self, data: Any) -> Client:
        client = Client(data[0], data[1])
        client.last_connect_time = data[2]
        client.props = data[3]
        return client


class RunSnapshotDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return RunSnapshot

    def decompose(self, target: RunSnapshot) -> Any:
        return [target.component_states, target.completed, target.job_id]

    def recompose(self, data: Any) -> RunSnapshot:
        snapshot = RunSnapshot(data[2])
        snapshot.component_states = data[0]
        snapshot.completed = data[1]
        return snapshot


class WorkspaceDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return Workspace

    def decompose(self, target: Workspace) -> Any:
        return [target.root_dir, target.name, target.config_folder]

    def recompose(self, data: Any) -> Workspace:
        return Workspace(data[0], data[1], data[2])


class SignalDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return Signal

    def decompose(self, target: Signal) -> Any:
        return [target.value, target.trigger_time, target.triggered]

    def recompose(self, data: Any) -> Signal:
        signal = Signal()
        signal.value = data[0]
        signal.trigger_time = data[1]
        signal.triggered = data[2]
        return signal


class AnalyticsDataTypeDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return AnalyticsDataType

    def decompose(self, target: AnalyticsDataType) -> Any:
        return target.name

    def recompose(self, data: Any) -> AnalyticsDataType:
        return AnalyticsDataType[data]


class NamespaceDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return Namespace

    def decompose(self, target: Namespace) -> Any:
        return vars(target)

    def recompose(self, data: Any) -> Namespace:
        return Namespace(**data)


def register():
    if register.registered:
        return

    fobs.register_folder(os.path.dirname(__file__), __package__)
    register.registered = True


register.registered = False
