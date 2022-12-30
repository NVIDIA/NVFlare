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
from abc import ABC, abstractmethod
from typing import Dict, Optional

import logging

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.tracker_types import TrackConst, TrackerType, ANALYTIC_EVENT_TYPE
from nvflare.widgets.widget import Widget


class ExperimentTracker(Widget):
    def __init__(self):
        super().__init__()
        self.engine = None
        self.fl_ctx: Optional[FLContext] = None
        self.logger = logging.getLogger(self._name)

    @abstractmethod
    def get_tracker_type(self) -> TrackerType:
        raise NotImplementedError

    @abstractmethod
    def get_initial_kwargs(self):
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            self.fl_ctx = fl_ctx
            self.add(
                key=TrackConst.SITE_KEY,
                value=fl_ctx.get_identity_name(),
                data_type=AnalyticsDataType.INIT_DATA,
                global_step=0,
                kwargs=self.get_initial_kwargs(),
            )

    def close(self):
        if self.engine:
            self.engine = None

    def add(
            self,
            key: str,
            value: any,
            data_type: AnalyticsDataType,
            global_step: Optional[int] = None,
            output_path: Optional[str] = None,
            kwargs: Optional[dict] = None,
    ):
        kwargs = kwargs if kwargs else {}
        if global_step:
            if not isinstance(global_step, int):
                raise TypeError(f"Expect global step to be an instance of int, but got {type(global_step)}")
            kwargs[TrackConst.GLOBAL_STEP_KEY] = global_step

        dxo = self.create_analytic_dxo(key=key, value=value, data_type=data_type, step=output_path)

        with self.engine.new_context() as fl_ctx:
            self.send_analytic_dxo(dxo=dxo, event_type=ANALYTIC_EVENT_TYPE)

    def send_analytic_dxo(self, dxo: DXO, event_type: str = ANALYTIC_EVENT_TYPE):
        """Sends analytic dxo.
        Args:
            dxo (DXO): analytic data in dxo.
            event_type (str): Event type.
        """
        if not isinstance(dxo, DXO):
            raise TypeError(f"expect dxo to be an instance of DXO, but got {type(dxo)}")
        self.fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=dxo.to_shareable(), private=True, sticky=False)
        self.fire_event(event_type=event_type, fl_ctx=self.fl_ctx)

    def create_analytic_dxo(
            self,
            key: str,
            value: any,
            data_type: AnalyticsDataType,
            step: Optional[int] = None,
            path: Optional[str] = None,
            kwargs: Optional[Dict] = None,
    ) -> DXO:
        """Creates the analytic DXO.
        Args:
            key (str): the tag associated with this value.
            value: the analytic data.
            data_type (AnalyticsDataType): analytic data type.
            step: Optional global step
            path: Optional output path, this is needed for log artifacts
            kwargs: Optional additional arguments to be passed into the receiver side's function.

        Returns:
            A DXO object that contains the analytic data.
        """
        data = AnalyticsData(tag=key, value=value, data_type=data_type,kwargs=kwargs,  step=step, path=path)
        dxo = data.to_dxo()
        return dxo
