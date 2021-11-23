# Copyright (c) 2021, NVIDIA CORPORATION.
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

import datetime

from nvflare.apis.analytix import Data as AnalytixData
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

from .widget import Widget


class GroupInfoCollector(object):
    def __init__(self):
        self.info = {}

    def set_info(self, group_name: str, info: dict):
        self.info[group_name] = info

    def add_info(self, group_name: str, info: dict):
        if group_name not in self.info:
            self.info[group_name] = info
        else:
            self.info[group_name].update(info)


class InfoCollector(Widget):

    """

    Info Structure:

        category (dict)
            group (dict)
                key/value (dict)

    """

    CATEGORY_STATS = "stats"
    CATEGORY_ERROR = "error"

    EVENT_TYPE_GET_STATS = "info_collector.get_stats"
    CTX_KEY_STATS_COLLECTOR = "info_collector.stats_collector"

    def __init__(self):
        Widget.__init__(self)
        self.categories = {}
        self.engine = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.reset_all()
            self.engine = fl_ctx.get_engine()
        elif event_type == EventType.END_RUN:
            self.engine = None
        elif event_type in (
            EventType.ERROR_LOG_AVAILABLE,
            EventType.WARNING_LOG_AVAILABLE,
            EventType.EXCEPTION_LOG_AVAILABLE,
        ):
            origin = fl_ctx.get_prop(FLContextKey.EVENT_ORIGIN, None)
            if origin:
                group_name = str(origin)
            else:
                group_name = "general"

            data = fl_ctx.get_prop(FLContextKey.EVENT_DATA, None)
            if not isinstance(data, Shareable):
                # not a valid error report
                self.log_error(
                    fl_ctx=fl_ctx,
                    msg="wrong event data type for event {}: expect Shareable but got {}".format(
                        event_type, type(data)
                    ),
                    fire_event=False,
                )
                return

            try:
                dxo = from_shareable(data)
            except:
                self.log_exception(
                    fl_ctx=fl_ctx, msg="invalid event data type for event {}".format(event_type), fire_event=False
                )
                return

            analytic_data = AnalytixData.from_dxo(dxo)

            if event_type == EventType.ERROR_LOG_AVAILABLE:
                key = "error"
            elif event_type == EventType.WARNING_LOG_AVAILABLE:
                key = "warning"
            else:
                key = "exception"

            self.add_error(group_name=group_name, key=key, err=analytic_data.value)

    def get_run_stats(self):
        self.reset_category(self.CATEGORY_STATS)

        # NOTE: it's important to assign self.engine to a new var!
        # This is because another thread may fire the END_RUN event, which will cause
        # self.engine to be set to None, just after checking it being None and before using it!
        engine = self.engine
        if not engine:
            return None

        # NOTE: we need a new context here to make sure all sticky props are copied!
        # We create a new StatusCollector to hold status info.
        # Do not use the InfoCollector itself for thread safety - multiple calls to
        # this method (from parallel admin commands) are possible at the same time!
        with self.engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            coll = GroupInfoCollector()
            fl_ctx.set_prop(key=self.CTX_KEY_STATS_COLLECTOR, value=coll, sticky=False, private=True)

            engine.fire_event(event_type=self.EVENT_TYPE_GET_STATS, fl_ctx=fl_ctx)
            # Get the StatusCollector from the fl_ctx, it could have been updated by other component.
            coll = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR)
            return coll.info

    def add_info(self, category: str, group_name: str, key: str, value):
        cat = self.categories.get(category, None)
        if not cat:
            cat = dict()
            self.categories[category] = cat
        group = cat.get(group_name, None)
        if not group:
            group = dict()
            cat[group_name] = group
        group[key] = value

    def set_info(self, category: str, group_name: str, info: dict):
        cat = self.categories.get(category, None)
        if not cat:
            cat = dict()
            self.categories[category] = cat
        cat[group_name] = info

    def get_category(self, category: str):
        return self.categories.get(category, None)

    def get_group(self, category: str, group_name: str):
        cat = self.categories.get(category, None)
        if not cat:
            return None
        return cat.get(group_name, None)

    def reset_all(self):
        self.categories = {}

    def reset_category(self, category: str):
        self.categories[category] = {}

    def reset_group(self, category: str, group_name: str):
        cat = self.categories.get(category, None)
        if not cat:
            return
        cat.get[group_name] = {}

    # some convenience methods
    def add_error(self, group_name: str, key: str, err: str):
        now = datetime.datetime.now()
        value = "{}: {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), err)

        self.add_info(category=self.CATEGORY_ERROR, group_name=group_name, key=key, value=value)

    def get_errors(self):
        return self.get_category(self.CATEGORY_ERROR)

    def reset_errors(self):
        self.reset_category(self.CATEGORY_ERROR)
