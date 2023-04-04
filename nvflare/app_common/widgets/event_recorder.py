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

import logging
import os
from typing import Dict

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable
from nvflare.widgets.widget import Widget


class _CtxPropReq(object):
    """Requirements of a prop in the FLContext.

    Arguments:
        dtype: data type of the prop.
        is_private: if this prop is private.
        is_sticky: if this prop is sticky.
        allow_none: if this prop can be None
    """

    def __init__(self, dtype, is_private, is_sticky, allow_none: bool = False):
        self.dtype = dtype
        self.is_private = is_private
        self.is_sticky = is_sticky
        self.allow_none = allow_none


class _EventReq(object):
    """Requirements for FL and peer context when an event is fired.

    Arguments:
        ctx_reqs: A dictionary that describes the requirements for fl_ctx. It maps property names to _CtxPropReq
        peer_ctx_reqs: A dictionary that describes the requirements for peer_ctx. It maps property names to _CtxPropReq
    """

    def __init__(
        self,
        ctx_reqs: Dict[str, _CtxPropReq],
        peer_ctx_reqs: Dict[str, _CtxPropReq],
        ctx_block_list: [str] = None,
        peer_ctx_block_list: [str] = None,
    ):
        self.ctx_reqs = ctx_reqs  # prop name => _CtxPropReq
        self.peer_ctx_reqs = peer_ctx_reqs

        if ctx_block_list is None:
            ctx_block_list = []

        if peer_ctx_block_list is None:
            peer_ctx_block_list = []

        self.ctx_block_list = ctx_block_list
        self.peer_ctx_block_list = peer_ctx_block_list


class _EventStats(object):
    """Stats of each event."""

    def __init__(self):
        self.call_count = 0
        self.prop_missing = 0
        self.prop_none_value = 0
        self.prop_dtype_mismatch = 0
        self.prop_attr_mismatch = 0
        self.prop_block_list_violation = 0
        self.peer_ctx_missing = 0


class EventRecorder(Widget):

    _KEY_CTX_TYPE = "ctx_type"
    _KEY_EVENT_TYPE = "event_type"
    _KEY_EVENT_STATS = "event_stats"
    _KEY_EVENT_REQ = "event_req"

    def __init__(self, log_file_name=None):
        """A component to record all system-wide events.

        Args:
            log_file_name (str, optional): the log filename to save recorded events. Defaults to None.
        """
        super().__init__()

        all_ctx_reqs = {
            "__run_num__": _CtxPropReq(dtype=str, is_private=False, is_sticky=True),
            "__identity_name__": _CtxPropReq(dtype=str, is_private=False, is_sticky=True),
        }

        run_req = _EventReq(ctx_reqs=all_ctx_reqs, peer_ctx_reqs={})
        self.event_reqs = {EventType.START_RUN: run_req, EventType.END_RUN: run_req}  # event type => _EventReq
        self.event_stats = {}  # event_type => _EventStats
        self._log_handler_added = False
        self.log_file_name = log_file_name if log_file_name else "event_recorded.txt"

    def event_tag(self, fl_ctx: FLContext):
        event_type = fl_ctx.get_prop(self._KEY_EVENT_TYPE, "?")
        event_id = fl_ctx.get_prop(FLContextKey.EVENT_ID, None)
        if event_id:
            return "[type={}, id={}]".format(event_type, event_id)
        else:
            return "[{}]".format(event_type)

    def event_error_tag(self, fl_ctx: FLContext):
        ctx_type = fl_ctx.get_prop(self._KEY_CTX_TYPE, "?")
        return "Event {}: in {},".format(self.event_tag(fl_ctx), ctx_type)

    def validate_prop(self, prop_name: str, req: _CtxPropReq, fl_ctx: FLContext):
        stats = fl_ctx.get_prop(self._KEY_EVENT_STATS, None)

        detail = fl_ctx.get_prop_detail(prop_name)
        if not isinstance(detail, dict):
            stats.prop_missing += 1
            self.logger.error("{} required prop '{}' doesn't exist".format(self.event_error_tag(fl_ctx), prop_name))
            return

        value = detail["value"]
        if value is None and not req.allow_none:
            stats.prop_none_value += 1
            self.logger.error(
                "{} prop '{}' is None, but None is not allowed".format(self.event_error_tag(fl_ctx), prop_name)
            )

        if req.dtype is not None:
            if not isinstance(value, req.dtype):
                stats.prop_dtype_mismatch += 1
                self.logger.error(
                    "{} prop '{}' should be {}, but got {}".format(
                        self.event_error_tag(fl_ctx), prop_name, req.dtype, type(value)
                    )
                )

        if req.is_private and not detail["private"]:
            stats.prop_attr_mismatch += 1
            self.logger.error(
                "{} prop '{}' should be private but is public".format(self.event_error_tag(fl_ctx), prop_name)
            )

        if req.is_private is not None and not req.is_private and detail["private"]:
            stats.prop_attr_mismatch += 1
            self.logger.error(
                "{} prop '{}' should be public but is private".format(self.event_error_tag(fl_ctx), prop_name)
            )

        if req.is_sticky and not detail["sticky"]:
            stats.prop_attr_mismatch += 1
            self.logger.error(
                "{} prop '{}' should be sticky but is non-sticky".format(self.event_error_tag(fl_ctx), prop_name)
            )

        if req.is_sticky is not None and not req.is_sticky and detail["sticky"]:
            stats.prop_attr_mismatch += 1
            self.logger.error(
                "{} prop '{}' should be non-sticky but is sticky".format(self.event_error_tag(fl_ctx), prop_name)
            )

    def check_block_list(self, block_list, fl_ctx: FLContext):
        stats = fl_ctx.get_prop(self._KEY_EVENT_STATS, None)
        for prop_name in block_list:
            detail = fl_ctx.get_prop_detail(prop_name)
            if detail:
                stats.prop_block_list_violation += 1
                self.logger.error("{} prop {} is not expected".format(self.event_error_tag(fl_ctx), prop_name))

    def check_props(self, fl_ctx: FLContext):
        event_req = fl_ctx.get_prop(self._KEY_EVENT_REQ)
        stats = fl_ctx.get_prop(self._KEY_EVENT_STATS)

        for prop_name, req in event_req.ctx_reqs.items():
            self.validate_prop(prop_name, req, fl_ctx)

        self.check_block_list(event_req.ctx_block_list, fl_ctx)

        if event_req.peer_ctx_reqs:
            peer_ctx = fl_ctx.get_peer_context()
            if not peer_ctx:
                stats.peer_ctx_missing += 1
                self.logger.error("{} expected peer_ctx not present".format(self.event_error_tag(fl_ctx)))
            else:
                for prop_name, req in event_req.peer_ctx_reqs.items():
                    self.validate_prop(prop_name, req, peer_ctx)
                self.check_block_list(event_req.peer_ctx_block_list, peer_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if not self._log_handler_added:
            workspace = fl_ctx.get_engine().get_workspace()
            app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
            output_file_handler = logging.FileHandler(os.path.join(app_dir, self.log_file_name))
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            output_file_handler.setFormatter(formatter)
            self.logger.addHandler(output_file_handler)
            self._log_handler_added = True
        event_stats = self.event_stats.get(event_type, None)
        if not event_stats:
            event_stats = _EventStats()
            self.event_stats[event_type] = event_stats

        fl_ctx.set_prop(key=self._KEY_EVENT_STATS, value=event_stats, private=True, sticky=False)
        fl_ctx.set_prop(key=self._KEY_EVENT_TYPE, value=event_type, private=True, sticky=False)
        fl_ctx.set_prop(key=self._KEY_CTX_TYPE, value="fl_ctx", private=True, sticky=False)

        self.log_info(fl_ctx, "Got event {}".format(self.event_tag(fl_ctx)), fire_event=False)
        event_stats.call_count += 1

        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx:
            event_id = fl_ctx.get_prop(FLContextKey.EVENT_ID)
            peer_ctx.set_prop(key=FLContextKey.EVENT_ID, value=event_id, private=True, sticky=False)
            peer_ctx.set_prop(key=self._KEY_EVENT_STATS, value=event_stats, private=True, sticky=False)
            peer_ctx.set_prop(key=self._KEY_EVENT_TYPE, value=event_type, private=True, sticky=False)
            peer_ctx.set_prop(key=self._KEY_CTX_TYPE, value="peer_ctx", private=True, sticky=False)
            self.log_info(
                fl_ctx, "Peer Context for event {}: {}".format(self.event_tag(fl_ctx), peer_ctx), fire_event=False
            )

        event_req = self.event_reqs.get(event_type, None)
        fl_ctx.set_prop(key=self._KEY_EVENT_REQ, value=event_req, private=True, sticky=False)
        if event_req:
            self.check_props(fl_ctx)

        if event_type == EventType.END_RUN:
            # print stats
            for e, s in self.event_stats.items():
                self.log_info(fl_ctx, "Stats of {}: {}".format(e, vars(s)), fire_event=False)


class ServerEventRecorder(EventRecorder):
    def __init__(self):
        """Server-specific event recorder."""
        super().__init__()

        task_data_filter_reqs = _EventReq(
            ctx_reqs={
                "__engine__": _CtxPropReq(dtype=ServerEngineSpec, is_private=True, is_sticky=True),
                FLContextKey.TASK_ID: _CtxPropReq(dtype=str, is_private=True, is_sticky=False),
                FLContextKey.TASK_NAME: _CtxPropReq(dtype=str, is_private=True, is_sticky=False),
                FLContextKey.TASK_DATA: _CtxPropReq(dtype=Shareable, is_private=True, is_sticky=False, allow_none=True),
                "testPrivateServerSticky": _CtxPropReq(dtype=str, is_private=True, is_sticky=True),
                "testPublicServerSticky": _CtxPropReq(dtype=str, is_private=False, is_sticky=True),
            },
            ctx_block_list=[
                "testPrivateServerNonSticky",
                "testPublicServerNonSticky",
                "testPrivateClientNonSticky",
                "testPublicClientNonSticky",
                "testPrivateClientSticky",
                "testPublicClientSticky",
            ],
            peer_ctx_reqs={
                "__run_num__": _CtxPropReq(dtype=str, is_private=None, is_sticky=None),
                "__identity_name__": _CtxPropReq(dtype=str, is_private=None, is_sticky=None),
                "testPublicClientSticky": _CtxPropReq(dtype=str, is_private=None, is_sticky=None),
            },
            peer_ctx_block_list=[
                "__engine__",
                "testPrivateClientSticky",
                "testPrivateClientNonSticky",
                "testPublicClientNonSticky",
            ],
        )
        self.event_reqs.update(
            {
                EventType.BEFORE_TASK_DATA_FILTER: task_data_filter_reqs,
                EventType.AFTER_TASK_DATA_FILTER: task_data_filter_reqs,
            }
        )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            fl_ctx.set_prop(
                key="testPrivateServerSticky", value="this is a server private sticky", private=True, sticky=True
            )

            fl_ctx.set_prop(
                key="testPublicServerSticky", value="this is a server public sticky", private=False, sticky=True
            )

            fl_ctx.set_prop(
                key="testPrivateServerNonSticky",
                value="this is a server private non-sticky",
                private=True,
                sticky=False,
            )

            fl_ctx.set_prop(
                key="testPublicServerNonSticky", value="this is a server public non-sticky", private=False, sticky=False
            )

        super().handle_event(event_type, fl_ctx)


class ClientEventRecorder(EventRecorder):
    def __init__(self):
        """Client-specific event recorder."""
        super().__init__()

        task_data_filter_reqs = _EventReq(
            ctx_reqs={
                "__engine__": _CtxPropReq(dtype=ClientEngineSpec, is_private=True, is_sticky=True),
                FLContextKey.TASK_ID: _CtxPropReq(dtype=str, is_private=True, is_sticky=False),
                FLContextKey.TASK_NAME: _CtxPropReq(dtype=str, is_private=True, is_sticky=False),
                FLContextKey.TASK_DATA: _CtxPropReq(dtype=Shareable, is_private=True, is_sticky=False, allow_none=True),
                "testPrivateClientSticky": _CtxPropReq(dtype=str, is_private=True, is_sticky=True),
                "testPublicClientSticky": _CtxPropReq(dtype=str, is_private=False, is_sticky=True),
            },
            ctx_block_list=[
                "testPrivateServerNonSticky",
                "testPublicServerNonSticky",
                "testPrivateClientNonSticky",
                "testPublicClientNonSticky",
                "testPrivateServerSticky",
                "testPublicServerSticky",
            ],
            peer_ctx_reqs={
                "__run_num__": _CtxPropReq(dtype=str, is_private=None, is_sticky=None),
                "__identity_name__": _CtxPropReq(dtype=str, is_private=None, is_sticky=None),
                "testPublicServerSticky": _CtxPropReq(dtype=str, is_private=None, is_sticky=None),
            },
            peer_ctx_block_list=[
                "__engine__",
                "testPrivateServerSticky",
                "testPrivateServerNonSticky",
                "testPublicServerNonSticky",
            ],
        )
        self.event_reqs.update(
            {
                EventType.BEFORE_TASK_DATA_FILTER: task_data_filter_reqs,
                EventType.AFTER_TASK_DATA_FILTER: task_data_filter_reqs,
            }
        )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            fl_ctx.set_prop(
                key="testPrivateClientSticky", value="this is a client private sticky", private=True, sticky=True
            )

            fl_ctx.set_prop(
                key="testPublicClientSticky", value="this is a client public sticky", private=False, sticky=True
            )

            fl_ctx.set_prop(
                key="testPrivateClientNonSticky",
                value="this is a client private non-sticky",
                private=True,
                sticky=False,
            )

            fl_ctx.set_prop(
                key="testPublicClientNonSticky", value="this is a client public non-sticky", private=False, sticky=False
            )

        super().handle_event(event_type, fl_ctx)
