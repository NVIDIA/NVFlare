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
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant


class SecurityHandler(FLComponent):
    def _process_before_broadcast(self, fl_ctx: FLContext):
        pass

    def _process_after_broadcast(self, fl_ctx: FLContext):
        pass

    def _process_before_all_gather_v(self, fl_ctx: FLContext):
        pass

    def _process_after_all_gather_v(self, fl_ctx: FLContext):
        pass

    def _format_msg(self, fl_ctx: FLContext, msg: str):
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK, "?")
        seq = fl_ctx.get_prop(Constant.PARAM_KEY_SEQ, "?")
        root = fl_ctx.get_prop(Constant.PARAM_KEY_ROOT)
        event = fl_ctx.get_prop(Constant.PARAM_KEY_EVENT, "?")
        if root:
            return f"[{event}: {rank=} {seq=} {root=}] {msg}"
        else:
            return f"[{event}: {rank=} {seq=}] {msg}"

    def info(self, fl_ctx: FLContext, msg: str):
        self.log_info(fl_ctx, self._format_msg(fl_ctx, msg), fire_event=False)

    def debug(self, fl_ctx: FLContext, msg: str):
        self.log_debug(fl_ctx, self._format_msg(fl_ctx, msg), fire_event=False)

    def error(self, fl_ctx: FLContext, msg: str):
        self.log_error(fl_ctx, self._format_msg(fl_ctx, msg), fire_event=False)

    def _abort(self, error: str, fl_ctx: FLContext):
        fl_ctx.set_prop(FLContextKey.FATAL_SYSTEM_ERROR, error, private=True)
        self.fire_event(Constant.EVENT_XGB_ABORTED, fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        fl_ctx.set_prop(key=Constant.PARAM_KEY_EVENT, value=event_type, private=True, sticky=False)
        if event_type == Constant.EVENT_BEFORE_BROADCAST:
            self._process_before_broadcast(fl_ctx)
        elif event_type == Constant.EVENT_AFTER_BROADCAST:
            self._process_after_broadcast(fl_ctx)
        elif event_type == Constant.EVENT_BEFORE_ALL_GATHER_V:
            self._process_before_all_gather_v(fl_ctx)
        elif event_type == Constant.EVENT_AFTER_ALL_GATHER_V:
            self._process_after_all_gather_v(fl_ctx)
