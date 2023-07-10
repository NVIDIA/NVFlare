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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.security import Security


class SecurityHandler(Security):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SECURITY_CHECK:
            engine = fl_ctx.get_engine()

            for id, component in engine.get_components().items():
                if isinstance(component, Security):
                    result, reason = component.authenticate(fl_ctx=fl_ctx)
                    if not result:
                        fl_ctx.set_prop(FLContextKey.AUTHENTICATION_RESULT, False, sticky=False)
                        fl_ctx.set_prop(FLContextKey.AUTHENTICATION_REASON, {id: reason}, sticky=False)

                    result, reason = component.authorize(fl_ctx=fl_ctx)
                    if not result:
                        fl_ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, False, sticky=False)
                        fl_ctx.set_prop(FLContextKey.AUTHORIZATION_REASON, {id: reason}, sticky=False)

