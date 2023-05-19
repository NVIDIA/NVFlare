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

from nvflare.private.admin_defs import Message
from nvflare.private.defs import ComponentCallerTopic, RequestHeader
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.widgets.comp_caller import ComponentCaller
from nvflare.widgets.widget import WidgetID


class ComponentCallerProcessor(RequestProcessor):
    def get_topics(self) -> [str]:
        return [ComponentCallerTopic.CALL_COMPONENT]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        caller = engine.get_widget(WidgetID.COMPONENT_CALLER)
        if not isinstance(caller, ComponentCaller):
            raise TypeError("caller must be ComponentCaller, but got {}".format(type(caller)))

        run_info = engine.get_current_run_info()
        if not run_info or run_info.job_id < 0:
            result = {"error": "app not running"}
        else:
            comp_target = req.get_header(RequestHeader.COMPONENT_TARGET)
            call_name = req.get_header(RequestHeader.CALL_NAME)
            call_params = req.body
            result = caller.call_components(target=comp_target, call_name=call_name, params=call_params)

        if not isinstance(result, dict):
            result = {}

        return Message(topic=req.topic, body=result)
