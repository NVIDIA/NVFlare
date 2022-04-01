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

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.scheduler_constants import AuxChannelTopic, ShareableHeader
from nvflare.apis.resource_manager_spec import ResourceConsumerSpec
from nvflare.apis.shareable import Shareable

# TODO:: GPU resource consumer
#     resource_consumer.consume(resources: dict)
#     -> set GPU ENV variable based on input allocate resources


def _stop_app_handler(topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
    engine = fl_ctx.get_engine()
    if not isinstance(engine, ClientEngineSpec):
        raise RuntimeError(f"engine inside fl_ctx should be of type ClientEngineSpec, but got {type(engine)}.")
    result = Shareable()
    try:
        app_name = request.get_header(ShareableHeader.APP_NAME)
        # TODO:: only ClientEngineInternalSpec has abort_app, and it does not take app_name as argument
        #   engine.abort_app()
    except Exception:
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    return result


def _dispatch_app_handler(topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
    engine = fl_ctx.get_engine()
    if not isinstance(engine, ClientEngineSpec):
        raise RuntimeError(f"engine inside fl_ctx should be of type ClientEngineSpec, but got {type(engine)}.")
    result = Shareable()
    try:
        app_name = request.get_header(ShareableHeader.APP_NAME)
        app_bytes = request.get_header(ShareableHeader.APP_BYTES)
        # TODO:: only ClientEngineInternalSpec has deploy_app ... should I copy the logic here
        #   or I required engine to be ClientEngineInternalSpec...
        # TODO:: only ClientEngineInternalSpec has start_app ... should I copy the logic here
        #   or I required engine to be ClientEngineInternalSpec...
    except Exception:
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    return result


class ResourceConsumer(ResourceConsumerSpec, FLComponent):
    def __init__(self):
        super().__init__()
        self.usable_resources = {}
        # TODO:: do we need to lock this
        self.initialized = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            # TODO:: where to do this init??
            if not self.initialized:
                engine = fl_ctx.get_engine()
                if not isinstance(engine, ClientEngineSpec):
                    raise RuntimeError(
                        f"engine inside fl_ctx should be of type ClientEngineSpec, but got {type(engine)}."
                    )
                engine.register_aux_message_handler(
                    topic=AuxChannelTopic.STOP_APP, message_handle_func=_stop_app_handler
                )
                engine.register_aux_message_handler(
                    topic=AuxChannelTopic.DISPATCH_APP, message_handle_func=_dispatch_app_handler
                )

    def consume(self, resources: dict):
        self.usable_resources = resources
