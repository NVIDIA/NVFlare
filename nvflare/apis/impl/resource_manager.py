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

from abc import ABC

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable

from .scheduler_constants import AuxChannelTopic, FLContextKey, ShareableHeader


def _check_resource_handler(topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
    resource_manager = fl_ctx.get_prop(FLContextKey.RESOURCE_MANAGER)
    if not isinstance(resource_manager, ResourceManagerSpec):
        raise RuntimeError(
            f"resource_manager should be of type ResourceManagerSpec, but got {type(ResourceManagerSpec)}."
        )
    result = Shareable()
    try:
        resource_spec = request.get_header(ShareableHeader.RESOURCE_SPEC)
        check_result, token = resource_manager.check_resources(resource_requirement=resource_spec, fl_ctx=fl_ctx)
        result.set_header(ShareableHeader.CHECK_RESOURCE_RESULT, check_result)
        result.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, token)
    except Exception:
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    return result


def _cancel_resource_handler(topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
    resource_manager = fl_ctx.get_prop(FLContextKey.RESOURCE_MANAGER)
    if not isinstance(resource_manager, ResourceManagerSpec):
        raise RuntimeError(
            f"resource_manager should be of type ResourceManagerSpec, but got {type(ResourceManagerSpec)}."
        )
    result = Shareable()
    try:
        resource_spec = request.get_header(ShareableHeader.RESOURCE_SPEC)
        token = request.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
        resource_manager.cancel_resources(resource_requirement=resource_spec, token=token, fl_ctx=fl_ctx)
    except Exception:
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    return result


class ResourceManager(FLComponent, ResourceManagerSpec, ABC):
    def __init__(self):
        super().__init__()
        # TODO:: do we need to lock this
        self.initialized = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            # TODO:: where to do this init??
            if not self.initialized:
                engine = fl_ctx.get_engine()
                if not isinstance(engine, ClientEngineSpec) or not isinstance(engine, ServerEngineSpec):
                    raise RuntimeError(
                        f"engine inside fl_ctx should be of type ClientEngineSpec or ServerEngineSpec, but got {type(engine)}."
                    )
                fl_ctx.set_prop(FLContextKey.RESOURCE_MANAGER, self)
                engine.register_aux_message_handler(
                    topic=AuxChannelTopic.CHECK_RESOURCE, message_handle_func=_check_resource_handler
                )
                engine.register_aux_message_handler(
                    topic=AuxChannelTopic.CANCEL_RESOURCE, message_handle_func=_cancel_resource_handler
                )
