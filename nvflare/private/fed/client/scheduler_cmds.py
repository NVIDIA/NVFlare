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

import pickle
from typing import List

from nvflare.apis.fl_constant import ReturnCode, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceConsumerSpec, ResourceManagerSpec
from nvflare.apis.scheduler_constants import ShareableHeader
from nvflare.apis.shareable import Shareable
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec


class CheckResourceProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.CHECK_RESOURCE]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        resource_manager = engine.get_component(SystemComponents.RESOURCE_MANAGER)
        if not isinstance(resource_manager, ResourceManagerSpec):
            raise RuntimeError(
                f"resource_manager should be of type ResourceManagerSpec, but got {type(resource_manager)}."
            )
        fl_ctx = FLContext()
        result = Shareable()
        try:
            resource_spec = pickle.loads(req.body)
            check_result, token = resource_manager.check_resources(resource_requirement=resource_spec, fl_ctx=fl_ctx)
            result.set_header(ShareableHeader.CHECK_RESOURCE_RESULT, check_result)
            result.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, token)
        except Exception:
            result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)

        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=pickle.dumps(result))


class StartJobProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.START_JOB]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        resource_manager = engine.get_component(SystemComponents.RESOURCE_MANAGER)
        if not isinstance(resource_manager, ResourceManagerSpec):
            raise RuntimeError(
                f"resource_manager should be of type ResourceManagerSpec, but got {type(resource_manager)}."
            )
        resource_consumer = engine.get_component(SystemComponents.RESOURCE_CONSUMER)
        if not isinstance(resource_consumer, ResourceConsumerSpec):
            raise RuntimeError(
                f"resource_consumer should be of type ResourceConsumerSpec, but got {type(resource_consumer)}."
            )

        resource_spec = pickle.loads(req.body)
        run_number = req.get_header(RequestHeader.RUN_NUM)
        token = req.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
        allocated_resources = resource_manager.allocate_resources(
            resource_requirement=resource_spec, token=token, fl_ctx=FLContext()
        )
        result = engine.start_app(
            run_number,
            allocated_resource=allocated_resources,
            token=token,
            resource_consumer=resource_consumer,
            resource_manager=resource_manager,
        )

        resource_consumer.consume(allocated_resources)

        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class CancelResourceProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.CANCEL_RESOURCE]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        resource_manager = engine.get_component(SystemComponents.RESOURCE_MANAGER)
        if not isinstance(resource_manager, ResourceManagerSpec):
            raise RuntimeError(
                f"resource_manager should be of type ResourceManagerSpec, but got {type(resource_manager)}."
            )
        with engine.new_context() as fl_ctx:
            result = Shareable()
            try:
                # resource_spec = req.get_header(ShareableHeader.RESOURCE_SPEC)
                resource_spec = pickle.loads(req.body)
                token = req.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
                resource_manager.cancel_resources(resource_requirement=resource_spec, token=token, fl_ctx=fl_ctx)
            except Exception:
                result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)

        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=pickle.dumps(result))
