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

import json
from typing import List

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode, SystemComponents
from nvflare.apis.resource_manager_spec import ResourceConsumerSpec, ResourceManagerSpec
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils import fobs
from nvflare.private.admin_defs import Message
from nvflare.private.defs import ERROR_MSG_PREFIX, RequestHeader, SysCommandTopic, TrainingTopic
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.scheduler_constants import ShareableHeader
from nvflare.security.logging import secure_format_exception


def _get_resource_manager(engine: ClientEngineInternalSpec):
    if not isinstance(engine, ClientEngineInternalSpec):
        raise ValueError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

    resource_manager = engine.get_component(SystemComponents.RESOURCE_MANAGER)
    if not isinstance(resource_manager, ResourceManagerSpec):
        raise ValueError(f"resource_manager should be of type ResourceManagerSpec, but got {type(resource_manager)}.")

    return resource_manager


def _get_resource_consumer(engine: ClientEngineInternalSpec):
    if not isinstance(engine, ClientEngineInternalSpec):
        raise ValueError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

    resource_consumer = engine.get_component(SystemComponents.RESOURCE_CONSUMER)
    if not isinstance(resource_consumer, ResourceConsumerSpec):
        raise ValueError(
            f"resource_consumer should be of type ResourceConsumerSpec, but got {type(resource_consumer)}."
        )

    return resource_consumer


class CheckResourceProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.CHECK_RESOURCE]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        result = Shareable()
        resource_manager = _get_resource_manager(engine)
        is_resource_enough, token = False, ""

        with engine.new_context() as fl_ctx:
            try:
                job_id = req.get_header(RequestHeader.JOB_ID, "")
                resource_spec = fobs.loads(req.body)
                fl_ctx.set_prop(key=FLContextKey.CLIENT_RESOURCE_SPECS, value=resource_spec, private=True, sticky=False)

                fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job_id, private=True, sticky=False)

                engine.fire_event(EventType.BEFORE_CHECK_RESOURCE_MANAGER, fl_ctx)
                block_reason = fl_ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)
                if block_reason:
                    is_resource_enough = False
                    token = block_reason
                else:
                    is_resource_enough, token = resource_manager.check_resources(
                        resource_requirement=resource_spec, fl_ctx=fl_ctx
                    )
            except Exception:
                result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)

        result.set_header(ShareableHeader.IS_RESOURCE_ENOUGH, is_resource_enough)
        result.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, token)

        return Message(topic="reply_" + req.topic, body=fobs.dumps(result))


class StartJobProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.START_JOB]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        resource_manager = _get_resource_manager(engine)

        allocated_resources = None
        try:
            resource_spec = fobs.loads(req.body)
            job_id = req.get_header(RequestHeader.JOB_ID)
            token = req.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
        except Exception as e:
            msg = f"{ERROR_MSG_PREFIX}: Start job execution exception, missing required information: {secure_format_exception(e)}."
            return Message(topic=f"reply_{req.topic}", body=msg)

        try:
            with engine.new_context() as fl_ctx:
                allocated_resources = resource_manager.allocate_resources(
                    resource_requirement=resource_spec, token=token, fl_ctx=fl_ctx
                )
            if allocated_resources:
                resource_consumer = _get_resource_consumer(engine)
                resource_consumer.consume(allocated_resources)
            result = engine.start_app(
                job_id,
                allocated_resource=allocated_resources,
                token=token,
                resource_manager=resource_manager,
            )
        except Exception as e:
            result = f"{ERROR_MSG_PREFIX}: Start job execution exception: {secure_format_exception(e)}."
            if allocated_resources:
                with engine.new_context() as fl_ctx:
                    resource_manager.free_resources(resources=allocated_resources, token=token, fl_ctx=fl_ctx)

        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class CancelResourceProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.CANCEL_RESOURCE]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        result = Shareable()
        resource_manager = _get_resource_manager(engine)

        with engine.new_context() as fl_ctx:
            try:
                resource_spec = fobs.loads(req.body)
                token = req.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
                resource_manager.cancel_resources(resource_requirement=resource_spec, token=token, fl_ctx=fl_ctx)
            except Exception:
                result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)

        return Message(topic="reply_" + req.topic, body=fobs.dumps(result))


class ReportResourcesProcessor(RequestProcessor):
    def get_topics(self) -> [str]:
        return [SysCommandTopic.REPORT_RESOURCES]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        resource_manager = _get_resource_manager(engine)
        resources = resource_manager.report_resources(engine.new_context())
        message = Message(topic="reply_" + req.topic, body=json.dumps(resources))
        return message
