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

import json
from typing import List

from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec

from .admin import RequestProcessor


class StartAppProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.START]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        run_number = int(req.get_header(RequestHeader.RUN_NUM))
        result = engine.start_app(run_number)
        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class AbortAppProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.ABORT]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        run_number = int(req.get_header(RequestHeader.RUN_NUM))
        result = engine.abort_app(run_number)
        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class AbortTaskProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.ABORT_TASK]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        run_number = int(req.get_header(RequestHeader.RUN_NUM))
        result = engine.abort_task(run_number)
        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class ShutdownClientProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.SHUTDOWN]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        result = engine.shutdown()
        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class RestartClientProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.RESTART]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        result = engine.restart()
        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class DeployProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.DEPLOY]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        run_number = req.get_header(RequestHeader.RUN_NUM)
        app_name = req.get_header(RequestHeader.APP_NAME)
        client_name = engine.get_client_name()
        result = engine.deploy_app(app_name=app_name, run_num=run_number, client_name=client_name, app_data=req.body)
        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)


class DeleteRunNumberProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.DELETE_RUN]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        run_number = int(req.get_header(RequestHeader.RUN_NUM))
        result = engine.delete_run(run_number)
        if not result:
            result = "OK"
        message = Message(topic="reply_" + req.topic, body=result)
        return message


class ClientStatusProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.CHECK_STATUS]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        result = engine.get_engine_status()
        # run_info = engine.get_current_run_info()
        # if not run_info or run_info.run_number < 0:
        #     result = {
        #         ClientStatusKey.RUN_NUM: 'none',
        #         ClientStatusKey.CURRENT_TASK: 'none'
        #     }
        # else:
        #     result = {
        #         ClientStatusKey.RUN_NUM: str(run_info.run_number),
        #         ClientStatusKey.CURRENT_TASK: run_info.current_task_name
        #     }
        result = json.dumps(result)
        message = Message(topic="reply_" + req.topic, body=result)
        return message


class SetRunNumberProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.SET_RUN_NUMBER]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        run_number = int(req.get_header(RequestHeader.RUN_NUM))
        result = engine.set_run_number(run_number)
        if not result:
            result = "OK"
        return Message(topic="reply_" + req.topic, body=result)
