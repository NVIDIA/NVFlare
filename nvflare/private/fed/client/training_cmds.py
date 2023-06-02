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

import json
import os
from typing import List

from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci.proto import MetaStatusValue, make_meta
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.lighter.utils import verify_folder_signature
from nvflare.private.admin_defs import Message, error_reply, ok_reply
from nvflare.private.defs import RequestHeader, ScopeInfoKey, TrainingTopic
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.utils.fed_utils import get_scope_info


class StartAppProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.START]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        job_id = req.get_header(RequestHeader.JOB_ID)
        result = engine.start_app(job_id)
        if not result:
            result = "OK"
        return ok_reply(topic=f"reply_{req.topic}", body=result)


class AbortAppProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.ABORT]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        job_id = req.get_header(RequestHeader.JOB_ID)
        result = engine.abort_app(job_id)
        if not result:
            result = "OK"
        return ok_reply(topic=f"reply_{req.topic}", body=result)


class AbortTaskProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.ABORT_TASK]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        job_id = req.get_header(RequestHeader.JOB_ID)
        result = engine.abort_task(job_id)
        if not result:
            result = "OK"
        return ok_reply(topic=f"reply_{req.topic}", body=result, meta=make_meta(MetaStatusValue.OK, result))


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
        return ok_reply(topic=f"reply_{req.topic}", body=result)


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
        return ok_reply(topic=f"reply_{req.topic}", body=result)


class DeployProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.DEPLOY]

    def process(self, req: Message, app_ctx) -> Message:
        # Note: this method executes in the Main process of the client
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        job_id = req.get_header(RequestHeader.JOB_ID)
        job_meta = json.loads(req.get_header(RequestHeader.JOB_META))
        app_name = req.get_header(RequestHeader.APP_NAME)
        client_name = engine.get_client_name()

        if not job_meta:
            return error_reply("missing job meta")

        err = engine.deploy_app(
            app_name=app_name, job_id=job_id, job_meta=job_meta, client_name=client_name, app_data=req.body
        )
        if err:
            return error_reply(err)

        kv_list = parse_vars(engine.args.set)
        secure_train = kv_list.get("secure_train", True)
        if secure_train:
            workspace = Workspace(root_dir=engine.args.workspace, site_name=client_name)
            app_path = workspace.get_app_dir(job_id)
            root_ca_path = os.path.join(workspace.get_startup_kit_dir(), "rootCA.pem")
            if not verify_folder_signature(app_path, root_ca_path):
                return error_reply(f"app {app_name} does not pass signature verification")
        return ok_reply(body=f"deployed {app_name} to {client_name}")


class DeleteRunNumberProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.DELETE_RUN]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        job_id = req.get_header(RequestHeader.JOB_ID)
        result = engine.delete_run(job_id)
        if not result:
            result = "OK"
        return ok_reply(topic=f"reply_{req.topic}", body=result)


class ClientStatusProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.CHECK_STATUS]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        result = engine.get_engine_status()
        result = json.dumps(result)
        return ok_reply(topic=f"reply_{req.topic}", body=result)


class ScopeInfoProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.GET_SCOPES]

    def process(self, req: Message, app_ctx) -> Message:
        scope_names, default_scope_name = get_scope_info()
        result = {ScopeInfoKey.SCOPE_NAMES: scope_names, ScopeInfoKey.DEFAULT_SCOPE: default_scope_name}
        result = json.dumps(result)
        return ok_reply(topic=f"reply_{req.topic}", body=result)
