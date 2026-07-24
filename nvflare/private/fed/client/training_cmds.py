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
import logging
import os
import tempfile
from typing import List

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci.proto import MetaStatusValue, make_meta
from nvflare.fuel.utils.zip_utils import unzip_all_from_bytes
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE
from nvflare.lighter.utils import verify_folder_signature
from nvflare.private.admin_defs import Message, error_reply, ok_reply
from nvflare.private.defs import RequestHeader, ScopeInfoKey, TrainingTopic
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.utils.fed_utils import get_scope_info, require_signed_jobs
from nvflare.security.logging import secure_format_exception

logger = logging.getLogger(__name__)


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
        job_meta = req.get_header(RequestHeader.JOB_META)
        app_name = req.get_header(RequestHeader.APP_NAME)
        client_name = engine.get_client_name()

        if not job_meta:
            return error_reply("missing job meta")

        from_hub_site = job_meta.get(JobMetaKey.FROM_HUB_SITE.value)
        if not from_hub_site:
            workspace = Workspace(root_dir=engine.args.workspace, site_name=client_name)
            root_ca_path = os.path.join(workspace.get_startup_kit_dir(), "rootCA.pem")
            # Verify the received bytes before deploying them. AppDeployer will
            # extract these same bytes into the run directory.
            with tempfile.TemporaryDirectory() as app_staging_dir:
                try:
                    unzip_all_from_bytes(req.body, app_staging_dir)
                except Exception as e:
                    logger.warning("failed to stage app %s: %s", app_name, secure_format_exception(e))
                    return error_reply(f"failed to stage app {app_name}")

                sig_file = os.path.join(app_staging_dir, NVFLARE_SIG_FILE)
                has_root_ca = os.path.exists(root_ca_path)
                if os.path.exists(sig_file) and has_root_ca:
                    if not verify_folder_signature(app_staging_dir, root_ca_path):
                        return error_reply(f"app {app_name} does not pass signature verification")
                elif has_root_ca and require_signed_jobs(workspace, WorkspaceConstants.CLIENT_STARTUP_CONFIG):
                    return error_reply("unsigned job rejected - signed deploy is required")

        err = engine.deploy_app(
            app_name=app_name, job_id=job_id, job_meta=job_meta, client_name=client_name, app_data=req.body
        )
        if err:
            return error_reply(err)

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


class ConfigureJobLogProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [TrainingTopic.CONFIGURE_JOB_LOG]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))

        fl_ctx = engine.new_context()
        site_name = fl_ctx.get_identity_name()
        job_id = req.get_header(RequestHeader.JOB_ID)

        err = engine.configure_job_log(job_id, req.body)
        if err:
            return error_reply(err)

        return ok_reply(topic=f"reply_{req.topic}", body=f"successfully configured {site_name} job {job_id} log")


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


class NotifyJobStatusProcessor(RequestProcessor):
    def get_topics(self) -> [str]:
        return [TrainingTopic.NOTIFY_JOB_STATUS]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        if not isinstance(engine, ClientEngineInternalSpec):
            raise TypeError("engine must be ClientEngineInternalSpec, but got {}".format(type(engine)))
        job_id = req.get_header(RequestHeader.JOB_ID)
        job_status = req.get_header(RequestHeader.JOB_STATUS)
        engine.notify_job_status(job_id, job_status)
        return ok_reply(topic=f"reply_{req.topic}", body=f"notify status: {job_status}")
