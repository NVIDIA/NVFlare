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

from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.data_event.utils import get_scope_property
from nvflare.fuel.hci.proto import MetaStatusValue, make_meta
from nvflare.fuel.sec.job_trust import JOB_AUTHORIZATION_CLOCK_SKEW, get_job_authorization, verify_job_authorization
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_module_logger
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE
from nvflare.lighter.utils import verify_folder_signature
from nvflare.private.admin_defs import Message, error_reply, ok_reply
from nvflare.private.defs import RequestHeader, ScopeInfoKey, TrainingTopic
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.utils.fed_utils import get_scope_info

logger = get_module_logger()


def _expected_server_identity(client_name: str) -> str:
    startup_config = ConfigService.get_section(SystemConfigs.STARTUP_CONF) or {}
    servers = startup_config.get("servers") if isinstance(startup_config, dict) else None
    if isinstance(servers, list) and servers:
        server = servers[0]
        if isinstance(server, dict):
            identity = str(server.get("identity") or "")
            if identity:
                return identity

    # The provision was done with an old version that has no server "identity" in the config.
    # To be backward compatible, we expect the identity to be the server host we connected to,
    # the same way client registration does (see communicator.py).
    expected_host = get_scope_property(scope_name=client_name, key=FLContextKey.SERVER_HOST_NAME)
    if expected_host:
        return str(expected_host)

    raise RuntimeError("expected server identity is not configured")


def _startup_config() -> dict:
    config = ConfigService.get_section(SystemConfigs.STARTUP_CONF)
    return config if isinstance(config, dict) else {}


def _job_authorization_clock_skew() -> float:
    """Clock skew (seconds) allowed when verifying server job authorization.

    Optional "job_authorization_clock_skew" in fed_client.json; must be a positive number.
    Invalid or non-positive values fall back to the default with a warning.
    """
    value = _startup_config().get("job_authorization_clock_skew")
    if value is None:
        return JOB_AUTHORIZATION_CLOCK_SKEW
    try:
        skew = float(value)
    except (TypeError, ValueError):
        skew = -1.0
    if skew <= 0:
        logger.warning(
            f"invalid job_authorization_clock_skew '{value}' in fed_client.json - must be a positive "
            f"number of seconds; using default {JOB_AUTHORIZATION_CLOCK_SKEW}"
        )
        return JOB_AUTHORIZATION_CLOCK_SKEW
    return skew


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
            app_path = workspace.get_app_dir(job_id)
            root_ca_path = os.path.join(workspace.get_startup_kit_dir(), "rootCA.pem")
            authorization = get_job_authorization(job_meta, app_name)
            if authorization:
                try:
                    verify_job_authorization(
                        authorization=authorization,
                        app_data=req.body,
                        root_ca_path=root_ca_path,
                        site_name=client_name,
                        expected_job_id=job_id,
                        expected_app_name=app_name,
                        expected_server_identity=_expected_server_identity(client_name),
                        clock_skew=_job_authorization_clock_skew(),
                    )
                except Exception as ex:
                    return error_reply(f"app {app_name} does not pass server authorization verification: {ex}")
            else:
                # Server-signed job authorization is required for every directly deployed job
                # (see docs/design/federated_admin_auth.md); only hub jobs are exempt. This is
                # intentionally not configurable. A pre-OIDC server does not attach the manifest,
                # so the SERVER must be upgraded before the FL clients during a rolling upgrade;
                # upgrading a client ahead of the server will reject deploys until the server is
                # upgraded too.
                return error_reply(f"app {app_name} is missing server job authorization")

            sig_file = os.path.join(app_path, NVFLARE_SIG_FILE)
            if os.path.exists(sig_file):
                if not verify_folder_signature(app_path, root_ca_path):
                    return error_reply(f"app {app_name} does not pass signature verification")
            # No elif on the client: require_signed_jobs is a server-side policy.
            # The server already rejected unsigned jobs before deploying to clients.
            # Accepted trust boundary: in a compromised-server scenario a malicious
            # unsigned job could reach the client, but at that point the server itself
            # is untrusted. Defense-in-depth here would require the client to independently
            # know the policy, which is not part of the current threat model.

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
