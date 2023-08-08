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

import os

from nvflare.apis.fl_constant import FLContextKey, MachineStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.private.fed.app.fl_conf import create_privacy_manager
from nvflare.private.fed.runner import Runner
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator
from nvflare.private.fed.server.server_status import ServerStatus
from nvflare.private.fed.utils.fed_utils import authorize_build_component
from nvflare.private.privacy_manager import PrivacyService
from nvflare.security.logging import secure_format_exception


def _set_up_run_config(workspace: Workspace, server, conf):
    runner_config = conf.runner_config

    # configure privacy control!
    privacy_manager = create_privacy_manager(workspace, names_only=False)
    if privacy_manager.is_policy_defined():
        if privacy_manager.components:
            for cid, comp in privacy_manager.components.items():
                runner_config.add_component(cid, comp)

    # initialize Privacy Service
    PrivacyService.initialize(privacy_manager)

    server.heart_beat_timeout = conf.heartbeat_timeout
    server.runner_config = conf.runner_config
    server.handlers = conf.handlers


class ServerAppRunner(Runner):
    def __init__(self, server) -> None:
        super().__init__()
        self.server = server

    def start_server_app(
        self, workspace: Workspace, args, app_root, job_id, snapshot, logger, kv_list=None, event_handlers=None
    ):
        try:
            server_config_file_name = os.path.join(app_root, args.server_config)

            conf = ServerJsonConfigurator(config_file_name=server_config_file_name, args=args, kv_list=kv_list)
            if event_handlers:
                fl_ctx = FLContext()
                fl_ctx.set_prop(FLContextKey.ARGS, args, sticky=False)
                fl_ctx.set_prop(FLContextKey.APP_ROOT, app_root, private=True, sticky=False)
                fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
                fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job_id, private=False, sticky=False)
                fl_ctx.set_prop(FLContextKey.CURRENT_RUN, job_id, private=False, sticky=False)
                conf.set_component_build_authorizer(
                    authorize_build_component, fl_ctx=fl_ctx, event_handlers=event_handlers
                )
            conf.configure()

            _set_up_run_config(workspace, self.server, conf)

            if not isinstance(self.server.engine, ServerEngine):
                raise TypeError(f"server.engine must be ServerEngine. Got type:{type(self.server.engine).__name__}")
            self.sync_up_parents_process(args)

            self.server.start_run(job_id, app_root, conf, args, snapshot)
        except Exception as e:
            with self.server.engine.new_context() as fl_ctx:
                fl_ctx.set_prop(key=FLContextKey.FATAL_SYSTEM_ERROR, value=True, private=True, sticky=True)
            logger.exception(f"FL server execution exception: {secure_format_exception(e)}")
            raise e
        finally:
            self.update_job_run_status()
            self.server.status = ServerStatus.STOPPED
            self.server.engine.engine_info.status = MachineStatus.STOPPED
            self.server.stop_training()

    def sync_up_parents_process(self, args):
        self.server.engine.sync_clients_from_main_process()

    def update_job_run_status(self):
        self.server.engine.update_job_run_status()

    def stop(self):
        self.server.engine.asked_to_stop = True
