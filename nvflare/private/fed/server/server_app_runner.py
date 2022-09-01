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

import os

from nvflare.apis.fl_constant import MachineStatus
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator
from nvflare.private.fed.server.server_status import ServerStatus


class ServerAppRunner:
    def start_server_app(self, server, args, app_root, job_id, snapshot, logger):

        try:
            server_config_file_name = os.path.join(app_root, args.server_config)

            conf = ServerJsonConfigurator(
                config_file_name=server_config_file_name,
            )
            conf.configure()

            self.set_up_run_config(server, conf)

            if not isinstance(server.engine, ServerEngine):
                raise TypeError(f"server.engine must be ServerEngine. Got type:{type(server.engine).__name__}")
            self.sync_up_parents_process(args, server)

            server.start_run(job_id, app_root, conf, args, snapshot)
        except BaseException as e:
            logger.exception(f"FL server execution exception: {e}", exc_info=True)
            raise e
        finally:
            server.status = ServerStatus.STOPPED
            server.engine.engine_info.status = MachineStatus.STOPPED
            server.stop_training()

    def sync_up_parents_process(self, args, server):
        server.engine.create_parent_connection(int(args.conn))
        server.engine.sync_clients_from_main_process()

    def set_up_run_config(self, server, conf):
        server.heart_beat_timeout = conf.heartbeat_timeout
        server.runner_config = conf.runner_config
        server.handlers = conf.handlers
