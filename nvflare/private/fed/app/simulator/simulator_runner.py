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
import os
import shutil
import sys
import threading

from nvflare.apis.job_def import JobMetaKey
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.simulator.simulator_client_app_runner import SimulatorClientAppRunner, SimulatorServerAppRunner


class SimulatorRunner:

    def run(self, simulator_root, args, logger, services, federated_client):
        meta_file = os.path.join(args.job_folder, "meta.json")
        with open(meta_file, "rb") as f:
            meta_data = f.read()
        meta = json.loads(meta_data)

        threading.Thread(target=self.start_server, args=[simulator_root, args, logger, services, meta]).start()

        threading.Thread(target=self.start_client, args=[simulator_root, args, federated_client, meta]).start()

    def start_server(self, simulator_root, args, logger, services, meta):
        # jid = str(uuid.uuid4())
        # meta[JobMetaKey.JOB_ID.value] = jid
        # meta[JobMetaKey.SUBMIT_TIME.value] = time.time()
        # meta[JobMetaKey.SUBMIT_TIME_ISO.value] = (
        #     datetime.datetime.fromtimestamp(meta[JobMetaKey.SUBMIT_TIME]).astimezone().isoformat()
        # )
        # meta[JobMetaKey.START_TIME.value] = ""
        # meta[JobMetaKey.DURATION.value] = "N/A"
        # meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value
        app_server_root = os.path.join(simulator_root, "app_server")
        for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p == "server":
                    app = os.path.join(args.job_folder, app_name)
                    shutil.copytree(app, app_server_root)

        args.server_config = os.path.join("config", AppFolderConstants.CONFIG_FED_SERVER)
        app_custom_folder = os.path.join(app_server_root, "custom")
        sys.path.append(app_custom_folder)

        server_app_runner = SimulatorServerAppRunner()
        snapshot = None
        server_app_runner.start_server_app(services, args, app_server_root, args.job_id, snapshot, logger)

    def start_client(self, simulator_root, args, federated_client, meta):
        for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p != "server":
                    app_client_root = os.path.join(simulator_root, "app_" + p)
                    app = os.path.join(args.job_folder, app_name)
                    shutil.copytree(app, app_client_root)

                    args.client_name = p
                    args.token = federated_client.token
                    client_app_runner = SimulatorClientAppRunner()
                    client_app_runner.start_run(app_client_root, args, args.config_folder, federated_client, False)
