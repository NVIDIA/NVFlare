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

import logging
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import MachineStatus, WorkspaceConstants
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.common_utils import get_open_ports
from nvflare.private.admin_defs import Message
from nvflare.private.defs import ERROR_MSG_PREFIX, ClientStatusKey, EngineConstant
from nvflare.private.event import fire_event
from nvflare.private.fed.utils.fed_utils import deploy_app

from .client_engine_internal_spec import ClientEngineInternalSpec
from .client_executor import ProcessExecutor
from .client_run_manager import ClientRunInfo
from .client_status import ClientStatus


class ClientEngine(ClientEngineInternalSpec):
    """ClientEngine runs in the client parent process."""

    def __init__(self, client, client_name, sender, args, rank, workers=5):
        """To init the ClientEngine.

        Args:
            client: FL client object
            client_name: client name
            sender: sender object
            args: command args
            rank: local process rank
            workers: number of workers
        """
        super().__init__()
        self.client = client
        self.client_name = client_name
        self.sender = sender
        self.args = args
        self.rank = rank
        self.client.process = None
        self.client_executor = ProcessExecutor(client.client_name, os.path.join(args.workspace, "startup"))

        self.fl_ctx_mgr = FLContextManager(
            engine=self, identity_name=client_name, job_id="", public_stickers={}, private_stickers={}
        )

        self.status = MachineStatus.STOPPED

        if workers < 1:
            raise ValueError("workers must >= 1")
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fl_components = [x for x in self.client.components.values() if isinstance(x, FLComponent)]

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        fire_event(event=event_type, handlers=self.fl_components, ctx=fl_ctx)

    def set_agent(self, admin_agent):
        self.admin_agent = admin_agent

    def do_validate(self, req: Message):
        self.logger.info("starting cross site validation.")
        _ = self.executor.submit(lambda p: _do_validate(*p), [self.sender, req])

        return "validate process started."

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def get_component(self, component_id: str) -> object:
        return self.client.components.get(component_id)

    def get_engine_status(self):
        running_jobs = []
        for job_id in self.get_all_job_ids():
            run_folder = os.path.join(self.args.workspace, WorkspaceConstants.WORKSPACE_PREFIX + str(job_id))
            app_file = os.path.join(run_folder, "fl_app.txt")
            if os.path.exists(app_file):
                with open(app_file, "r") as f:
                    app_name = f.readline().strip()
            job = {
                ClientStatusKey.APP_NAME: app_name,
                ClientStatusKey.JOB_ID: job_id,
                ClientStatusKey.STATUS: self.client_executor.check_status(self.client, job_id),
            }
            running_jobs.append(job)

        result = {
            ClientStatusKey.CLIENT_NAME: self.client.client_name,
            ClientStatusKey.RUNNING_JOBS: running_jobs,
        }
        return result

    def start_app(
        self,
        job_id: str,
        allocated_resource: dict = None,
        token: str = None,
        resource_consumer=None,
        resource_manager=None,
    ) -> str:
        status = self.client_executor.get_status(job_id)
        if status == ClientStatus.STARTED:
            return "Client app already started."

        app_root = os.path.join(
            self.args.workspace,
            WorkspaceConstants.WORKSPACE_PREFIX + str(job_id),
            WorkspaceConstants.APP_PREFIX + self.client.client_name,
        )
        if not os.path.exists(app_root):
            return f"{ERROR_MSG_PREFIX}: Client app does not exist. Please deploy it before starting client."

        if self.client.enable_byoc:
            app_custom_folder = os.path.join(app_root, "custom")
            try:
                sys.path.index(app_custom_folder)
            except ValueError:
                self.remove_custom_path()
                sys.path.append(app_custom_folder)
        else:
            app_custom_folder = ""

        self.logger.info("Starting client app. rank: {}".format(self.rank))

        open_port = get_open_ports(1)[0]

        self.client_executor.start_train(
            self.client,
            job_id,
            self.args,
            app_root,
            app_custom_folder,
            open_port,
            allocated_resource,
            token,
            resource_consumer,
            resource_manager,
            list(self.client.servers.values())[0]["target"],
        )

        return "Start the client app..."

    def get_client_name(self):
        return self.client.client_name

    def _write_token_file(self, job_id, open_port):
        token_file = os.path.join(self.args.workspace, EngineConstant.CLIENT_TOKEN_FILE)
        if os.path.exists(token_file):
            os.remove(token_file)
        with open(token_file, "wt") as f:
            f.write(
                "%s\n%s\n%s\n%s\n%s\n%s\n"
                % (
                    self.client.token,
                    self.client.ssid,
                    job_id,
                    self.client.client_name,
                    open_port,
                    list(self.client.servers.values())[0]["target"],
                )
            )

    def remove_custom_path(self):
        regex = re.compile(".*/run_.*/custom")
        custom_paths = list(filter(regex.search, sys.path))
        for path in custom_paths:
            sys.path.remove(path)

    def abort_app(self, job_id: str) -> str:
        status = self.client_executor.get_status(job_id)
        if status == ClientStatus.STOPPED:
            return "Client app already stopped."

        if status == ClientStatus.NOT_STARTED:
            return "Client app has not started."

        if status == ClientStatus.STARTING:
            return "Client app is starting, please wait for client to have started before abort."

        self.client_executor.abort_train(self.client, job_id)

        return "Abort signal has been sent to the client App."

    def abort_task(self, job_id: str) -> str:
        status = self.client_executor.get_status(job_id)
        if status == ClientStatus.NOT_STARTED:
            return "Client app has not started."

        if status == ClientStatus.STARTING:
            return "Client app is starting, please wait for started before abort_task."

        self.client_executor.abort_task(self.client, job_id)

        return "Abort signal has been sent to the current task."

    def shutdown(self) -> str:
        self.logger.info("Client shutdown...")
        touch_file = os.path.join(self.args.workspace, "shutdown.fl")
        self.client_executor.close()
        self.fire_event(EventType.SYSTEM_END, self.new_context())

        _ = self.executor.submit(lambda p: _shutdown_client(*p), [self.client, self.admin_agent, touch_file])

        self.executor.shutdown()
        return "Shutdown the client..."

    def restart(self) -> str:
        self.logger.info("Client shutdown...")
        touch_file = os.path.join(self.args.workspace, "restart.fl")
        self.client_executor.close()
        self.fire_event(EventType.SYSTEM_END, self.new_context())
        _ = self.executor.submit(lambda p: _shutdown_client(*p), [self.client, self.admin_agent, touch_file])

        self.executor.shutdown()
        return "Restart the client..."

    def deploy_app(self, app_name: str, job_id: str, client_name: str, app_data) -> str:
        workspace = os.path.join(self.args.workspace, WorkspaceConstants.WORKSPACE_PREFIX + str(job_id))

        if deploy_app(app_name, client_name, workspace, app_data):
            return f"Deployed app {app_name} to {client_name}"
        else:
            return f"{ERROR_MSG_PREFIX}: Failed to deploy_app"

    def delete_run(self, job_id: str) -> str:
        job_id_folder = os.path.join(self.args.workspace, WorkspaceConstants.WORKSPACE_PREFIX + str(job_id))
        if os.path.exists(job_id_folder):
            shutil.rmtree(job_id_folder)
        return f"Delete run folder: {job_id_folder}."

    def get_current_run_info(self, job_id) -> ClientRunInfo:
        return self.client_executor.get_run_info(job_id)

    def get_errors(self, job_id):
        return self.client_executor.get_errors(job_id)

    def reset_errors(self, job_id):
        self.client_executor.reset_errors(job_id)

    def send_aux_command(self, shareable: Shareable, job_id):
        return self.client_executor.send_aux_command(shareable, job_id)

    def get_all_job_ids(self):
        return self.client_executor.get_run_processes_keys()


def _do_validate(sender, message):
    print("starting the validate process .....")
    time.sleep(60)
    print("Generating processing result ......")
    reply = Message(topic=message.topic, body="")
    sender.send_result(reply)
    pass


def _shutdown_client(federated_client, admin_agent, touch_file):
    with open(touch_file, "a"):
        os.utime(touch_file, None)

    try:
        print("About to shutdown the client...")
        federated_client.communicator.heartbeat_done = True
        time.sleep(3)
        federated_client.close()

        if federated_client.process:
            federated_client.process.terminate()

        admin_agent.shutdown()
    except BaseException as e:
        traceback.print_exc()
        print("Failed to shutdown client: " + str(e))
