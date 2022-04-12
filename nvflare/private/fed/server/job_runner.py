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

import os.path
import threading
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import RunProcessKey
from nvflare.apis.fl_constant import SystemComponents, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import RunStatus, Job
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.utils.fed_utils import deploy_app


class JobRunner(FLComponent):

    def __init__(self, workspace_root: str) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self.ask_to_stop = False
        self.scheduler = None
        self.running_jobs = {}
        self.lock = threading.Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            engine = fl_ctx.get_engine()
            self.scheduler = engine.get_component(SystemComponents.JOB_SCHEDULER)

    def _deploy_job(self, job: Job, sites: list, fl_ctx: FLContext) -> str:
        """deploy the application to the list of participants

        Args:
            job:
            sites:
            fl_ctx:

        Returns:

        """
        engine = fl_ctx.get_engine()
        run_number = job.job_id
        workspace = os.path.join(self.workspace_root, WorkspaceConstants.WORKSPACE_PREFIX + run_number)
        count = 1
        while os.path.exists(workspace):
            run_number = run_number + "_" + str(count)
            workspace = os.path.join(self.workspace_root, WorkspaceConstants.WORKSPACE_PREFIX + run_number)
            count += 1

        for app_name, participants in job.get_deployment().items():
            app_data = job.get_application(app_name, fl_ctx)

            client_sites = []
            for p in participants:
                if p == "server":
                    success = deploy_app(app_name=app_name, site_name="server", workspace=workspace, app_data=app_data)
                    if not success:
                        raise RuntimeError("Failed to deploy the App to the server")
                else:
                    if p in sites:
                        client_sites.append(p)

            self._deploy_clients(app_data, app_name, run_number, client_sites, engine)

        self.fire_event(EventType.JOB_DEPLOYED, fl_ctx)
        return run_number

    def _deploy_clients(self, app_data, app_name, run_number, client_sites, engine):
        # deploy app to all the client sites
        admin_server = engine.server.admin_server
        message = Message(topic=TrainingTopic.DEPLOY, body=app_data)
        message.set_header(RequestHeader.RUN_NUM, run_number)
        message.set_header(RequestHeader.APP_NAME, app_name)
        replies = self._send_to_clients(admin_server, client_sites, engine, message)
        if not replies:
            raise RuntimeError("Failed to deploy the App to the clients")

    def _send_to_clients(self, admin_server, client_sites, engine, message):
        clients, invalid_inputs = engine.validate_clients(client_sites)
        requests = {}
        for c in clients:
            requests.update({c.token: message})
        replies = admin_server.send_requests(requests, timeout_secs=admin_server.timeout)
        return replies

    def _start_run(self, run_number, client_sites: list, fl_ctx: FLContext):
        """Start the application

        Args:
            run_number:
            client_sites:
            fl_ctx:

        Returns:

        """
        engine = fl_ctx.get_engine()
        err = engine.start_app_on_server(run_number)
        if err:
            raise RuntimeError("Could not start the server App.")

        admin_server = engine.server.admin_server
        message = Message(topic=TrainingTopic.START, body="")
        message.set_header(RequestHeader.RUN_NUM, run_number)
        replies = self._send_to_clients(admin_server, client_sites, engine, message)
        if not replies:
            raise RuntimeError("Failed to start the App to the clients")

        self.fire_event(EventType.JOB_STARTED, fl_ctx)

    def _stop_run(self, run_number, fl_ctx: FLContext):
        """Stop the application

        Args:
            run_number:
            fl_ctx:

        Returns:

        """
        engine = fl_ctx.get_engine()
        run_process = engine.run_processes.get(run_number)
        if run_process:
            admin_server = engine.server.admin_server

            client_sites = run_process.get(RunProcessKey.PARTICIPANTS)
            message = Message(topic=TrainingTopic.ABORT, body="")
            message.set_header(RequestHeader.RUN_NUM, str(run_number))
            replies = self._send_to_clients(admin_server, client_sites, engine, message)
            if not replies:
                self.log_error(fl_ctx,f"Failed to send abort command to clients for run_{run_number}")

            err = engine.abort_app_on_server(run_number)
            if err:
                self.log_error(fl_ctx, f"Failed to abort the server for run_.{run_number}")

    def _job_complete_process(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        while not self.ask_to_stop:
            for run_number in self.running_jobs.keys():
                if run_number not in engine.run_processes.keys():
                    with self.lock:
                        job = self.running_jobs.get(run_number)
                        job_manager.set_status(job.job_id, RunStatus.FINISHED_COMPLETED, fl_ctx)
                        del self.running_jobs[run_number]
            time.sleep(1.0)

    def run(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        # threading.Thread(target=self._job_complete_process, args=[fl_ctx]).start()

        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        while not self.ask_to_stop:
            if job_manager:
                # approved_jobs = job_manager.get_jobs_by_status(RunStatus.APPROVED, fl_ctx)
                approved_jobs = job_manager.get_jobs_by_status(RunStatus.SUBMITTED, fl_ctx)
                if self.scheduler:
                    (ready_job, sites) = self.scheduler.schedule_job(job_candidates=approved_jobs, fl_ctx=fl_ctx)

                    if ready_job:
                        try:
                            run_number = self._deploy_job(ready_job, sites, fl_ctx)
                            job_manager.set_status(ready_job.job_id, RunStatus.DISPATCHED, fl_ctx)
                            self._start_run(run_number, sites, fl_ctx)
                            with self.lock:
                                self.running_jobs[run_number] = ready_job
                            job_manager.set_status(ready_job.job_id, RunStatus.RUNNING, fl_ctx)
                        except:
                            self.log_error(fl_ctx, f"Failed to run the Job ID: {ready_job.job_id}")

            time.sleep(1.0)

    def stop_run(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        for run_number in engine.run_processes.keys():
            self._stop_run(run_number, fl_ctx)
            job = self.running_jobs.get(run_number)
            if job:
                job_manager.set_status(job.job_id, RunStatus.FINISHED_ABORTED, fl_ctx)

        self.ask_to_stop = True
