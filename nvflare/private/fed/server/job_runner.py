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
from typing import List

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, RunProcessKey, SystemComponents, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import ALL_SITES, Job, RunStatus
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server.admin import check_client_replies
from nvflare.private.fed.utils.fed_utils import deploy_app


def _send_to_clients(admin_server, client_sites: List[str], engine, message):
    clients, invalid_inputs = engine.validate_clients(client_sites)
    if invalid_inputs:
        raise RuntimeError(f"invalid clients: {invalid_inputs}.")
    requests = {}
    for c in clients:
        requests.update({c.token: message})
    replies = admin_server.send_requests(requests, timeout_secs=admin_server.timeout)
    return replies


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

    def _deploy_clients(self, app_data, app_name, run_number, client_sites: List[str], fl_ctx):
        engine = fl_ctx.get_engine()
        # deploy app to all the client sites
        admin_server = engine.server.admin_server
        message = Message(topic=TrainingTopic.DEPLOY, body=app_data)
        message.set_header(RequestHeader.RUN_NUM, run_number)
        message.set_header(RequestHeader.APP_NAME, app_name)
        self.log_debug(fl_ctx, f"Send deploy command to the clients for run: {run_number}")
        replies = _send_to_clients(admin_server, client_sites, engine, message)
        return replies

    def _deploy_job(self, job: Job, sites: dict, fl_ctx: FLContext) -> str:
        """deploy the application to the list of participants

        Args:
            job: job to be deployed
            sites: participating sites
            fl_ctx: FLContext

        Returns:
            run_number
        """
        fl_ctx.remove_prop(FLContextKey.JOB_RUN_NUMBER)
        engine = fl_ctx.get_engine()
        run_number = job.job_id
        workspace = os.path.join(self.workspace_root, WorkspaceConstants.WORKSPACE_PREFIX + run_number)
        count = 1
        while os.path.exists(workspace):
            run_number = job.job_id + ":" + str(count)
            workspace = os.path.join(self.workspace_root, WorkspaceConstants.WORKSPACE_PREFIX + run_number)
            count += 1
        fl_ctx.set_prop(FLContextKey.JOB_RUN_NUMBER, run_number)

        for app_name, participants in job.get_deployment().items():
            app_data = job.get_application(app_name, fl_ctx)

            if len(participants) == 1 and participants[0].upper() == ALL_SITES:
                participants = ["server"]
                participants.extend([client.name for client in engine.get_clients()])

            client_sites = []
            for p in participants:
                if p == "server":
                    success = deploy_app(app_name=app_name, site_name="server", workspace=workspace, app_data=app_data)
                    self.log_info(
                        fl_ctx, f"Application {app_name} deployed to the server for run: {run_number}", fire_event=False
                    )
                    if not success:
                        raise RuntimeError(f"Failed to deploy the App: {app_name} to the server")
                else:
                    if p in sites:
                        client_sites.append(p)

            if client_sites:
                replies = self._deploy_clients(app_data, app_name, run_number, client_sites, fl_ctx)
                check_client_replies(replies=replies, client_sites=client_sites, command="deploy the App")
                display_sites = ",".join(client_sites)
                self.log_info(
                    fl_ctx,
                    f"Application {app_name} deployed to the clients: {display_sites} for run: {run_number}",
                    fire_event=False,
                )

        self.fire_event(EventType.JOB_DEPLOYED, fl_ctx)
        return run_number

    def _start_run(self, run_number: str, job: Job, client_sites: dict, fl_ctx: FLContext):
        """Start the application

        Args:
            run_number: run_number
            client_sites: participating sites
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()
        job_clients = engine.get_job_clients(client_sites)
        err = engine.start_app_on_server(run_number, job_id=job.job_id, job_clients=job_clients)
        if err:
            raise RuntimeError(f"Could not start the server App for Run: {run_number}.")

        replies = engine.start_client_job(run_number, client_sites)
        client_sites_names = list(client_sites.keys())
        check_client_replies(replies=replies, client_sites=client_sites_names, command=f"start job ({run_number})")
        display_sites = ",".join(client_sites_names)

        self.log_info(fl_ctx, f"Started run: {run_number} for clients: {display_sites}")
        self.fire_event(EventType.JOB_STARTED, fl_ctx)

    def _stop_run(self, run_number, fl_ctx: FLContext):
        """Stop the application

        Args:
            run_number: run_number to be stopped
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()
        run_process = engine.run_processes.get(run_number)
        if run_process:
            client_sites = run_process.get(RunProcessKey.PARTICIPANTS)
            self.abort_client_run(run_number, client_sites, fl_ctx)

            err = engine.abort_app_on_server(run_number)
            if err:
                self.log_error(fl_ctx, f"Failed to abort the server for run: {run_number}")

    def abort_client_run(self, run_number, client_sites: List[str], fl_ctx):
        """Send the abort run command to the clients

        Args:
            run_number: run_number
            client_sites: Clients to be aborted
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()
        admin_server = engine.server.admin_server
        message = Message(topic=TrainingTopic.ABORT, body="")
        message.set_header(RequestHeader.RUN_NUM, str(run_number))
        self.log_debug(fl_ctx, f"Send abort command to the clients for run: {run_number}")
        try:
            replies = _send_to_clients(admin_server, client_sites, engine, message)
            check_client_replies(replies=replies, client_sites=client_sites, command="abort the run")
        except RuntimeError as e:
            self.log_error(fl_ctx, f"Failed to abort run ({run_number}) on the clients: {e}")

    def _delete_run(self, run_number, client_sites: List[str], fl_ctx: FLContext):
        """Deletes the run workspace

        Args:
            run_number: run_number
            client_sites: participating sites
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()

        admin_server = engine.server.admin_server
        message = Message(topic=TrainingTopic.DELETE_RUN, body="")
        message.set_header(RequestHeader.RUN_NUM, str(run_number))
        self.log_debug(fl_ctx, f"Send delete_run command to the clients for run: {run_number}")
        try:
            replies = _send_to_clients(admin_server, client_sites, engine, message)
            check_client_replies(replies=replies, client_sites=client_sites, command="send delete_run command")
        except RuntimeError as e:
            self.log_error(fl_ctx, f"Failed to execute delete run ({run_number}) on the clients: {e}")

        err = engine.delete_run_number(run_number)
        if err:
            self.log_error(fl_ctx, f"Failed to delete_run the server for run: {run_number}")

    def _job_complete_process(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        while not self.ask_to_stop:
            for run_number in list(self.running_jobs.keys()):
                if run_number not in engine.run_processes.keys():
                    with self.lock:
                        job = self.running_jobs.get(run_number)
                        if job:
                            if run_number in engine.execution_exception_run_processes:
                                self.log_info(fl_ctx, f"Try to abort run ({run_number}) on clients.")
                                run_process = engine.execution_exception_run_processes[run_number]

                                # stop client run
                                client_sites = run_process.get(RunProcessKey.PARTICIPANTS)
                                self.abort_client_run(run_number, client_sites, fl_ctx)

                                job_manager.set_status(job.job_id, RunStatus.FINISHED_EXECUTION_EXCEPTION, fl_ctx)
                            else:
                                job_manager.set_status(job.job_id, RunStatus.FINISHED_COMPLETED, fl_ctx)
                            del self.running_jobs[run_number]
                            fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job.job_id)
                            self.fire_event(EventType.JOB_COMPLETED, fl_ctx)
                            self.log_debug(fl_ctx, f"Finished running job:{job.job_id}")
            time.sleep(1.0)

    def run(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        threading.Thread(target=self._job_complete_process, args=[fl_ctx]).start()

        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        if job_manager:
            while not self.ask_to_stop:
                # approved_jobs = job_manager.get_jobs_by_status(RunStatus.APPROVED, fl_ctx)
                approved_jobs = job_manager.get_jobs_by_status(RunStatus.SUBMITTED, fl_ctx)
                if self.scheduler:
                    (ready_job, sites) = self.scheduler.schedule_job(job_candidates=approved_jobs, fl_ctx=fl_ctx)
                    if ready_job:
                        with self.lock:
                            client_sites = {k: v for k, v in sites.items() if k != "server"}
                            run_number = None
                            try:
                                self.log_info(fl_ctx, f"Got the job:{ready_job.job_id} from the scheduler to run")
                                fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, ready_job.job_id)
                                run_number = self._deploy_job(ready_job, sites, fl_ctx)
                                job_manager.set_status(ready_job.job_id, RunStatus.DISPATCHED, fl_ctx)
                                self._start_run(
                                    run_number=run_number,
                                    job=ready_job,
                                    client_sites=client_sites,
                                    fl_ctx=fl_ctx,
                                )
                                self.running_jobs[run_number] = ready_job
                                job_manager.set_status(ready_job.job_id, RunStatus.RUNNING, fl_ctx)
                            except Exception as e:
                                if run_number:
                                    if run_number in self.running_jobs:
                                        del self.running_jobs[run_number]
                                    self._stop_run(run_number, fl_ctx)
                                job_manager.set_status(ready_job.job_id, RunStatus.FAILED_TO_RUN, fl_ctx)
                                self.fire_event(EventType.JOB_ABORTED, fl_ctx)
                                self.log_error(fl_ctx, f"Failed to run the Job ({ready_job.job_id}): {e}")

                time.sleep(1.0)
        else:
            self.log_error(fl_ctx, "There's no Job Manager defined. Won't be able to run the jobs.")

    def restore_running_job(self, run_number: str, job_id: str, job_clients, snapshot, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        engine.start_app_on_server(run_number, job_id=job_id, job_clients=job_clients, snapshot=snapshot)

        try:
            job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
            job = job_manager.get_job(jid=job_id, fl_ctx=fl_ctx)
            with self.lock:
                self.running_jobs[run_number] = job
        except Exception as e:
            self.log_error(fl_ctx, f"Failed to restore the job: {job_id} to the running job table: {e}.")

    def stop_run(self, run_number: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        with self.lock:
            self._stop_run(run_number, fl_ctx)
            job = self.running_jobs.get(run_number)
            if job:
                self.log_info(fl_ctx, f"Stop the job run: {run_number}")
                fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job.job_id)
                job_manager.set_status(job.job_id, RunStatus.FINISHED_ABORTED, fl_ctx)
                del self.running_jobs[run_number]
                self.fire_event(EventType.JOB_ABORTED, fl_ctx)
            else:
                raise RuntimeError(f"Job run: {run_number} does not exist.")

    def stop_all_runs(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        for run_number in engine.run_processes.keys():
            self.stop_run(run_number, fl_ctx)

        self.log_info(fl_ctx, "Stop all the running jobs.")
        self.ask_to_stop = True
