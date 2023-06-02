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

import json
import os
import shutil
import threading
import time
from typing import Dict, List, Tuple

from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey, RunProcessKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import ALL_SITES, Job, JobMetaKey, RunStatus
from nvflare.apis.job_scheduler_spec import DispatchInfo
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.zip_utils import zip_directory_to_bytes
from nvflare.lighter.utils import verify_folder_signature
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server.admin import check_client_replies
from nvflare.private.fed.server.server_state import HotState
from nvflare.private.fed.utils.app_deployer import AppDeployer
from nvflare.security.logging import secure_format_exception


def _send_to_clients(admin_server, client_sites: List[str], engine, message, timeout=None, optional=False):
    clients, invalid_inputs = engine.validate_targets(client_sites)
    if invalid_inputs:
        raise RuntimeError(f"unknown clients: {invalid_inputs}.")
    requests = {}
    for c in clients:
        requests.update({c.token: message})

    if timeout is None:
        timeout = admin_server.timeout
    replies = admin_server.send_requests(requests, timeout_secs=timeout, optional=optional)
    return replies


def _get_active_job_participants(connected_clients: Dict[str, Client], participants: Dict[str, Client]) -> List[str]:
    """Gets active job participants.

        Some clients might be dropped/dead during job execution.
        No need to abort those clients.

    Args:
        connected_clients: Clients that are currently connected.
        participants: Clients that were participating when the job started.

    Returns:
        A list of active job participants name.
    """
    client_sites_names = []
    for token, client in participants.items():
        if token in connected_clients:
            client_sites_names.append(client.name)

    return client_sites_names


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
        elif event_type in [EventType.JOB_COMPLETED, EventType.END_RUN]:
            self._save_workspace(fl_ctx)
        elif event_type == EventType.SYSTEM_END:
            self.stop()

    def _make_deploy_message(self, job: Job, app_data, app_name):
        message = Message(topic=TrainingTopic.DEPLOY, body=app_data)
        message.set_header(RequestHeader.REQUIRE_AUTHZ, "true")

        message.set_header(RequestHeader.ADMIN_COMMAND, AdminCommandNames.SUBMIT_JOB)
        message.set_header(RequestHeader.JOB_ID, job.job_id)
        message.set_header(RequestHeader.APP_NAME, app_name)

        message.set_header(RequestHeader.SUBMITTER_NAME, job.meta.get(JobMetaKey.SUBMITTER_NAME))
        message.set_header(RequestHeader.SUBMITTER_ORG, job.meta.get(JobMetaKey.SUBMITTER_ORG))
        message.set_header(RequestHeader.SUBMITTER_ROLE, job.meta.get(JobMetaKey.SUBMITTER_ROLE))

        message.set_header(RequestHeader.USER_NAME, job.meta.get(JobMetaKey.SUBMITTER_NAME))
        message.set_header(RequestHeader.USER_ORG, job.meta.get(JobMetaKey.SUBMITTER_ORG))
        message.set_header(RequestHeader.USER_ROLE, job.meta.get(JobMetaKey.SUBMITTER_ROLE))

        message.set_header(RequestHeader.JOB_META, json.dumps(job.meta))
        return message

    def _deploy_job(self, job: Job, sites: dict, fl_ctx: FLContext) -> Tuple[str, list]:
        """Deploy the application to the list of participants

        Args:
            job: job to be deployed
            sites: participating sites
            fl_ctx: FLContext

        Returns:  job id, failed_clients

        """
        fl_ctx.remove_prop(FLContextKey.JOB_RUN_NUMBER)
        fl_ctx.remove_prop(FLContextKey.JOB_DEPLOY_DETAIL)
        engine = fl_ctx.get_engine()
        run_number = job.job_id
        fl_ctx.set_prop(FLContextKey.JOB_RUN_NUMBER, run_number)
        workspace = Workspace(root_dir=self.workspace_root, site_name="server")

        client_deploy_requests = {}
        client_token_to_name = {}
        client_token_to_reply = {}
        deploy_detail = []
        fl_ctx.set_prop(FLContextKey.JOB_DEPLOY_DETAIL, deploy_detail)

        for app_name, participants in job.get_deployment().items():
            app_data = job.get_application(app_name, fl_ctx)

            if len(participants) == 1 and participants[0].upper() == ALL_SITES:
                participants = ["server"]
                participants.extend([client.name for client in engine.get_clients()])

            client_sites = []
            for p in participants:
                if p == "server":
                    app_deployer = AppDeployer(
                        app_name=app_name, workspace=workspace, job_id=job.job_id, job_meta=job.meta, app_data=app_data
                    )

                    err = app_deployer.deploy()
                    if err:
                        deploy_detail.append(f"server: {err}")
                        raise RuntimeError(f"Failed to deploy app '{app_name}': {err}")

                    kv_list = parse_vars(engine.args.set)
                    secure_train = kv_list.get("secure_train", True)
                    if secure_train:
                        app_path = workspace.get_app_dir(job.job_id)
                        root_ca_path = os.path.join(workspace.get_startup_kit_dir(), "rootCA.pem")
                        if not verify_folder_signature(app_path, root_ca_path):
                            err = "job signature verification failed"
                            deploy_detail.append(f"server: {err}")
                            raise RuntimeError(f"Failed to verify app '{app_name}': {err}")

                    self.log_info(
                        fl_ctx, f"Application {app_name} deployed to the server for job: {run_number}", fire_event=False
                    )
                    deploy_detail.append("server: OK")
                else:
                    if p in sites:
                        client_sites.append(p)

            if client_sites:
                message = self._make_deploy_message(job, app_data, app_name)
                clients, invalid_inputs = engine.validate_targets(client_sites)

                if invalid_inputs:
                    deploy_detail.append("invalid_clients: {}".format(",".join(invalid_inputs)))
                    raise RuntimeError(f"unknown clients: {invalid_inputs}.")

                for c in clients:
                    assert isinstance(c, Client)
                    client_token_to_name[c.token] = c.name
                    client_deploy_requests[c.token] = message
                    client_token_to_reply[c.token] = None

                display_sites = ",".join(client_sites)
                self.log_info(
                    fl_ctx,
                    f"App {app_name} to be deployed to the clients: {display_sites} for run: {run_number}",
                    fire_event=False,
                )

        abort_job = False
        failed_clients = []
        if client_deploy_requests:
            engine = fl_ctx.get_engine()
            admin_server = engine.server.admin_server
            client_token_to_reply = admin_server.send_requests_and_get_reply_dict(
                client_deploy_requests, timeout_secs=admin_server.timeout
            )

            # check replies and see whether required clients are okay
            for client_token, reply in client_token_to_reply.items():
                client_name = client_token_to_name[client_token]
                if reply:
                    assert isinstance(reply, Message)
                    rc = reply.get_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
                    if rc != ReturnCode.OK:
                        failed_clients.append(client_name)
                        deploy_detail.append(f"{client_name}: {reply.body}")
                    else:
                        deploy_detail.append(f"{client_name}: OK")
                else:
                    deploy_detail.append(f"{client_name}: unknown")

            # see whether any of the failed clients are required
            if failed_clients:
                num_ok_sites = len(client_deploy_requests) - len(failed_clients)
                if job.min_sites and num_ok_sites < job.min_sites:
                    abort_job = True
                    deploy_detail.append(f"num_ok_sites {num_ok_sites} < required_min_sites {job.min_sites}")
                elif job.required_sites:
                    for c in failed_clients:
                        if c in job.required_sites:
                            abort_job = True
                            deploy_detail.append(f"failed to deploy to required client {c}")

        if abort_job:
            raise RuntimeError("deploy failure", deploy_detail)

        self.fire_event(EventType.JOB_DEPLOYED, fl_ctx)
        return run_number, failed_clients

    def _start_run(self, job_id: str, job: Job, client_sites: Dict[str, DispatchInfo], fl_ctx: FLContext):
        """Start the application

        Args:
            job_id: job_id
            client_sites: participating sites
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()
        job_clients = engine.get_job_clients(client_sites)
        err = engine.start_app_on_server(job_id, job=job, job_clients=job_clients)
        if err:
            raise RuntimeError(f"Could not start the server App for job: {job_id}.")

        replies = engine.start_client_job(job_id, client_sites)
        client_sites_names = list(client_sites.keys())
        check_client_replies(replies=replies, client_sites=client_sites_names, command=f"start job ({job_id})")
        display_sites = ",".join(client_sites_names)

        self.log_info(fl_ctx, f"Started run: {job_id} for clients: {display_sites}")
        self.fire_event(EventType.JOB_STARTED, fl_ctx)

    def _stop_run(self, job_id, fl_ctx: FLContext):
        """Stop the application

        Args:
            job_id: job_id to be stopped
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()
        run_process = engine.run_processes.get(job_id)
        if run_process:
            participants: Dict[str, Client] = run_process.get(RunProcessKey.PARTICIPANTS)
            active_client_sites_names = _get_active_job_participants(
                connected_clients=engine.client_manager.clients, participants=participants
            )

            self.abort_client_run(job_id, active_client_sites_names, fl_ctx)

            err = engine.abort_app_on_server(job_id)
            if err:
                self.log_error(fl_ctx, f"Failed to abort the server for run: {job_id}: {err}")

    def abort_client_run(self, job_id, client_sites: List[str], fl_ctx):
        """Send the abort run command to the clients

        Args:
            job_id: job_id
            client_sites: Clients to be aborted
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()
        admin_server = engine.server.admin_server
        message = Message(topic=TrainingTopic.ABORT, body="")
        message.set_header(RequestHeader.JOB_ID, str(job_id))
        self.log_debug(fl_ctx, f"Send abort command to the clients for run: {job_id}")
        try:
            _ = _send_to_clients(admin_server, client_sites, engine, message, timeout=2.0, optional=True)
            # There isn't much we can do here if a client didn't get the message or send a reply
            # check_client_replies(replies=replies, client_sites=client_sites, command="abort the run")
        except RuntimeError as e:
            self.log_error(fl_ctx, f"Failed to abort run ({job_id}) on the clients: {secure_format_exception(e)}")

    def _delete_run(self, job_id, client_sites: List[str], fl_ctx: FLContext):
        """Deletes the run workspace

        Args:
            job_id: job_id
            client_sites: participating sites
            fl_ctx: FLContext
        """
        engine = fl_ctx.get_engine()

        admin_server = engine.server.admin_server
        message = Message(topic=TrainingTopic.DELETE_RUN, body="")
        message.set_header(RequestHeader.JOB_ID, str(job_id))
        self.log_debug(fl_ctx, f"Send delete_run command to the clients for run: {job_id}")
        try:
            replies = _send_to_clients(admin_server, client_sites, engine, message)
            check_client_replies(replies=replies, client_sites=client_sites, command="send delete_run command")
        except RuntimeError as e:
            self.log_error(
                fl_ctx, f"Failed to execute delete run ({job_id}) on the clients: {secure_format_exception(e)}"
            )

        err = engine.delete_job_id(job_id)
        if err:
            self.log_error(fl_ctx, f"Failed to delete_run the server for run: {job_id}")

    def _job_complete_process(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        while not self.ask_to_stop:
            for job_id in list(self.running_jobs.keys()):
                if job_id not in engine.run_processes.keys():
                    job = self.running_jobs.get(job_id)
                    if job:
                        if not job.run_aborted:
                            self._update_job_status(engine, job, job_manager, fl_ctx)
                        with self.lock:
                            del self.running_jobs[job_id]
                        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job.job_id)
                        self.fire_event(EventType.JOB_COMPLETED, fl_ctx)
                        self.log_debug(fl_ctx, f"Finished running job:{job.job_id}")
                    engine.remove_exception_process(job_id)
            time.sleep(1.0)

    def _update_job_status(self, engine, job, job_manager, fl_ctx):
        exception_run_processes = engine.exception_run_processes
        if job.job_id in exception_run_processes:
            self.log_info(fl_ctx, f"Try to abort run ({job.job_id}) on clients.")
            run_process = exception_run_processes[job.job_id]

            # stop client run
            participants: Dict[str, Client] = run_process.get(RunProcessKey.PARTICIPANTS)
            active_client_sites_names = _get_active_job_participants(
                connected_clients=engine.client_manager.clients, participants=participants
            )
            self.abort_client_run(job.job_id, active_client_sites_names, fl_ctx)

            process_return_code = run_process.get(RunProcessKey.PROCESS_RETURN_CODE)
            if process_return_code == -9:
                job_manager.set_status(job.job_id, RunStatus.FINISHED_ABNORMAL, fl_ctx)
            else:
                job_manager.set_status(job.job_id, RunStatus.FINISHED_EXECUTION_EXCEPTION, fl_ctx)
        else:
            job_manager.set_status(job.job_id, RunStatus.FINISHED_COMPLETED, fl_ctx)

    def _save_workspace(self, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        workspace = Workspace(root_dir=self.workspace_root)
        run_dir = workspace.get_run_dir(job_id)
        workspace_data = zip_directory_to_bytes(run_dir, "")
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)

        job_manager.save_workspace(job_id, workspace_data, fl_ctx)
        shutil.rmtree(run_dir)

    def run(self, fl_ctx: FLContext):
        """Starts job runner."""
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        if job_manager:
            thread = threading.Thread(target=self._job_complete_process, args=[fl_ctx])
            thread.start()

            while not self.ask_to_stop:
                if not isinstance(engine.server.server_state, HotState):
                    time.sleep(1.0)
                    continue
                approved_jobs = job_manager.get_jobs_by_status(RunStatus.SUBMITTED, fl_ctx)
                self.log_debug(
                    fl_ctx, f"{fl_ctx.get_identity_name()} Got approved_jobs: {approved_jobs} from the job_manager"
                )

                if self.scheduler:
                    ready_job, sites = self.scheduler.schedule_job(
                        job_manager=job_manager, job_candidates=approved_jobs, fl_ctx=fl_ctx
                    )

                    if ready_job:
                        if self._check_job_status(job_manager, ready_job.job_id, RunStatus.SUBMITTED, fl_ctx):
                            self.log_info(fl_ctx, f"Job: {ready_job.job_id} is not in SUBMITTED. It won't be deployed.")
                            continue
                        client_sites = {k: v for k, v in sites.items() if k != "server"}
                        job_id = None
                        try:
                            self.log_info(fl_ctx, f"Got the job: {ready_job.job_id} from the scheduler to run")
                            fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, ready_job.job_id)
                            job_id, failed_clients = self._deploy_job(ready_job, sites, fl_ctx)
                            job_manager.set_status(ready_job.job_id, RunStatus.DISPATCHED, fl_ctx)

                            deploy_detail = fl_ctx.get_prop(FLContextKey.JOB_DEPLOY_DETAIL)
                            if deploy_detail:
                                job_manager.update_meta(
                                    ready_job.job_id,
                                    {
                                        JobMetaKey.JOB_DEPLOY_DETAIL.value: deploy_detail,
                                        JobMetaKey.SCHEDULE_COUNT.value: ready_job.meta[
                                            JobMetaKey.SCHEDULE_COUNT.value
                                        ],
                                        JobMetaKey.LAST_SCHEDULE_TIME.value: ready_job.meta[
                                            JobMetaKey.LAST_SCHEDULE_TIME.value
                                        ],
                                        JobMetaKey.SCHEDULE_HISTORY.value: ready_job.meta[
                                            JobMetaKey.SCHEDULE_HISTORY.value
                                        ],
                                    },
                                    fl_ctx,
                                )

                            if failed_clients:
                                deployable_clients = {k: v for k, v in client_sites.items() if k not in failed_clients}
                            else:
                                deployable_clients = client_sites

                            if self._check_job_status(job_manager, ready_job.job_id, RunStatus.DISPATCHED, fl_ctx):
                                self.log_info(
                                    fl_ctx, f"Job: {ready_job.job_id} is not in DISPATCHED. It won't be start to run."
                                )
                                continue

                            self._start_run(
                                job_id=job_id,
                                job=ready_job,
                                client_sites=deployable_clients,
                                fl_ctx=fl_ctx,
                            )
                            with self.lock:
                                self.running_jobs[job_id] = ready_job
                            job_manager.set_status(ready_job.job_id, RunStatus.RUNNING, fl_ctx)
                        except Exception as e:
                            if job_id:
                                if job_id in self.running_jobs:
                                    with self.lock:
                                        del self.running_jobs[job_id]
                                self._stop_run(job_id, fl_ctx)
                            job_manager.set_status(ready_job.job_id, RunStatus.FAILED_TO_RUN, fl_ctx)

                            deploy_detail = fl_ctx.get_prop(FLContextKey.JOB_DEPLOY_DETAIL)
                            if deploy_detail:
                                job_manager.update_meta(
                                    ready_job.job_id, {JobMetaKey.JOB_DEPLOY_DETAIL.value: deploy_detail}, fl_ctx
                                )

                            self.fire_event(EventType.JOB_ABORTED, fl_ctx)
                            self.log_error(
                                fl_ctx, f"Failed to run the Job ({ready_job.job_id}): {secure_format_exception(e)}"
                            )

                time.sleep(1.0)

            thread.join()
        else:
            self.log_error(fl_ctx, "There's no Job Manager defined. Won't be able to run the jobs.")

    def _check_job_status(self, job_manager, job_id, job_run_status, fl_ctx: FLContext):
        reload_job = job_manager.get_job(job_id, fl_ctx)
        return reload_job.meta.get(JobMetaKey.STATUS) != job_run_status

    def stop(self):
        self.ask_to_stop = True

    def restore_running_job(self, run_number: str, job_id: str, job_clients, snapshot, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        try:
            job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
            job = job_manager.get_job(jid=job_id, fl_ctx=fl_ctx)
            err = engine.start_app_on_server(run_number, job=job, job_clients=job_clients, snapshot=snapshot)
            if err:
                raise RuntimeError(f"Could not restore the server App for job: {job_id}.")
            with self.lock:
                self.running_jobs[job_id] = job
            self.scheduler.restore_scheduled_job(job_id)
        except Exception as e:
            self.log_error(
                fl_ctx, f"Failed to restore the job: {job_id} to the running job table: {secure_format_exception(e)}."
            )

    def update_unfinished_jobs(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        all_jobs = []
        dispatched_jobs = job_manager.get_jobs_by_status(RunStatus.DISPATCHED, fl_ctx)
        all_jobs.extend(dispatched_jobs)
        running_jobs = job_manager.get_jobs_by_status(RunStatus.RUNNING, fl_ctx)
        all_jobs.extend(running_jobs)

        for job in all_jobs:
            try:
                job_manager.set_status(job.job_id, RunStatus.ABANDONED, fl_ctx)
                self.logger.info(f"Update the previous running job: {job.job_id} to ABANDONED.")
            except Exception as e:
                self.log_error(
                    fl_ctx, f"Failed to update the job: {job.job_id} to ABANDONED: {secure_format_exception(e)}."
                )

    def stop_run(self, job_id: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        self._stop_run(job_id, fl_ctx)

        job = self.running_jobs.get(job_id)
        if job:
            self.log_info(fl_ctx, f"Stop the job run: {job_id}")
            fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job.job_id)
            job.run_aborted = True
            job_manager.set_status(job.job_id, RunStatus.FINISHED_ABORTED, fl_ctx)
            self.fire_event(EventType.JOB_ABORTED, fl_ctx)
            return ""
        else:
            self.log_error(fl_ctx, f"Job {job_id} is not running. It can not be stopped.")
            return f"Job {job_id} is not running."

    def stop_all_runs(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        for job_id in engine.run_processes.keys():
            self.stop_run(job_id, fl_ctx)

        self.log_info(fl_ctx, "Stop all the running jobs.")
        # also stop the job runner
        self.ask_to_stop = True

    def remove_running_job(self, job_id: str):
        with self.lock:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
        self.scheduler.remove_scheduled_job(job_id)
