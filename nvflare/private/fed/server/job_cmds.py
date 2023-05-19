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

import datetime
import json
import logging
import os
import shutil
from typing import Dict, List

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.apis.client import Client
from nvflare.apis.fl_constant import AdminCommandNames, RunProcessKey
from nvflare.apis.job_def import Job, JobDataKey, JobMetaKey, TopDir
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec, RunStatus
from nvflare.apis.utils.job_utils import convert_legacy_zipped_app_to_job
from nvflare.fuel.hci.base64_utils import b64str_to_bytes, bytes_to_b64str
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ConfirmMethod, MetaKey, MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.argument_utils import SafeArgumentParser
from nvflare.fuel.utils.obj_utils import get_size
from nvflare.fuel.utils.zip_utils import ls_zip_from_bytes, unzip_all_from_bytes, zip_directory_to_bytes
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .cmd_utils import CommandUtil

MAX_DOWNLOAD_JOB_SIZE = 50 * 1024 * 1024 * 1204
CLONED_META_KEYS = {
    JobMetaKey.JOB_NAME.value,
    JobMetaKey.JOB_FOLDER_NAME.value,
    JobMetaKey.DEPLOY_MAP.value,
    JobMetaKey.RESOURCE_SPEC.value,
    JobMetaKey.CONTENT_LOCATION.value,
    JobMetaKey.RESULT_LOCATION.value,
    JobMetaKey.APPROVALS.value,
    JobMetaKey.MIN_CLIENTS.value,
    JobMetaKey.MANDATORY_CLIENTS.value,
}


def _create_list_job_cmd_parser():
    parser = SafeArgumentParser(prog=AdminCommandNames.LIST_JOBS)
    parser.add_argument("job_id", nargs="?", help="Job ID prefix")
    parser.add_argument("-d", action="store_true", help="Show detailed list")
    parser.add_argument("-u", action="store_true", help="List jobs submitted by the same user")
    parser.add_argument("-r", action="store_true", help="List jobs in reverse order of submission time")
    parser.add_argument("-n", help="Filter by job name prefix")
    parser.add_argument(
        "-m",
        type=int,
        help="Maximum number of jobs that will be listed",
    )
    return parser


class JobCommandModule(CommandModule, CommandUtil):
    """Command module with commands for job management."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_spec(self):
        return CommandModuleSpec(
            name="job_mgmt",
            cmd_specs=[
                CommandSpec(
                    name=AdminCommandNames.DELETE_WORKSPACE,
                    description="delete the workspace of a job",
                    usage=f"{AdminCommandNames.DELETE_WORKSPACE} job_id",
                    handler_func=self.delete_job_id,
                    authz_func=self.authorize_job,
                    enabled=False,
                    confirm=ConfirmMethod.AUTH,
                ),
                CommandSpec(
                    name=AdminCommandNames.START_APP,
                    description="start the FL app",
                    usage=f"{AdminCommandNames.START_APP} job_id server|client|all",
                    handler_func=self.start_app,
                    authz_func=self.authorize_job,
                ),
                CommandSpec(
                    name=AdminCommandNames.LIST_JOBS,
                    description="list submitted jobs",
                    usage=f"{AdminCommandNames.LIST_JOBS} [-n name_prefix] [-d] [-u] [-r] [-m num_of_jobs] [job_id_prefix]",
                    handler_func=self.list_jobs,
                    authz_func=self.command_authz_required,
                ),
                CommandSpec(
                    name=AdminCommandNames.GET_JOB_META,
                    description="get meta info of specified job",
                    usage=f"{AdminCommandNames.GET_JOB_META} job_id",
                    handler_func=self.get_job_meta,
                    authz_func=self.authorize_job,
                ),
                CommandSpec(
                    name=AdminCommandNames.DELETE_JOB,
                    description="delete a job and persisted workspace",
                    usage=f"{AdminCommandNames.DELETE_JOB} job_id",
                    handler_func=self.delete_job,
                    authz_func=self.authorize_job,
                    confirm=ConfirmMethod.AUTH,
                ),
                CommandSpec(
                    name=AdminCommandNames.ABORT_JOB,
                    description="abort a job if it is running or dispatched",
                    usage=f"{AdminCommandNames.ABORT_JOB} job_id",
                    handler_func=self.abort_job,  # see if running, if running, send abort command
                    authz_func=self.authorize_job,
                    confirm=ConfirmMethod.YESNO,
                ),
                CommandSpec(
                    name=AdminCommandNames.ABORT_TASK,
                    description="abort the client current task execution",
                    usage=f"{AdminCommandNames.ABORT_TASK} job_id <client-name>",
                    handler_func=self.abort_task,
                    authz_func=self.authorize_abort_client_task,
                ),
                CommandSpec(
                    name=AdminCommandNames.CLONE_JOB,
                    description="clone a job with a new job_id",
                    usage=f"{AdminCommandNames.CLONE_JOB} job_id",
                    handler_func=self.clone_job,
                    authz_func=self.authorize_job,
                ),
                CommandSpec(
                    name=AdminCommandNames.SUBMIT_JOB,
                    description="submit a job",
                    usage=f"{AdminCommandNames.SUBMIT_JOB} job_folder",
                    handler_func=self.submit_job,
                    authz_func=self.command_authz_required,
                    client_cmd=ftd.UPLOAD_FOLDER_FQN,
                ),
                CommandSpec(
                    name=AdminCommandNames.DOWNLOAD_JOB,
                    description="download a specified job",
                    usage=f"{AdminCommandNames.DOWNLOAD_JOB} job_id",
                    handler_func=self.download_job,
                    authz_func=self.authorize_job,
                    client_cmd=ftd.DOWNLOAD_FOLDER_FQN,
                ),
            ],
        )

    def authorize_job(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error(
                "syntax error: missing job_id", meta=make_meta(MetaStatusValue.SYNTAX_ERROR, "missing job_id")
            )
            return PreAuthzReturnCode.ERROR

        job_id = args[1].lower()
        conn.set_prop(self.JOB_ID, job_id)
        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager

        with engine.new_context() as fl_ctx:
            job = job_def_manager.get_job(job_id, fl_ctx)

        if not job:
            conn.append_error(
                f"Job with ID {job_id} doesn't exist", meta=make_meta(MetaStatusValue.INVALID_JOB_ID, job_id)
            )
            return PreAuthzReturnCode.ERROR

        conn.set_prop(self.JOB, job)

        conn.set_prop(ConnProps.SUBMITTER_NAME, job.meta.get(JobMetaKey.SUBMITTER_NAME, ""))
        conn.set_prop(ConnProps.SUBMITTER_ORG, job.meta.get(JobMetaKey.SUBMITTER_ORG, ""))
        conn.set_prop(ConnProps.SUBMITTER_ROLE, job.meta.get(JobMetaKey.SUBMITTER_ROLE, ""))

        if len(args) > 2:
            err = self.validate_command_targets(conn, args[2:])
            if err:
                conn.append_error(err, meta=make_meta(MetaStatusValue.SYNTAX_ERROR, err))
                return PreAuthzReturnCode.ERROR

        return PreAuthzReturnCode.REQUIRE_AUTHZ

    def abort_task(self, conn, args: List[str]) -> str:
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        message = new_message(conn, topic=TrainingTopic.ABORT_TASK, body="", require_authz=False)
        message.set_header(RequestHeader.JOB_ID, str(job_id))
        replies = self.send_request_to_clients(conn, message)
        return self.process_replies_to_table(conn, replies)

    def _start_app_on_clients(self, conn: Connection, job_id: str) -> bool:
        engine = conn.app_ctx
        client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
        run_process = engine.run_processes.get(job_id, {})
        if not run_process:
            conn.append_error(f"Job: {job_id} is not running.")
            return False

        participants: Dict[str, Client] = run_process.get(RunProcessKey.PARTICIPANTS, {})
        wrong_clients = []
        for client in client_names:
            client_valid = False
            for _, p in participants.items():
                if client == p.name:
                    client_valid = True
                    break
            if not client_valid:
                wrong_clients.append(client)

        if wrong_clients:
            display_clients = ",".join(wrong_clients)
            conn.append_error(f"{display_clients} are not in the job running list.")
            return False

        err = engine.check_app_start_readiness(job_id)
        if err:
            conn.append_error(err)
            return False

        message = new_message(conn, topic=TrainingTopic.START, body="", require_authz=False)
        message.set_header(RequestHeader.JOB_ID, job_id)
        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)
        return True

    def start_app(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        if len(args) < 3:
            conn.append_error("Please provide the target name (client / all) for start_app command.")
            return

        target_type = args[2]
        if target_type == self.TARGET_TYPE_SERVER:
            # if not self._start_app_on_server(conn, job_id):
            #     return
            conn.append_error("start_app command only supports client app start.")
            return
        elif target_type == self.TARGET_TYPE_CLIENT:
            if not self._start_app_on_clients(conn, job_id):
                return
        else:
            # # all
            # success = self._start_app_on_server(conn, job_id)
            #
            # if success:
            client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
            if client_names:
                if not self._start_app_on_clients(conn, job_id):
                    return
        conn.append_success("")

    def delete_job_id(self, conn: Connection, args: List[str]):
        job_id = args[1]
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngine):
            raise TypeError("engine must be ServerEngine but got {}".format(type(engine)))

        if job_id in engine.run_processes.keys():
            conn.append_error(f"Current running run_{job_id} can not be deleted.")
            return

        err = engine.delete_job_id(job_id)
        if err:
            conn.append_error(err)
            return

        # ask clients to delete this RUN
        message = new_message(conn, topic=TrainingTopic.DELETE_RUN, body="", require_authz=False)
        message.set_header(RequestHeader.JOB_ID, str(job_id))
        clients = engine.get_clients()
        if clients:
            conn.set_prop(self.TARGET_CLIENT_TOKENS, [x.token for x in clients])
            replies = self.send_request_to_clients(conn, message)
            self.process_replies_to_table(conn, replies)

        conn.append_success("")

    def list_jobs(self, conn: Connection, args: List[str]):
        try:
            parser = _create_list_job_cmd_parser()
            parsed_args = parser.parse_args(args[1:])

            engine = conn.app_ctx
            job_def_manager = engine.job_def_manager
            if not isinstance(job_def_manager, JobDefManagerSpec):
                raise TypeError(
                    f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                )

            with engine.new_context() as fl_ctx:
                jobs = job_def_manager.get_all_jobs(fl_ctx)
            if jobs:
                id_prefix = parsed_args.job_id
                name_prefix = parsed_args.n
                max_jobs_listed = parsed_args.m
                user_name = conn.get_prop(ConnProps.USER_NAME, "") if parsed_args.u else None

                filtered_jobs = [job for job in jobs if self._job_match(job.meta, id_prefix, name_prefix, user_name)]
                if not filtered_jobs:
                    conn.append_string(
                        "No jobs matching the specified criteria.",
                        meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOBS: []}),
                    )
                    return

                reverse = True if parsed_args.r else False
                filtered_jobs.sort(key=lambda job: job.meta.get(JobMetaKey.SUBMIT_TIME.value, 0.0), reverse=reverse)

                if max_jobs_listed:
                    if reverse:
                        filtered_jobs = filtered_jobs[:max_jobs_listed]
                    else:
                        filtered_jobs = filtered_jobs[-max_jobs_listed:]

                if parsed_args.d:
                    self._send_detail_list(conn, filtered_jobs)
                else:
                    self._send_summary_list(conn, filtered_jobs)

            else:
                conn.append_string("No jobs found.", meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOBS: []}))
        except Exception as e:
            conn.append_error(
                secure_format_exception(e),
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, info=secure_format_exception(e)),
            )
            return

        conn.append_success("")

    def delete_job(self, conn: Connection, args: List[str]):
        job = conn.get_prop(self.JOB)
        if not job:
            conn.append_error(
                "program error: job not set in conn", meta=make_meta(MetaStatusValue.INTERNAL_ERROR, "no job")
            )
            return

        job_id = conn.get_prop(self.JOB_ID)
        if job.meta.get(JobMetaKey.STATUS, "") in [RunStatus.DISPATCHED.value, RunStatus.RUNNING.value]:
            conn.append_error(
                f"job: {job_id} is running, could not be deleted at this time.",
                meta=make_meta(MetaStatusValue.JOB_RUNNING, job_id),
            )
            return

        try:
            engine = conn.app_ctx
            job_def_manager = engine.job_def_manager

            with engine.new_context() as fl_ctx:
                job_def_manager.delete(job_id, fl_ctx)
                conn.append_string(f"Job {job_id} deleted.")
        except Exception as e:
            conn.append_error(
                f"exception occurred: {secure_format_exception(e)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"exception {type(e)}"),
            )
            return
        conn.append_success("", meta=make_meta(MetaStatusValue.OK))

    def get_job_meta(self, conn: Connection, args: List[str]):
        job_id = conn.get_prop(self.JOB_ID)
        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager
        if not isinstance(job_def_manager, JobDefManagerSpec):
            raise TypeError(
                f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
            )
        with engine.new_context() as fl_ctx:
            job = job_def_manager.get_job(jid=job_id, fl_ctx=fl_ctx)
            if job:
                conn.append_dict(job.meta, meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOB_META: job.meta}))
            else:
                conn.append_error(
                    f"job {job_id} does not exist", meta=make_meta(MetaStatusValue.INVALID_JOB_ID, job_id)
                )

    def abort_job(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        job_runner = engine.job_runner

        try:
            job_id = conn.get_prop(self.JOB_ID)
            with engine.new_context() as fl_ctx:
                job_manager = engine.job_def_manager
                job = job_manager.get_job(job_id, fl_ctx)
                job_status = job.meta.get(JobMetaKey.STATUS)
                if job_status in [RunStatus.SUBMITTED, RunStatus.DISPATCHED]:
                    job_manager.set_status(job.job_id, RunStatus.FINISHED_ABORTED, fl_ctx)
                    message = f"Aborted the job {job_id} before running it."
                    conn.append_string(message)
                    conn.append_success("", meta=make_meta(MetaStatusValue.OK, message))
                    return
                elif job_status.startswith("FINISHED:"):
                    message = f"Job for {job_id} is already completed."
                    conn.append_string(message)
                    conn.append_success("", meta=make_meta(MetaStatusValue.OK, message))
                else:
                    message = job_runner.stop_run(job_id, fl_ctx)
                    if message:
                        conn.append_error(message, meta=make_meta(MetaStatusValue.INTERNAL_ERROR, message))
                    else:
                        message = "Abort signal has been sent to the server app."
                        conn.append_string(message)
                        conn.append_success("", meta=make_meta(MetaStatusValue.OK, message))
        except Exception as e:
            conn.append_error(
                f"Exception occurred trying to abort job: {secure_format_exception(e)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"exception {type(e)}"),
            )
            return

    def clone_job(self, conn: Connection, args: List[str]):
        job = conn.get_prop(self.JOB)
        job_id = conn.get_prop(self.JOB_ID)
        engine = conn.app_ctx
        try:
            if not isinstance(engine, ServerEngine):
                raise TypeError(f"engine is not of type ServerEngine, but got {type(engine)}")
            job_def_manager = engine.job_def_manager
            if not isinstance(job_def_manager, JobDefManagerSpec):
                raise TypeError(
                    f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                )
            with engine.new_context() as fl_ctx:
                data_bytes = job_def_manager.get_content(job_id, fl_ctx)

                job_meta = {str(k): job.meta[k] for k in job.meta.keys() & CLONED_META_KEYS}

                # set the submitter info for the new job
                job_meta[JobMetaKey.SUBMITTER_NAME.value] = conn.get_prop(ConnProps.USER_NAME)
                job_meta[JobMetaKey.SUBMITTER_ORG.value] = conn.get_prop(ConnProps.USER_ORG)
                job_meta[JobMetaKey.SUBMITTER_ROLE.value] = conn.get_prop(ConnProps.USER_ROLE)
                job_meta[JobMetaKey.CLONED_FROM.value] = job_id

                meta = job_def_manager.create(job_meta, data_bytes, fl_ctx)
                new_job_id = meta.get(JobMetaKey.JOB_ID)
                conn.append_string("Cloned job {} as: {}".format(job_id, new_job_id))
        except Exception as e:
            conn.append_error(
                f"Exception occurred trying to clone job: {secure_format_exception(e)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"exception {type(e)}"),
            )
            return
        conn.append_success("", meta=make_meta(status=MetaStatusValue.OK, extra={MetaKey.JOB_ID: new_job_id}))

    def authorize_list_files(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error("syntax error: missing job_id")
            return False, None

        if len(args) > 3:
            conn.append_error("syntax error: too many arguments")
            return False, None

        return self.authorize_job(conn=conn, args=args[:2])

    def list_files(self, conn: Connection, args: List[str]):
        job_id = conn.get_prop(self.JOB_ID)

        if len(args) == 2:
            conn.append_string("job\nworkspace\n\nSpecify the job or workspace dir to see detailed contents.")
            return
        else:
            file = args[2]

        engine = conn.app_ctx
        try:
            job_def_manager = engine.job_def_manager
            if not isinstance(job_def_manager, JobDefManagerSpec):
                raise TypeError(
                    f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                )
            with engine.new_context() as fl_ctx:
                job_data = job_def_manager.get_job_data(job_id, fl_ctx)
                if file.startswith(TopDir.JOB):
                    file = file[len(TopDir.JOB) :]
                    file = file.lstrip("/")
                    data_bytes = job_data[JobDataKey.JOB_DATA.value]
                    ls_info = ls_zip_from_bytes(data_bytes)
                elif file.startswith(TopDir.WORKSPACE):
                    file = file[len(TopDir.WORKSPACE) :]
                    file = file.lstrip("/")
                    workspace_bytes = job_data[JobDataKey.WORKSPACE_DATA.value]
                    ls_info = ls_zip_from_bytes(workspace_bytes)
                else:
                    conn.append_error("syntax error: top level directory must be job or workspace")
                    return
                return_string = "%-46s %19s %12s\n" % ("File Name", "Modified    ", "Size")
                for zinfo in ls_info:
                    date = "%d-%02d-%02d %02d:%02d:%02d" % zinfo.date_time[:6]
                    if zinfo.filename.startswith(file):
                        return_string += "%-46s %s %12d\n" % (zinfo.filename, date, zinfo.file_size)
                conn.append_string(return_string)
        except Exception as e:
            secure_log_traceback()
            conn.append_error(f"Exception occurred trying to get job from store: {secure_format_exception(e)}")
            return
        conn.append_success("")

    @staticmethod
    def _job_match(job_meta: Dict, id_prefix: str, name_prefix: str, user_name: str) -> bool:
        return (
            ((not id_prefix) or job_meta.get("job_id").lower().startswith(id_prefix.lower()))
            and ((not name_prefix) or job_meta.get("name").lower().startswith(name_prefix.lower()))
            and ((not user_name) or job_meta.get("submitter_name") == user_name)
        )

    @staticmethod
    def _send_detail_list(conn: Connection, jobs: List[Job]):
        list_of_jobs = []
        for job in jobs:
            JobCommandModule._set_duration(job)
            conn.append_string(json.dumps(job.meta, indent=4))
            list_of_jobs.append(job.meta)
        conn.append_string("", meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOBS: list_of_jobs}))

    @staticmethod
    def _send_summary_list(conn: Connection, jobs: List[Job]):
        table = conn.append_table(["Job ID", "Name", "Status", "Submit Time", "Run Duration"], name=MetaKey.JOBS)
        for job in jobs:
            JobCommandModule._set_duration(job)
            table_row = [
                job.meta.get(JobMetaKey.JOB_ID.value, ""),
                CommandUtil.get_job_name(job.meta),
                job.meta.get(JobMetaKey.STATUS.value, ""),
                job.meta.get(JobMetaKey.SUBMIT_TIME_ISO.value, ""),
                str(job.meta.get(JobMetaKey.DURATION.value, "N/A")),
            ]
            table.add_row(
                table_row,
                meta={
                    MetaKey.JOB_ID: job.meta.get(JobMetaKey.JOB_ID.value, ""),
                    MetaKey.JOB_NAME: CommandUtil.get_job_name(job.meta),
                    MetaKey.STATUS: job.meta.get(JobMetaKey.STATUS.value, ""),
                    MetaKey.SUBMIT_TIME: job.meta.get(JobMetaKey.SUBMIT_TIME_ISO.value, ""),
                    MetaKey.DURATION: str(job.meta.get(JobMetaKey.DURATION.value, "N/A")),
                },
            )

    @staticmethod
    def _set_duration(job):
        if job.meta.get(JobMetaKey.STATUS) == RunStatus.RUNNING.value:
            start_time = datetime.datetime.strptime(job.meta.get(JobMetaKey.START_TIME.value), "%Y-%m-%d %H:%M:%S.%f")
            duration = datetime.datetime.now() - start_time
            job.meta[JobMetaKey.DURATION.value] = str(duration)

    def submit_job(self, conn: Connection, args: List[str]):
        folder_name = args[1]
        zip_b64str = args[2]

        data_bytes = convert_legacy_zipped_app_to_job(b64str_to_bytes(zip_b64str))
        engine = conn.app_ctx

        try:
            with engine.new_context() as fl_ctx:
                job_validator = JobMetaValidator()
                valid, error, meta = job_validator.validate(folder_name, data_bytes)
                if not valid:
                    conn.append_error(error, meta=make_meta(MetaStatusValue.INVALID_JOB_DEFINITION, error))
                    return

                job_def_manager = engine.job_def_manager
                if not isinstance(job_def_manager, JobDefManagerSpec):
                    raise TypeError(
                        f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                    )

                # set submitter info
                meta[JobMetaKey.SUBMITTER_NAME.value] = conn.get_prop(ConnProps.USER_NAME, "")
                meta[JobMetaKey.SUBMITTER_ORG.value] = conn.get_prop(ConnProps.USER_ORG, "")
                meta[JobMetaKey.SUBMITTER_ROLE.value] = conn.get_prop(ConnProps.USER_ROLE, "")

                meta = job_def_manager.create(meta, data_bytes, fl_ctx)
                job_id = meta.get(JobMetaKey.JOB_ID)
                conn.append_string(f"Submitted job: {job_id}")
                conn.append_success("", meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOB_ID: job_id}))
        except Exception as e:
            conn.append_error(
                f"Exception occurred trying to submit job: {secure_format_exception(e)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"exception {type(e)} occurred"),
            )
            return

    def _unzip_data(self, download_dir, job_data, job_id):
        job_id_dir = os.path.join(download_dir, job_id)
        if os.path.exists(job_id_dir):
            shutil.rmtree(job_id_dir)
        os.mkdir(job_id_dir)

        data_bytes = job_data[JobDataKey.JOB_DATA.value]
        job_dir = os.path.join(job_id_dir, "job")
        os.mkdir(job_dir)
        unzip_all_from_bytes(data_bytes, job_dir)

        workspace_bytes = job_data[JobDataKey.WORKSPACE_DATA.value]
        workspace_dir = os.path.join(job_id_dir, "workspace")
        os.mkdir(workspace_dir)
        if workspace_bytes is not None:
            unzip_all_from_bytes(workspace_bytes, workspace_dir)

    def download_job(self, conn: Connection, args: List[str]):
        job_id = args[1]
        download_dir = conn.get_prop(ConnProps.DOWNLOAD_DIR)
        download_job_url = conn.get_prop(ConnProps.DOWNLOAD_JOB_URL)

        engine = conn.app_ctx
        try:
            job_def_manager = engine.job_def_manager
            if not isinstance(job_def_manager, JobDefManagerSpec):
                raise TypeError(
                    f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                )
            with engine.new_context() as fl_ctx:
                job_data = job_def_manager.get_job_data(job_id, fl_ctx)
                size = get_size(job_data, seen=None)
                if size > MAX_DOWNLOAD_JOB_SIZE:
                    conn.append_string(
                        ftd.DOWNLOAD_URL_MARKER + download_job_url + job_id,
                        meta=make_meta(
                            MetaStatusValue.OK,
                            extra={MetaKey.JOB_ID: job_id, MetaKey.JOB_DOWNLOAD_URL: download_job_url + job_id},
                        ),
                    )
                    return

                self._unzip_data(download_dir, job_data, job_id)
        except Exception as e:
            conn.append_error(f"Exception occurred trying to get job from store: {secure_format_exception(e)}")
            return
        try:
            data = zip_directory_to_bytes(download_dir, job_id)
            b64str = bytes_to_b64str(data)
            conn.append_string(b64str, meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOB_ID: job_id}))
        except FileNotFoundError:
            conn.append_error("No record found for job '{}'".format(job_id))
        except Exception:
            secure_log_traceback()
            conn.append_error("Exception occurred during attempt to zip data to send for job: {}".format(job_id))
