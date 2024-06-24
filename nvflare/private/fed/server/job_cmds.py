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
import shutil
import uuid
from typing import Dict, List

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey, ReturnCode, RunProcessKey, ServerCommandKey
from nvflare.apis.job_def import Job, JobMetaKey, is_valid_job_id
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec, RunStatus
from nvflare.apis.shareable import Shareable
from nvflare.apis.storage import DATA, JOB_ZIP, META, META_JSON, WORKSPACE, WORKSPACE_ZIP
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ConfirmMethod, MetaKey, MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.binary_transfer import BinaryTransfer
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.argument_utils import SafeArgumentParser
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .cmd_utils import CommandUtil

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
    JobMetaKey.DATA_STORAGE_FORMAT.value,
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


class JobCommandModule(CommandModule, CommandUtil, BinaryTransfer):
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
                    client_cmd=ftd.PUSH_FOLDER_FQN,
                ),
                CommandSpec(
                    name=AdminCommandNames.DOWNLOAD_JOB,
                    description="download a specified job",
                    usage=f"{AdminCommandNames.DOWNLOAD_JOB} job_id [destination]",
                    handler_func=self.download_job,
                    authz_func=self.authorize_job,
                    client_cmd=ftd.PULL_FOLDER_FQN,
                ),
                # DOWNLOAD_JOB_FILE is an internal command that the client automatically issues
                # during the download process of a job.
                # This command is not visible to the user and cannot be issued by the user.
                CommandSpec(
                    name=AdminCommandNames.DOWNLOAD_JOB_FILE,
                    description="download a specified job file",
                    usage=f"{AdminCommandNames.DOWNLOAD_JOB_FILE} job_id file_name",
                    handler_func=self.pull_file,
                    authz_func=self.authorize_job_file,
                    client_cmd=ftd.PULL_BINARY_FQN,
                    visible=False,
                ),
                CommandSpec(
                    name=AdminCommandNames.APP_COMMAND,
                    description="execute an app-defined command",
                    usage=f"{AdminCommandNames.APP_COMMAND} job_id topic # cmd_data",
                    handler_func=self.do_app_command,
                    authz_func=self.authorize_job_id,
                ),
            ],
        )

    def authorize_job_file(self, conn: Connection, args: List[str]):
        """
        Args: cmd_name tx_id job_id file_name [end]
        """
        if len(args) < 4:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_error(f"Usage: {cmd_entry.usage}", meta=make_meta(MetaStatusValue.SYNTAX_ERROR))
            return PreAuthzReturnCode.ERROR
        job_id = args[2]
        args_for_authz = [args[0], job_id]
        return self.authorize_job_id(conn, args_for_authz)

    def authorize_job_id(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error(
                "syntax error: missing job_id", meta=make_meta(MetaStatusValue.SYNTAX_ERROR, "missing job_id")
            )
            return PreAuthzReturnCode.ERROR

        job_id = args[1].lower()
        if not is_valid_job_id(job_id):
            conn.append_error(f"invalid job_id {job_id}", meta=make_meta(MetaStatusValue.INVALID_JOB_ID, job_id))
            return PreAuthzReturnCode.ERROR

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
        return PreAuthzReturnCode.REQUIRE_AUTHZ

    def authorize_job(self, conn: Connection, args: List[str]):
        rc = self.authorize_job_id(conn, args)
        if rc == PreAuthzReturnCode.ERROR:
            return rc

        if len(args) > 2:
            err = self.validate_command_targets(conn, args[2:])
            if err:
                conn.append_error(err, meta=make_meta(MetaStatusValue.INVALID_TARGET, err))
                return PreAuthzReturnCode.ERROR

        return PreAuthzReturnCode.REQUIRE_AUTHZ

    def _start_app_on_clients(self, conn: Connection, job_id: str) -> bool:
        engine = conn.app_ctx
        client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
        run_process = engine.run_processes.get(job_id, {})
        if not run_process:
            conn.append_error(f"Job {job_id} is not running.")
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
                job_meta = {str(k): job.meta[k] for k in job.meta.keys() & CLONED_META_KEYS}

                # set the submitter info for the new job
                job_meta[JobMetaKey.SUBMITTER_NAME.value] = conn.get_prop(ConnProps.USER_NAME)
                job_meta[JobMetaKey.SUBMITTER_ORG.value] = conn.get_prop(ConnProps.USER_ORG)
                job_meta[JobMetaKey.SUBMITTER_ROLE.value] = conn.get_prop(ConnProps.USER_ROLE)
                job_meta[JobMetaKey.CLONED_FROM.value] = job_id

                meta = job_def_manager.clone(from_jid=job_id, meta=job_meta, fl_ctx=fl_ctx)
                new_job_id = meta.get(JobMetaKey.JOB_ID)
                conn.append_string(f"Cloned job {job_id} as: {new_job_id}")
        except Exception as e:
            conn.append_error(
                f"Exception occurred trying to clone job: {secure_format_exception(e)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"exception {type(e)}"),
            )
            return
        conn.append_success("", meta=make_meta(status=MetaStatusValue.OK, extra={MetaKey.JOB_ID: new_job_id}))

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
        zip_file_name = conn.extra

        engine = conn.app_ctx
        try:
            with engine.new_context() as fl_ctx:
                job_validator = JobMetaValidator()
                valid, error, meta = job_validator.validate(folder_name, zip_file_name)
                if not valid:
                    conn.append_error(error, meta=make_meta(MetaStatusValue.INVALID_JOB_DEFINITION, error))
                    return

                job_def_manager = engine.job_def_manager
                if not isinstance(job_def_manager, JobDefManagerSpec):
                    raise TypeError(
                        f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                    )

                fl_ctx.set_prop(FLContextKey.JOB_META, meta, private=True, sticky=False)
                engine.fire_event(EventType.SUBMIT_JOB, fl_ctx)
                block_reason = fl_ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)
                if block_reason:
                    # submitted job blocked
                    self.logger.error(f"submitted job is blocked: {block_reason}")
                    conn.append_error(
                        block_reason, meta=make_meta(MetaStatusValue.INVALID_JOB_DEFINITION, block_reason)
                    )
                    return

                # set submitter info
                meta[JobMetaKey.SUBMITTER_NAME.value] = conn.get_prop(ConnProps.USER_NAME, "")
                meta[JobMetaKey.SUBMITTER_ORG.value] = conn.get_prop(ConnProps.USER_ORG, "")
                meta[JobMetaKey.SUBMITTER_ROLE.value] = conn.get_prop(ConnProps.USER_ROLE, "")
                meta[JobMetaKey.JOB_FOLDER_NAME.value] = folder_name
                custom_props = conn.get_prop(ConnProps.CUSTOM_PROPS)
                if custom_props:
                    meta[JobMetaKey.CUSTOM_PROPS.value] = custom_props

                meta = job_def_manager.create(meta, zip_file_name, fl_ctx)
                job_id = meta.get(JobMetaKey.JOB_ID)

                # os.remove(zip_file_name)  # the file is no longer needed
                conn.append_string(f"Submitted job: {job_id}")
                conn.append_success("", meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOB_ID: job_id}))

        except Exception as e:
            conn.append_error(
                f"Exception occurred trying to submit job: {secure_format_exception(e)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"exception {type(e)} occurred"),
            )
            return

    def _clean_up_download(self, conn: Connection, tx_id: str):
        """
        Remove the job download folder
        """
        job_download_dir = self.tx_path(conn, tx_id)
        shutil.rmtree(job_download_dir, ignore_errors=True)

    def pull_file(self, conn: Connection, args: List[str]):
        """
        Args: cmd_name tx_id folder_name file_name [end]
        """
        if len(args) < 4:
            # NOTE: this should never happen since args have been validated by authorize_job_file!
            self.logger.error("syntax error: missing tx_id folder_name file name")
            return

        tx_id = args[1]
        folder_name = args[2]
        file_name = args[3]
        self.download_file(conn, tx_id, folder_name, file_name)
        if len(args) > 4:
            # this is the end of the download - remove the download dir
            self._clean_up_download(conn, tx_id)

    def download_job(self, conn: Connection, args: List[str]):
        """
        Job download uses binary protocol for more efficient download.
        - Retrieve job data from job store. This puts job files (meta, data, and workspace) in a transfer folder
        - Returns job file names, a TX ID, and a command name for downloading files to the admin client
        - Admin client downloads received file names one by one. It signals the end of download in the last command.
        """
        job_id = args[1]
        self.logger.debug(f"pull_job called for {job_id}")

        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager
        if not isinstance(job_def_manager, JobDefManagerSpec):
            self.logger.error(
                f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
            )
            conn.append_error("internal error", meta=make_meta(MetaStatusValue.INTERNAL_ERROR))
            return

        # It is possible that the same job is downloaded in multiple sessions at the same time.
        # To allow this, we use a separate sub-folder in the download_dir for each download.
        # This sub-folder is named with a transaction ID (tx_id), which is a UUID.
        # The folder path for download the job is: <download_dir>/<tx_id>/<job_id>.
        tx_id = str(uuid.uuid4())  # generate a new tx_id
        job_download_dir = self.tx_path(conn, tx_id)  # absolute path of the job download dir.
        with engine.new_context() as fl_ctx:
            try:
                job_def_manager.get_storage_for_download(job_id, job_download_dir, DATA, JOB_ZIP, fl_ctx)
                job_def_manager.get_storage_for_download(job_id, job_download_dir, META, META_JSON, fl_ctx)
                job_def_manager.get_storage_for_download(job_id, job_download_dir, WORKSPACE, WORKSPACE_ZIP, fl_ctx)

                self.download_folder(
                    conn,
                    tx_id=tx_id,
                    folder_name=job_id,
                    download_file_cmd_name=AdminCommandNames.DOWNLOAD_JOB_FILE,
                )
            except Exception as e:
                secure_log_traceback()
                self.logger.error(f"exception downloading job {job_id}: {secure_format_exception(e)}")
                self._clean_up_download(conn, tx_id)
                conn.append_error("internal error", meta=make_meta(MetaStatusValue.INTERNAL_ERROR))

    def do_app_command(self, conn: Connection, args: List[str]):
        # cmd job_id topic
        if len(args) != 3:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}", meta=make_meta(MetaStatusValue.SYNTAX_ERROR, ""))
            return

        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError(f"engine must be ServerEngineInternalSpec but got {type(engine)}")

        job_id = conn.get_prop(self.JOB_ID)
        topic = args[2]
        cmd_data = conn.get_prop(ConnProps.CMD_PROPS)

        if job_id not in engine.run_processes:
            conn.append_error(
                f"Job_id: {job_id} is not running.", meta=make_meta(MetaStatusValue.JOB_NOT_RUNNING, job_id)
            )
            return

        timeout = conn.get_prop(ConnProps.CMD_TIMEOUT)
        if not timeout:
            timeout = 5.0
        result = engine.send_app_command(job_id, topic, cmd_data, timeout)
        if result is None:
            conn.append_error(
                "command execution error: no result", meta=make_meta(MetaStatusValue.NO_REPLY, "no result")
            )
            return

        if not isinstance(result, Shareable):
            conn.append_error(
                f"command execution internal error: invalid result type {type(result)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"invalid result type {type(result)}"),
            )
            return

        rc = result.get_return_code()
        if rc != ReturnCode.OK:
            reason = result.get_header(ServerCommandKey.REASON)
            conn.append_error(
                f"command execution error: {rc=} {reason=}", meta=make_meta(MetaStatusValue.ERROR, f"{rc=} {reason=}")
            )
            return

        reply = result.get(ServerCommandKey.DATA)
        conn.append_dict(reply)
