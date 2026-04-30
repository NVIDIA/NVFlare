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
import io
import json
import os
import re
import shutil
import threading
import uuid
import weakref
from typing import Dict, List, Optional, Set
from zipfile import BadZipFile, ZipFile

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey, ReturnCode, ServerCommandKey, WorkspaceConstants
from nvflare.apis.job_def import (
    ALL_SITES,
    DEFAULT_STUDY,
    SERVER_SITE_NAME,
    Job,
    JobMetaKey,
    SubmitRecordKey,
    SubmitRecordState,
    get_job_meta_study,
    is_valid_job_id,
)
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec, RunStatus
from nvflare.apis.shareable import Shareable
from nvflare.apis.storage import (
    DATA,
    JOB_ZIP,
    META,
    META_JSON,
    WORKSPACE,
    WORKSPACE_ZIP,
    DataTypes,
    StorageException,
    StorageSpec,
)
from nvflare.apis.utils.job_submit_token import canonical_job_content_hash
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ConfirmMethod, MetaKey, MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.binary_transfer import BinaryTransfer
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.argument_utils import SafeArgumentParser
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.private.fed.utils.fed_utils import extract_participants
from nvflare.security.logging import secure_format_exception, secure_log_traceback
from nvflare.security.study_registry import StudyRegistryService

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
    JobMetaKey.STUDY.value,
}

_SUBMIT_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SUBMIT_TOKEN_CONFLICT_STATUS = "submit_token_conflict"


def _validate_submit_token(submit_token: str) -> str:
    if submit_token is None:
        return None
    if not isinstance(submit_token, str) or not _SUBMIT_TOKEN_PATTERN.fullmatch(submit_token):
        raise ValueError("submit_token must be non-empty, at most 128 characters, and match ^[A-Za-z0-9._:-]{1,128}$")
    return submit_token


def _active_study_from_conn(conn: Connection) -> str:
    return conn.get_prop(ConnProps.ACTIVE_STUDY, DEFAULT_STUDY) or DEFAULT_STUDY


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
    parser.add_argument("--submit-token", dest="submit_token", help="retry-safe submit token")
    return parser


def _create_submit_job_cmd_parser():
    parser = SafeArgumentParser(prog=AdminCommandNames.SUBMIT_JOB)
    parser.add_argument("folder_name", help="Uploaded job folder name")
    parser.add_argument("--submit-token", dest="submit_token", help="retry-safe submit token")
    return parser


def _create_get_job_log_cmd_parser():
    parser = SafeArgumentParser(prog=AdminCommandNames.GET_JOB_LOG)
    parser.add_argument("job_id", help="Job ID")
    parser.add_argument("target", nargs="?", default=SERVER_SITE_NAME, help="server, all, or a client site name")
    return parser


class JobCommandModule(CommandModule, CommandUtil, BinaryTransfer):
    """Command module with commands for job management."""

    MAX_RETURNED_JOB_LOG_BYTES = 5 * 1024 * 1024
    _submit_token_locks = weakref.WeakValueDictionary()
    _submit_token_locks_guard = threading.Lock()

    def __init__(self):
        super().__init__()
        self.logger = get_obj_logger(self)

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
                    name=AdminCommandNames.CONFIGURE_JOB_LOG,
                    description="configure logging of a running job",
                    usage=f"{AdminCommandNames.CONFIGURE_JOB_LOG} job_id server|client <client-name>... config",
                    handler_func=self.configure_job_log,
                    authz_func=self.authorize_configure_job_log,
                ),
                CommandSpec(
                    name=AdminCommandNames.LIST_JOBS,
                    description="list submitted jobs",
                    usage=(
                        f"{AdminCommandNames.LIST_JOBS} [-n name_prefix] [-d] [-u] [-r] "
                        "[-m num_of_jobs] [--submit-token token] [job_id_prefix]"
                    ),
                    handler_func=self.list_jobs,
                    authz_func=self.command_authz_required,
                ),
                CommandSpec(
                    name=AdminCommandNames.GET_JOB_LOG,
                    description="get job log text from the server-side log store",
                    usage=f"{AdminCommandNames.GET_JOB_LOG} job_id [server|all|client_name]",
                    handler_func=self.get_job_log,
                    authz_func=self.authorize_job_id,
                ),
                CommandSpec(
                    name=AdminCommandNames.GET_JOB_META,
                    description="get meta info of specified job",
                    usage=f"{AdminCommandNames.GET_JOB_META} job_id",
                    handler_func=self.get_job_meta,
                    authz_func=self.authorize_job,
                ),
                CommandSpec(
                    name=AdminCommandNames.LIST_JOB,
                    description="list additional components of specified job",
                    usage=f"{AdminCommandNames.LIST_JOB} job_id",
                    handler_func=self.list_job_components,
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
                    usage=f"{AdminCommandNames.SUBMIT_JOB} job_folder [--submit-token token]",
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
                CommandSpec(
                    name=AdminCommandNames.DOWNLOAD_JOB_COMPONENTS,
                    description="download additional components for a specified job",
                    usage=f"{AdminCommandNames.DOWNLOAD_JOB_COMPONENTS} job_id",
                    authz_func=self.authorize_job,
                    handler_func=self.download_job_components,
                    client_cmd=ftd.PULL_FOLDER_FQN,
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

        requested_study = _active_study_from_conn(conn)
        if requested_study and get_job_meta_study(job.meta) != requested_study:
            conn.append_error(
                f"Job with ID {job_id} doesn't exist", meta=make_meta(MetaStatusValue.INVALID_JOB_ID, job_id)
            )
            return PreAuthzReturnCode.ERROR

        conn.set_prop(self.JOB, job)
        conn.set_prop(ConnProps.SUBMITTER_NAME, job.meta.get(JobMetaKey.SUBMITTER_NAME, ""))
        conn.set_prop(ConnProps.SUBMITTER_ORG, job.meta.get(JobMetaKey.SUBMITTER_ORG, ""))
        conn.set_prop(ConnProps.SUBMITTER_ROLE, job.meta.get(JobMetaKey.SUBMITTER_ROLE, ""))
        if not self._apply_study_role_for_authz(conn):
            return PreAuthzReturnCode.ERROR
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

    def authorize_configure_job_log(self, conn: Connection, args: List[str]):
        if len(args) < 4:
            conn.append_error("syntax error: please provide job_id, target_type, and config")
            return PreAuthzReturnCode.ERROR
        return self.authorize_job(conn, args[:-1])

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

    def configure_job_log(self, conn: Connection, args: List[str]):
        if len(args) < 4:
            conn.append_error("syntax error: please provide job_id, target_type, and config")
            return

        job_id = args[1]
        target_type = args[2]
        config = args[-1]

        engine = conn.app_ctx
        if not isinstance(engine, ServerEngine):
            raise TypeError("engine must be ServerEngine but got {}".format(type(engine)))

        try:
            with engine.new_context() as fl_ctx:
                job_manager = engine.job_def_manager
                job = job_manager.get_job(job_id, fl_ctx)
                job_status = job.meta.get(JobMetaKey.STATUS)
                if job_status != RunStatus.RUNNING.value:
                    conn.append_error(f"Job {job_id} must be running but is {job_status}")
                    return
        except Exception as e:
            conn.append_error(
                f"Exception occurred trying to check job status {job_id} for configure_job_log: {secure_format_exception(e)}",
                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, f"exception {type(e)}"),
            )
            return

        if target_type in [self.TARGET_TYPE_SERVER, self.TARGET_TYPE_ALL]:
            err = engine.configure_job_log(str(job_id), config)
            if err:
                conn.append_error(err)
                return

            conn.append_string(f"successfully configured server job {job_id} log")

        if target_type in [self.TARGET_TYPE_CLIENT, self.TARGET_TYPE_ALL]:
            message = new_message(conn, topic=TrainingTopic.CONFIGURE_JOB_LOG, body=config, require_authz=False)
            message.set_header(RequestHeader.JOB_ID, str(job_id))
            replies = self.send_request_to_clients(conn, message)
            self.process_replies_to_table(conn, replies)

        if target_type not in [self.TARGET_TYPE_ALL, self.TARGET_TYPE_CLIENT, self.TARGET_TYPE_SERVER]:
            conn.append_error(
                "invalid target type {}. Usage: configure_job_log job_id server|client <client-name>...|all config".format(
                    target_type
                )
            )

    def list_jobs(self, conn: Connection, args: List[str]):
        try:
            parser = _create_list_job_cmd_parser()
            parsed_args = parser.parse_args(args[1:])
            submit_token = _validate_submit_token(parsed_args.submit_token)
            requested_study = _active_study_from_conn(conn)

            engine = conn.app_ctx
            job_def_manager = engine.job_def_manager
            if not isinstance(job_def_manager, JobDefManagerSpec):
                raise TypeError(
                    f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                )

            with engine.new_context() as fl_ctx:
                if submit_token:
                    jobs = self._list_jobs_by_submit_token(
                        job_def_manager,
                        requested_study,
                        self._submitter_from_conn(conn),
                        submit_token,
                        fl_ctx,
                    )
                else:
                    jobs = job_def_manager.get_all_jobs(fl_ctx)
            if jobs:
                id_prefix = parsed_args.job_id
                name_prefix = parsed_args.n
                max_jobs_listed = parsed_args.m
                user_name = conn.get_prop(ConnProps.USER_NAME, "") if parsed_args.u else None

                filtered_jobs = [
                    job for job in jobs if self._job_match(job.meta, id_prefix, name_prefix, user_name, requested_study)
                ]
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
        except ValueError as e:
            conn.append_error(str(e), meta=make_meta(MetaStatusValue.SYNTAX_ERROR, str(e)))
            return
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
                normalized_meta = dict(job.meta)
                normalized_meta[JobMetaKey.STUDY.value] = get_job_meta_study(job.meta)
                conn.append_dict(
                    normalized_meta, meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOB_META: normalized_meta})
                )
            else:
                conn.append_error(
                    f"job {job_id} does not exist", meta=make_meta(MetaStatusValue.INVALID_JOB_ID, job_id)
                )

    def get_job_log(self, conn: Connection, args: List[str]):
        try:
            parser = _create_get_job_log_cmd_parser()
            parsed_args = parser.parse_args(args[1:])
        except Exception as e:
            conn.append_error(
                secure_format_exception(e),
                meta=make_meta(MetaStatusValue.SYNTAX_ERROR, secure_format_exception(e)),
            )
            return

        prop_job_id = conn.get_prop(self.JOB_ID)
        parsed_job_id = parsed_args.job_id.lower() if isinstance(parsed_args.job_id, str) else parsed_args.job_id
        if prop_job_id is not None and parsed_job_id is not None and prop_job_id != parsed_job_id:
            conn.append_error(
                "job_id mismatch between connection property and parsed argument",
                meta=make_meta(
                    MetaStatusValue.SYNTAX_ERROR,
                    "job_id mismatch between connection property and parsed argument",
                ),
            )
            return

        job_id = prop_job_id if prop_job_id is not None else parsed_job_id
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngine):
            raise TypeError("engine must be ServerEngine but got {}".format(type(engine)))

        target = parsed_args.target
        target_lower = target.lower() if isinstance(target, str) else target
        payload = {"logs": {}}

        # Accept the protocol token "@ALL" and the CLI/API spelling "all".
        all_targets = {ALL_SITES.lower(), "all"}
        workspace_zip = None
        workspace_zip_loaded = False
        if target_lower in all_targets:
            workspace_zip = self._read_stored_workspace_zip(conn, job_id)
            workspace_zip_loaded = True

        if target_lower in {SERVER_SITE_NAME, *all_targets}:
            server_log = self._read_server_job_log(
                conn, job_id, workspace_zip=workspace_zip, workspace_zip_loaded=workspace_zip_loaded
            )
            if server_log is not None or target_lower == SERVER_SITE_NAME:
                payload["logs"][SERVER_SITE_NAME] = server_log or ""
            else:
                payload.setdefault("unavailable", {})[SERVER_SITE_NAME] = "server log not available for this job"

        try:
            if target_lower not in {SERVER_SITE_NAME, *all_targets}:
                self._add_client_job_log(conn, payload, job_id, target)
            elif target_lower in all_targets:
                self._add_all_client_job_logs(
                    conn, payload, job_id, workspace_zip=workspace_zip, workspace_zip_loaded=workspace_zip_loaded
                )
        except TypeError as e:
            error = secure_format_exception(e)
            conn.append_error(error, meta=make_meta(MetaStatusValue.INTERNAL_ERROR, error))
            return

        conn.append_dict(payload, meta=make_meta(MetaStatusValue.OK))

    def _read_server_job_log(
        self, conn: Connection, job_id: str, workspace_zip: bytes = None, workspace_zip_loaded: bool = False
    ) -> Optional[str]:
        engine = conn.app_ctx
        log_text = self._read_live_server_job_log(engine, job_id)
        if log_text is not None:
            return log_text
        if workspace_zip_loaded:
            return self._extract_server_log_from_workspace_zip(workspace_zip)
        return self._read_stored_server_job_log(conn, job_id)

    def _read_live_server_job_log(self, engine, job_id: str) -> Optional[str]:
        workspace = engine.get_workspace()
        log_file = os.path.join(workspace.get_log_root(job_id), WorkspaceConstants.LOG_FILE_NAME)
        try:
            if os.path.exists(log_file):
                return "".join(self._collect_job_log_lines(log_file))
        except FileNotFoundError:
            # The log file can disappear between the existence check and the open() if the
            # active run rotates or cleans up the workspace. Treat that as unavailable
            # rather than surfacing an internal error to the admin client.
            return None
        return None

    def _read_stored_server_job_log(self, conn: Connection, job_id: str) -> Optional[str]:
        workspace_zip = self._read_stored_workspace_zip(conn, job_id)
        return self._extract_server_log_from_workspace_zip(workspace_zip)

    def _read_stored_workspace_zip(self, conn: Connection, job_id: str) -> Optional[bytes]:
        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager
        if not isinstance(job_def_manager, JobDefManagerSpec):
            return None

        with engine.new_context() as fl_ctx:
            try:
                workspace_zip = job_def_manager.get_storage_component(jid=job_id, component=WORKSPACE, fl_ctx=fl_ctx)
            except StorageException:
                return None

        return workspace_zip

    def _extract_server_log_from_workspace_zip(self, workspace_zip: bytes) -> Optional[str]:
        if not workspace_zip:
            return None

        try:
            with ZipFile(io.BytesIO(workspace_zip), "r") as zip_file:
                log_name = self._find_server_log_member(zip_file.namelist())
                if not log_name:
                    return None
                with zip_file.open(log_name, "r") as log_file:
                    data = log_file.read(self.MAX_RETURNED_JOB_LOG_BYTES + 1)
        except (BadZipFile, KeyError, OSError, TypeError):
            return None

        return self._decode_job_log_data(data)

    @staticmethod
    def _find_server_log_member(member_names: List[str]) -> Optional[str]:
        log_file_name = WorkspaceConstants.LOG_FILE_NAME
        if log_file_name in member_names:
            return log_file_name
        server_log_name = f"{SERVER_SITE_NAME}/{log_file_name}"
        if server_log_name in member_names:
            return server_log_name
        suffix = f"/{server_log_name}"
        for member_name in member_names:
            if not member_name.endswith("/") and member_name.endswith(suffix):
                return member_name
        return None

    def _add_client_job_log(self, conn: Connection, payload: dict, job_id: str, client_name: str):
        text = self._read_client_job_log(conn, job_id, client_name)
        if text is None:
            payload.setdefault("unavailable", {})[client_name] = "client log stream not available for this job"
        else:
            payload["logs"][client_name] = text

    def _add_all_client_job_logs(
        self,
        conn: Connection,
        payload: dict,
        job_id: str,
        workspace_zip: bytes = None,
        workspace_zip_loaded: bool = False,
    ):
        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager
        if not isinstance(job_def_manager, JobDefManagerSpec):
            raise TypeError(
                f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
            )

        with engine.new_context() as fl_ctx:
            available_sites = self._get_available_client_log_sites(job_def_manager, job_id, fl_ctx)
            if workspace_zip_loaded:
                workspace_client_logs = self._extract_client_logs_from_workspace_zip(workspace_zip)
            else:
                # Defensive path for direct helper calls. get_job_log preloads the workspace
                # ZIP before calling this for --site all.
                workspace_client_logs = self._read_workspace_client_job_logs(job_def_manager, job_id, fl_ctx)
            available_sites.update(workspace_client_logs.keys())

        known_sites = self._get_job_client_targets(conn.get_prop(self.JOB))
        for client_name in sorted(known_sites | available_sites):
            text = self._read_live_client_job_log(engine, job_id, client_name)
            if text is None:
                text = workspace_client_logs.get(client_name)
            if text is None:
                with engine.new_context() as fl_ctx:
                    data = job_def_manager.get_client_data(
                        jid=job_id,
                        client_name=client_name,
                        data_type=self._client_log_data_type(),
                        fl_ctx=fl_ctx,
                    )
                text = self._decode_job_log_data(data)
            if text is not None:
                payload["logs"][client_name] = text

        missing_sites = known_sites - set(payload["logs"].keys()) - {SERVER_SITE_NAME}
        if missing_sites:
            unavailable = payload.setdefault("unavailable", {})
            for client_name in sorted(missing_sites):
                unavailable[client_name] = "client log stream not available for this job"

    def _read_client_job_log(self, conn: Connection, job_id: str, client_name: str) -> Optional[str]:
        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager
        if not isinstance(job_def_manager, JobDefManagerSpec):
            raise TypeError(
                f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
            )

        text = self._read_live_client_job_log(engine, job_id, client_name)
        if text is not None:
            return text

        text = self._read_stored_client_job_log(conn, job_id, client_name)
        if text is not None:
            return text

        with engine.new_context() as fl_ctx:
            data = job_def_manager.get_client_data(
                jid=job_id,
                client_name=client_name,
                data_type=self._client_log_data_type(),
                fl_ctx=fl_ctx,
            )
        return self._decode_job_log_data(data)

    def _read_live_client_job_log(self, engine, job_id: str, client_name: str) -> Optional[str]:
        workspace = engine.get_workspace()
        log_file = os.path.join(workspace.get_log_root(job_id), client_name, WorkspaceConstants.LOG_FILE_NAME)
        try:
            if os.path.exists(log_file):
                return "".join(self._collect_job_log_lines(log_file))
        except FileNotFoundError:
            return None
        return None

    def _read_stored_client_job_log(self, conn: Connection, job_id: str, client_name: str) -> Optional[str]:
        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager
        if not isinstance(job_def_manager, JobDefManagerSpec):
            return None

        with engine.new_context() as fl_ctx:
            try:
                workspace_zip = job_def_manager.get_storage_component(jid=job_id, component=WORKSPACE, fl_ctx=fl_ctx)
            except StorageException:
                return None

        return self._extract_client_log_from_workspace_zip(workspace_zip, client_name)

    def _extract_client_log_from_workspace_zip(self, workspace_zip: bytes, client_name: str) -> Optional[str]:
        if not workspace_zip:
            return None

        try:
            with ZipFile(io.BytesIO(workspace_zip), "r") as zip_file:
                log_name = self._find_client_log_member(zip_file.namelist(), client_name)
                if not log_name:
                    return None
                with zip_file.open(log_name, "r") as log_file:
                    data = log_file.read(self.MAX_RETURNED_JOB_LOG_BYTES + 1)
        except (BadZipFile, KeyError, OSError, TypeError):
            return None

        return self._decode_job_log_data(data)

    @staticmethod
    def _find_client_log_member(member_names: List[str], client_name: str) -> Optional[str]:
        log_file_name = WorkspaceConstants.LOG_FILE_NAME
        client_log_name = f"{client_name}/{log_file_name}"
        if client_log_name in member_names:
            return client_log_name
        suffix = f"/{client_log_name}"
        for member_name in member_names:
            if not member_name.endswith("/") and member_name.endswith(suffix):
                return member_name
        return None

    @staticmethod
    def _client_log_data_type() -> str:
        return f"{DataTypes.LOG.value}_{WorkspaceConstants.LOG_FILE_NAME}"

    def _get_available_client_log_sites(self, job_def_manager, job_id: str, fl_ctx) -> Set[str]:
        component_prefix = f"{self._client_log_data_type()}_"
        components = job_def_manager.list_components(jid=job_id, fl_ctx=fl_ctx) or []
        return {
            component[len(component_prefix) :]
            for component in components
            if component.startswith(component_prefix) and component[len(component_prefix) :]
        }

    def _read_workspace_client_job_logs(self, job_def_manager, job_id: str, fl_ctx) -> Dict[str, str]:
        try:
            workspace_zip = job_def_manager.get_storage_component(jid=job_id, component=WORKSPACE, fl_ctx=fl_ctx)
        except StorageException:
            return {}

        return self._extract_client_logs_from_workspace_zip(workspace_zip)

    def _extract_client_logs_from_workspace_zip(self, workspace_zip: bytes) -> Dict[str, str]:
        if not workspace_zip:
            return {}

        try:
            with ZipFile(io.BytesIO(workspace_zip), "r") as zip_file:
                log_members = self._find_workspace_client_log_members(zip_file.namelist())
                logs = {}
                for client_name, log_name in log_members.items():
                    try:
                        with zip_file.open(log_name, "r") as log_file:
                            data = log_file.read(self.MAX_RETURNED_JOB_LOG_BYTES + 1)
                    except (KeyError, OSError):
                        continue
                    text = self._decode_job_log_data(data)
                    if text is not None:
                        logs[client_name] = text
                return logs
        except (BadZipFile, OSError, TypeError):
            return {}

    @staticmethod
    def _find_workspace_client_log_sites(member_names: List[str]) -> Set[str]:
        return set(JobCommandModule._find_workspace_client_log_members(member_names))

    @staticmethod
    def _find_workspace_client_log_members(member_names: List[str]) -> Dict[str, str]:
        members = {}
        exact_members = {}
        log_file_name = WorkspaceConstants.LOG_FILE_NAME
        for member_name in member_names:
            if member_name.endswith("/") or not member_name.endswith(f"/{log_file_name}"):
                continue
            parts = member_name.split("/")
            if len(parts) >= 2 and parts[-2] and parts[-2] != SERVER_SITE_NAME:
                if len(parts) == 2:
                    exact_members[parts[-2]] = member_name
                elif parts[-2] not in members:
                    members[parts[-2]] = member_name
        # Prefer exact two-part paths (client_name/fl.log) over deeper paths
        # (run_1/client_name/fl.log) by letting exact_members overwrite members.
        members.update(exact_members)
        return members

    @staticmethod
    def _get_job_client_targets(job) -> Set[str]:
        if not job or not getattr(job, "meta", None):
            return set()

        deploy_map = job.meta.get(JobMetaKey.DEPLOY_MAP.value, {})
        if not isinstance(deploy_map, dict):
            return set()

        targets = set()
        for deployments in deploy_map.values():
            for site_name in extract_participants(deployments):
                if not site_name or site_name == SERVER_SITE_NAME or site_name.upper() == ALL_SITES:
                    continue
                targets.add(site_name)
        return targets

    def _decode_job_log_data(self, data) -> Optional[str]:
        if data is None:
            return None
        if isinstance(data, str):
            raw_data = data.encode("utf-8", errors="replace")
        else:
            raw_data = bytes(data)

        truncated = len(raw_data) > self.MAX_RETURNED_JOB_LOG_BYTES
        if truncated:
            raw_data = raw_data[: self.MAX_RETURNED_JOB_LOG_BYTES]
        text = raw_data.decode("utf-8", errors="replace")
        if truncated:
            text += f"\n... output truncated after {self.MAX_RETURNED_JOB_LOG_BYTES} bytes ...\n"
        return text

    def _collect_job_log_lines(self, log_file: str):
        lines = []
        collected_bytes = 0
        truncated_by_bytes = False

        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_len = len(line.encode("utf-8", errors="replace"))
                if collected_bytes + line_len > self.MAX_RETURNED_JOB_LOG_BYTES:
                    truncated_by_bytes = True
                    break
                collected_bytes += line_len
                lines.append(line)

        if truncated_by_bytes:
            lines.append(f"... output truncated after {self.MAX_RETURNED_JOB_LOG_BYTES} bytes ...\n")
        return lines

    def list_job_components(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error("Usage: list_job_components job_id", meta=make_meta(MetaStatusValue.SYNTAX_ERROR))
            return

        job_id = conn.get_prop(self.JOB_ID)
        engine = conn.app_ctx
        job_def_manager = engine.job_def_manager
        if not isinstance(job_def_manager, JobDefManagerSpec):
            raise TypeError(
                f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
            )
        with engine.new_context() as fl_ctx:
            list_of_data = job_def_manager.list_components(jid=job_id, fl_ctx=fl_ctx)
            if list_of_data:
                system_components = {"workspace", "meta", "scheduled", "data"}
                filtered_data = [item for item in list_of_data if item not in system_components]
                if filtered_data:
                    data_str = ", ".join(filtered_data)
                    conn.append_string(data_str)
                    conn.append_success(
                        "", meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOB_COMPONENTS: filtered_data})
                    )
                else:
                    conn.append_error(
                        "No additional job components found.",
                        meta=make_meta(MetaStatusValue.NO_JOB_COMPONENTS, "No additional job components found."),
                    )
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
                if job_status in [RunStatus.SUBMITTED.value, RunStatus.DISPATCHED.value]:
                    job_manager.set_status(job.job_id, RunStatus.FINISHED_ABORTED, fl_ctx)
                    message = f"Aborted the job {job_id} before running it."
                    conn.append_string(message)
                    conn.append_success("", meta=make_meta(MetaStatusValue.OK, message))
                    return
                elif job_status and job_status.startswith("FINISHED:"):
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
                job_meta[JobMetaKey.SUBMITTER_ROLE.value] = conn.get_prop(ConnProps.USER_ROLE, "")
                job_meta[JobMetaKey.CLONED_FROM.value] = job_id
                job_meta[JobMetaKey.STUDY.value] = get_job_meta_study(job.meta)

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
    def _job_match(job_meta: Dict, id_prefix: str, name_prefix: str, user_name: str, requested_study: str) -> bool:
        job_id = (job_meta.get(JobMetaKey.JOB_ID.value) or "").lower()
        job_name = (job_meta.get(JobMetaKey.JOB_NAME.value) or "").lower()
        return (
            ((not id_prefix) or job_id.startswith(id_prefix.lower()))
            and ((not name_prefix) or job_name.startswith(name_prefix.lower()))
            and ((not user_name) or job_meta.get(JobMetaKey.SUBMITTER_NAME.value) == user_name)
            and get_job_meta_study(job_meta) == requested_study
        )

    @staticmethod
    def _submitter_from_conn(conn: Connection) -> dict:
        return {
            "name": conn.get_prop(ConnProps.USER_NAME, ""),
            "org": conn.get_prop(ConnProps.USER_ORG, ""),
            "role": conn.get_prop(ConnProps.USER_ROLE, ""),
        }

    @staticmethod
    def _canonical_job_content_hash(zip_file_name: str) -> str:
        return canonical_job_content_hash(zip_file_name)

    @staticmethod
    def _require_submit_record_method(job_def_manager: JobDefManagerSpec, method_name: str):
        method = getattr(job_def_manager, method_name, None)
        if not callable(method):
            raise RuntimeError(f"job_def_manager does not implement required submit-token method '{method_name}'")
        return method

    def _get_submit_record(
        self, job_def_manager: JobDefManagerSpec, study: str, submitter: dict, submit_token: str, fl_ctx
    ):
        method = self._require_submit_record_method(job_def_manager, "get_submit_record")
        return method(study, submitter, submit_token, fl_ctx)

    def _create_submit_record(self, job_def_manager: JobDefManagerSpec, record: dict, fl_ctx):
        method = self._require_submit_record_method(job_def_manager, "create_submit_record")
        return method(record, fl_ctx)

    def _update_submit_record(self, job_def_manager: JobDefManagerSpec, record: dict, fl_ctx):
        method = self._require_submit_record_method(job_def_manager, "update_submit_record")
        return method(record, fl_ctx)

    def _new_submit_record(
        self,
        job_def_manager: JobDefManagerSpec,
        *,
        study: str,
        submitter: dict,
        submit_token: str,
        job_content_hash: str,
        job_name: str,
        job_folder_name: str,
        state: str,
    ) -> dict:
        method = self._require_submit_record_method(job_def_manager, "new_submit_record")
        return method(
            study=study,
            submitter=submitter,
            submit_token=submit_token,
            job_content_hash=job_content_hash,
            job_name=job_name,
            job_folder_name=job_folder_name,
            state=state,
        )

    @staticmethod
    def _append_submit_token_conflict(conn: Connection, record: dict):
        existing_job_id = record.get(SubmitRecordKey.JOB_ID.value) if isinstance(record, dict) else None
        extra = {"code": "SUBMIT_TOKEN_CONFLICT"}
        if existing_job_id:
            extra[MetaKey.JOB_ID] = existing_job_id
        conn.append_error(
            "SUBMIT_TOKEN_CONFLICT: submit token was already used for different job content. "
            "Use a new submit token for a new job.",
            meta=make_meta(
                _SUBMIT_TOKEN_CONFLICT_STATUS,
                "submit token was already used for different job content",
                extra=extra,
            ),
        )

    def _job_for_submit_record(self, job_def_manager: JobDefManagerSpec, record: dict, fl_ctx):
        if not isinstance(record, dict):
            return None
        job_id = record.get(SubmitRecordKey.JOB_ID.value)
        if not job_id:
            return None
        return job_def_manager.get_job(job_id, fl_ctx)

    @staticmethod
    def _job_id_from_job(job) -> Optional[str]:
        if not job:
            return None
        job_id = getattr(job, "job_id", None)
        if job_id:
            return job_id
        meta = getattr(job, "meta", None)
        if isinstance(meta, dict):
            return meta.get(JobMetaKey.JOB_ID.value)
        return None

    @classmethod
    def _submit_token_lock(cls, study: str, submitter: dict, submit_token: str):
        key = (
            study,
            submitter.get("name", ""),
            submitter.get("org", ""),
            submitter.get("role", ""),
            submit_token,
        )
        with cls._submit_token_locks_guard:
            lock = cls._submit_token_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                cls._submit_token_locks[key] = lock
            return lock

    def _handle_submit_token_record(
        self,
        conn: Connection,
        job_def_manager: JobDefManagerSpec,
        *,
        study: str,
        submitter: dict,
        submit_token: str,
        job_content_hash: str,
        meta: dict,
        folder_name: str,
        zip_file_name: str,
        fl_ctx,
    ):
        with self._submit_token_lock(study, submitter, submit_token):
            return self._handle_submit_token_record_locked(
                conn,
                job_def_manager,
                study=study,
                submitter=submitter,
                submit_token=submit_token,
                job_content_hash=job_content_hash,
                meta=meta,
                folder_name=folder_name,
                zip_file_name=zip_file_name,
                fl_ctx=fl_ctx,
            )

    def _handle_submit_token_record_locked(
        self,
        conn: Connection,
        job_def_manager: JobDefManagerSpec,
        *,
        study: str,
        submitter: dict,
        submit_token: str,
        job_content_hash: str,
        meta: dict,
        folder_name: str,
        zip_file_name: str,
        fl_ctx,
    ):
        record = self._get_submit_record(job_def_manager, study, submitter, submit_token, fl_ctx)
        if record:
            if record.get(SubmitRecordKey.JOB_CONTENT_HASH.value) != job_content_hash:
                self._append_submit_token_conflict(conn, record)
                return None
            job = self._job_for_submit_record(job_def_manager, record, fl_ctx)
            existing_job_id = self._job_id_from_job(job)
            if existing_job_id:
                return existing_job_id
            recorded_job_id = record.get(SubmitRecordKey.JOB_ID.value)
            if not recorded_job_id:
                raise RuntimeError("submit record is missing job_id")
            meta[JobMetaKey.JOB_ID.value] = recorded_job_id
            created_meta = job_def_manager.create(meta, zip_file_name, fl_ctx)
            record[SubmitRecordKey.STATE.value] = SubmitRecordState.CREATED.value
            self._update_submit_record(job_def_manager, record, fl_ctx)
            return created_meta.get(JobMetaKey.JOB_ID.value)

        record = self._new_submit_record(
            job_def_manager,
            study=study,
            submitter=submitter,
            submit_token=submit_token,
            job_content_hash=job_content_hash,
            job_name=CommandUtil.get_job_name(meta),
            job_folder_name=folder_name,
            state=SubmitRecordState.CREATING.value,
        )
        job_id = record.get(SubmitRecordKey.JOB_ID.value)
        if not job_id:
            raise RuntimeError("submit record is missing job_id")
        meta[JobMetaKey.JOB_ID.value] = job_id
        try:
            create_result = self._create_submit_record(job_def_manager, record, fl_ctx)
        except Exception:
            existing = self._get_submit_record(job_def_manager, study, submitter, submit_token, fl_ctx)
            if not existing:
                raise
            if existing.get(SubmitRecordKey.JOB_CONTENT_HASH.value) != job_content_hash:
                self._append_submit_token_conflict(conn, existing)
                return None
            job = self._job_for_submit_record(job_def_manager, existing, fl_ctx)
            existing_job_id = self._job_id_from_job(job)
            if existing_job_id:
                return existing_job_id
            meta[JobMetaKey.JOB_ID.value] = existing.get(SubmitRecordKey.JOB_ID.value)
            created_meta = job_def_manager.create(meta, zip_file_name, fl_ctx)
            existing[SubmitRecordKey.STATE.value] = SubmitRecordState.CREATED.value
            self._update_submit_record(job_def_manager, existing, fl_ctx)
            return created_meta.get(JobMetaKey.JOB_ID.value)
        if create_result is False:
            existing = self._get_submit_record(job_def_manager, study, submitter, submit_token, fl_ctx)
            if not existing:
                raise RuntimeError("submit record creation failed and no existing record was found")
            if existing.get(SubmitRecordKey.JOB_CONTENT_HASH.value) != job_content_hash:
                self._append_submit_token_conflict(conn, existing)
                return None
            job = self._job_for_submit_record(job_def_manager, existing, fl_ctx)
            resolved_job_id = self._job_id_from_job(job)
            if resolved_job_id:
                return resolved_job_id
            existing_job_id = existing.get(SubmitRecordKey.JOB_ID.value)
            if not existing_job_id:
                raise RuntimeError("submit record is missing job_id")
            meta[JobMetaKey.JOB_ID.value] = existing_job_id
            created_meta = job_def_manager.create(meta, zip_file_name, fl_ctx)
            existing[SubmitRecordKey.STATE.value] = SubmitRecordState.CREATED.value
            self._update_submit_record(job_def_manager, existing, fl_ctx)
            return created_meta.get(JobMetaKey.JOB_ID.value)

        created_meta = job_def_manager.create(meta, zip_file_name, fl_ctx)
        record[SubmitRecordKey.STATE.value] = SubmitRecordState.CREATED.value
        self._update_submit_record(job_def_manager, record, fl_ctx)
        return created_meta.get(JobMetaKey.JOB_ID.value)

    def _list_jobs_by_submit_token(
        self,
        job_def_manager: JobDefManagerSpec,
        study: str,
        submitter: dict,
        submit_token: str,
        fl_ctx,
    ) -> List[Job]:
        get_job_by_submit_token = getattr(job_def_manager, "get_job_by_submit_token", None)
        if callable(get_job_by_submit_token):
            job = get_job_by_submit_token(study, submitter, submit_token, fl_ctx)
            if not job:
                return []
            return job if isinstance(job, list) else [job]

        record = self._get_submit_record(job_def_manager, study, submitter, submit_token, fl_ctx)
        job = self._job_for_submit_record(job_def_manager, record, fl_ctx)
        return [job] if job else []

    @staticmethod
    def _send_detail_list(conn: Connection, jobs: List[Job]):
        list_of_jobs = []
        for job in jobs:
            try:
                JobCommandModule._set_duration(job)
            except Exception:
                pass
            normalized_meta = dict(job.meta)
            normalized_meta[JobMetaKey.STUDY.value] = get_job_meta_study(job.meta)
            conn.append_string(json.dumps(normalized_meta, indent=4))
            list_of_jobs.append(normalized_meta)
        conn.append_string("", meta=make_meta(MetaStatusValue.OK, extra={MetaKey.JOBS: list_of_jobs}))

    @staticmethod
    def _send_summary_list(conn: Connection, jobs: List[Job]):
        table = conn.append_table(["Job ID", "Name", "Status", "Submit Time", "Run Duration"], name=MetaKey.JOBS)
        for job in jobs:
            try:
                JobCommandModule._set_duration(job)
            except Exception:
                pass
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
                    JobMetaKey.STUDY.value: get_job_meta_study(job.meta),
                },
            )

    @staticmethod
    def _set_duration(job):
        if job.meta.get(JobMetaKey.STATUS) == RunStatus.RUNNING.value:
            try:
                start_time = datetime.datetime.strptime(
                    job.meta.get(JobMetaKey.START_TIME.value), "%Y-%m-%d %H:%M:%S.%f"
                )
            except (TypeError, ValueError):
                return
            duration = datetime.datetime.now() - start_time
            job.meta[JobMetaKey.DURATION.value] = str(duration)

    def submit_job(self, conn: Connection, args: List[str]):
        try:
            parser = _create_submit_job_cmd_parser()
            parsed_args = parser.parse_args(args[1:])
            folder_name = parsed_args.folder_name
            submit_token = _validate_submit_token(parsed_args.submit_token)
        except ValueError as e:
            conn.append_error(str(e), meta=make_meta(MetaStatusValue.SYNTAX_ERROR, str(e)))
            return

        zip_file_name = conn.get_prop(ConnProps.FILE_LOCATION)
        if not zip_file_name:
            conn.append_error("missing upload file", meta=make_meta(MetaStatusValue.INTERNAL_ERROR))
            return

        engine = conn.app_ctx
        try:
            with engine.new_context() as fl_ctx:
                job_validator = JobMetaValidator()
                valid, error, meta = job_validator.validate(folder_name, zip_file_name)
                if not valid:
                    conn.append_error(error, meta=make_meta(MetaStatusValue.INVALID_JOB_DEFINITION, error))
                    return

                # Strip privileged meta keys that must only be set by internal server components.
                # A user-submitted job is never "from hub site" — only HubAppDeployer sets this flag
                # after verifying the job, and it operates on the server side.
                meta.pop(JobMetaKey.FROM_HUB_SITE.value, None)
                # Submit-token is server-owned submission metadata. User job metadata must not expose it.
                meta.pop(SubmitRecordKey.SUBMIT_TOKEN.value, None)

                job_def_manager = engine.job_def_manager
                if not isinstance(job_def_manager, JobDefManagerSpec):
                    raise TypeError(
                        f"job_def_manager in engine is not of type JobDefManagerSpec, but got {type(job_def_manager)}"
                    )

                meta[JobMetaKey.STUDY.value] = _active_study_from_conn(conn)
                registry = StudyRegistryService.get_registry()
                requested_study = meta[JobMetaKey.STUDY.value]
                if registry:
                    enrolled_sites = registry.get_sites(requested_study)
                    if enrolled_sites is not None:
                        deploy_map = meta.get(JobMetaKey.DEPLOY_MAP.value, {})
                        invalid_sites = []
                        seen_invalid_sites = set()
                        for deployments in deploy_map.values():
                            for site_name in extract_participants(deployments):
                                if site_name == SERVER_SITE_NAME or site_name.upper() == ALL_SITES:
                                    continue
                                if site_name not in enrolled_sites:
                                    if site_name not in seen_invalid_sites:
                                        invalid_sites.append(site_name)
                                        seen_invalid_sites.add(site_name)
                        if invalid_sites:
                            if len(invalid_sites) == 1:
                                error = f"site '{invalid_sites[0]}' is not enrolled in study '{requested_study}'"
                            else:
                                quoted_names = ", ".join(f"'{name}'" for name in invalid_sites)
                                error = f"sites {quoted_names} are not enrolled in study '{requested_study}'"
                            conn.append_error(error, meta=make_meta(MetaStatusValue.INVALID_JOB_DEFINITION, error))
                            return

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
                submitter = self._submitter_from_conn(conn)
                meta[JobMetaKey.SUBMITTER_NAME.value] = submitter["name"]
                meta[JobMetaKey.SUBMITTER_ORG.value] = submitter["org"]
                meta[JobMetaKey.SUBMITTER_ROLE.value] = submitter["role"]
                meta[JobMetaKey.JOB_FOLDER_NAME.value] = folder_name
                custom_props = conn.get_prop(ConnProps.CUSTOM_PROPS)
                if custom_props:
                    meta[JobMetaKey.CUSTOM_PROPS.value] = custom_props

                if submit_token:
                    job_content_hash = self._canonical_job_content_hash(zip_file_name)
                    job_id = self._handle_submit_token_record(
                        conn,
                        job_def_manager,
                        study=requested_study,
                        submitter=submitter,
                        submit_token=submit_token,
                        job_content_hash=job_content_hash,
                        meta=meta,
                        folder_name=folder_name,
                        zip_file_name=zip_file_name,
                        fl_ctx=fl_ctx,
                    )
                    if job_id is None:
                        return
                else:
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

    def _download_job_comps(self, conn: Connection, args: List[str], get_comps_f):
        """
        Job download uses binary protocol for more efficient download.
        - Retrieve job data from job store. This puts job files (meta, data, and workspace) in a transfer folder
        - Returns job file names, a TX ID, and a command name for downloading files to the admin client
        - Admin client downloads received file names one by one. It signals the end of download in the last command.
        """
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
        job_id = args[1]

        with engine.new_context() as fl_ctx:
            comps = get_comps_f(job_def_manager, job_id, fl_ctx)
            if not comps:
                conn.append_string(
                    "No components to download",
                    meta=make_meta(
                        MetaStatusValue.NO_JOB_COMPONENTS,
                        info="No components to download",
                    ),
                )
                return

            try:
                for ct, file_name in comps:
                    job_def_manager.get_storage_for_download(job_id, job_download_dir, ct, file_name, fl_ctx)

                self.download_folder(
                    conn,
                    tx_id=tx_id,
                    folder_name=job_id,
                )
            except Exception as e:
                secure_log_traceback()
                self.logger.error(f"exception downloading job {job_id}: {secure_format_exception(e)}")
                self._clean_up_download(conn, tx_id)
                conn.append_error("internal error", meta=make_meta(MetaStatusValue.INTERNAL_ERROR))

    def _get_default_job_components(self, job_def_manager, job_id, fl_ctx):
        return [(DATA, JOB_ZIP), (META, META_JSON), (WORKSPACE, WORKSPACE_ZIP)]

    def download_job(self, conn: Connection, args: List[str]):
        """
        Job download uses binary protocol for more efficient download.
        - Retrieve job data from job store. This puts job files (meta, data, and workspace) in a transfer folder
        - Returns job file names, a TX ID, and a command name for downloading files to the admin client
        - Admin client downloads received file names one by one. It signals the end of download in the last command.
        """
        self._download_job_comps(conn, args, self._get_default_job_components)

    def download_job_components(self, conn: Connection, args: List[str]):
        """Download additional job components (e.g., ERRORLOG_site-1) for a specified job.

        Based on job download but downloads the additional components for a job that job download does
        not download.
        """
        self._download_job_comps(conn, args, self._get_extra_job_components)

    def _get_extra_job_components(self, job_def_manager, job_id, fl_ctx):
        all_components = job_def_manager.list_components(jid=job_id, fl_ctx=fl_ctx)
        if all_components:
            return [
                (item, item)
                for item in all_components
                if item not in {WORKSPACE, META, "scheduled", DATA} and StorageSpec.is_valid_component(item)
            ]
        else:
            return None

    def do_app_command(self, conn: Connection, args: List[str]):
        # cmd job_id topic
        if len(args) != 3:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_error(f"Usage: {cmd_entry.usage}", meta=make_meta(MetaStatusValue.SYNTAX_ERROR, ""))
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
        if timeout is None:
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
        conn.append_dict(reply, meta=make_meta(MetaStatusValue.OK))
