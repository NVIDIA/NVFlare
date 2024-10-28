# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import time
from typing import List, Optional

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.api import AdminAPI, APIStatus, ResultKey
from nvflare.fuel.hci.client.config import FLAdminClientStarterConfigurator
from nvflare.fuel.hci.client.overseer_service_finder import ServiceFinderByOverseer
from nvflare.fuel.hci.cmd_arg_utils import (
    process_targets_into_str,
    validate_file_string,
    validate_options_string,
    validate_path_string,
    validate_required_target_string,
    validate_sp_string,
)
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ProtoKey

from .api_spec import (
    AuthenticationError,
    AuthorizationError,
    ClientInfo,
    ClientsStillRunning,
    InternalError,
    InvalidArgumentError,
    InvalidJobDefinition,
    InvalidTarget,
    JobInfo,
    JobNotDone,
    JobNotFound,
    JobNotRunning,
    MonitorReturnCode,
    NoClientsAvailable,
    NoConnection,
    NoReply,
    ServerInfo,
    SessionClosed,
    SessionSpec,
    SystemInfo,
    TargetType,
)

_VALID_TARGET_TYPES = [TargetType.ALL, TargetType.SERVER, TargetType.CLIENT]


class Session(SessionSpec):
    def __init__(self, username: str = None, startup_path: str = None, secure_mode: bool = True, debug: bool = False):
        """Initializes a session with the NVFLARE system.

        Args:
            username (str): string of username to log in with
            startup_path (str): path to the provisioned startup kit, which contains endpoint of the system
            secure_mode (bool): whether to run in secure mode or not
        """
        assert isinstance(username, str), "username must be str"
        self.username = username
        assert isinstance(startup_path, str), "startup_path must be str"
        self.secure_mode = secure_mode

        assert os.path.isdir(startup_path), f"startup kit does not exist at {startup_path}"

        workspace = Workspace(root_dir=startup_path)
        conf = FLAdminClientStarterConfigurator(workspace)
        conf.configure()

        admin_config = conf.config_data.get("admin", None)
        if not admin_config:
            raise ConfigError("Missing admin section in fed_admin configuration.")

        ca_cert = admin_config.get("ca_cert", "")
        client_cert = admin_config.get("client_cert", "")
        client_key = admin_config.get("client_key", "")

        if admin_config.get("with_ssl"):
            if len(ca_cert) <= 0:
                raise ConfigError("missing CA Cert file name field ca_cert in fed_admin configuration")

            if len(client_cert) <= 0:
                raise ConfigError("missing Client Cert file name field client_cert in fed_admin configuration")

            if len(client_key) <= 0:
                raise ConfigError("missing Client Key file name field client_key in fed_admin configuration")
        else:
            ca_cert = None
            client_key = None
            client_cert = None

        upload_dir = admin_config.get("upload_dir")
        download_dir = admin_config.get("download_dir")
        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        if self.secure_mode:
            if not os.path.isfile(ca_cert):
                raise ConfigError(f"rootCA.pem does not exist at {ca_cert}")

            if not os.path.isfile(client_cert):
                raise ConfigError(f"client.crt does not exist at {client_cert}")

            if not os.path.isfile(client_key):
                raise ConfigError(f"client.key does not exist at {client_key}")

        service_finder = ServiceFinderByOverseer(conf.overseer_agent)

        self.api = AdminAPI(
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            upload_dir=upload_dir,
            download_dir=download_dir,
            service_finder=service_finder,
            user_name=username,
            insecure=(not self.secure_mode),
            debug=debug,
            event_handlers=conf.handlers,
        )
        self.upload_dir = upload_dir
        self.download_dir = download_dir
        self.overseer_agent = conf.overseer_agent

    def close(self):
        """Close the session."""
        self.api.close()

    def try_connect(self, timeout):
        if self.api.closed:
            raise SessionClosed("session closed")

        start_time = time.time()
        while not self.api.is_ready():
            if time.time() - start_time > timeout:
                self.api.close()
                raise NoConnection(f"cannot connect to FLARE in {timeout} seconds")
            time.sleep(0.5)

    def _do_command(self, command: str, enforce_meta=True, props=None):
        if self.api.closed:
            raise SessionClosed("session closed")

        result = self.api.do_command(command, props=props)
        if not isinstance(result, dict):
            raise InternalError(f"result from server must be dict but got {type(result)}")

        # Check meta status if available
        # There are still some commands that do not return meta. But for commands that do return meta, we will check
        # its meta status first.
        meta = result.get(ResultKey.META, None)
        if meta:
            if not isinstance(meta, dict):
                raise InternalError(f"meta must be dict but got {type(meta)}")

            cmd_status = meta.get(MetaKey.STATUS, MetaStatusValue.OK)
            info = meta.get(MetaKey.INFO, "")
            if cmd_status == MetaStatusValue.INVALID_JOB_DEFINITION:
                raise InvalidJobDefinition(f"invalid job definition: {info}")
            elif cmd_status == MetaStatusValue.NOT_AUTHORIZED:
                raise AuthorizationError(f"user not authorized for the action '{command}: {info}'")
            elif cmd_status == MetaStatusValue.NOT_AUTHENTICATED:
                raise AuthenticationError(f"user not authenticated: {info}")
            elif cmd_status == MetaStatusValue.SYNTAX_ERROR:
                raise InternalError(f"syntax error: {info}")
            elif cmd_status == MetaStatusValue.INVALID_JOB_ID:
                raise JobNotFound(f"no such job: {info}")
            elif cmd_status == MetaStatusValue.JOB_RUNNING:
                raise JobNotDone(f"job {info} is still running")
            elif cmd_status == MetaStatusValue.JOB_NOT_RUNNING:
                raise JobNotRunning(f"job {info} is not running")
            elif cmd_status == MetaStatusValue.CLIENTS_RUNNING:
                raise ClientsStillRunning("one or more clients are still running")
            elif cmd_status == MetaStatusValue.NO_CLIENTS:
                raise NoClientsAvailable("no clients available")
            elif cmd_status == MetaStatusValue.INTERNAL_ERROR:
                raise InternalError(f"server internal error: {info}")
            elif cmd_status == MetaStatusValue.INVALID_TARGET:
                raise InvalidTarget(info)
            elif cmd_status == MetaStatusValue.NO_REPLY:
                raise NoReply(info)
            elif cmd_status != MetaStatusValue.OK:
                raise InternalError(f"{cmd_status}: {info}")

        # Then check API Status. There are cases that a command does not return meta or ran into errors before
        # setting meta. Even if the command does return meta, still need to make sure APIStatus is good.
        status = result.get(ResultKey.STATUS, None)
        if not status:
            raise InternalError("missing status in result")

        if status in [APIStatus.ERROR_CERT, APIStatus.ERROR_AUTHENTICATION]:
            raise AuthenticationError(f"user not authenticated: {status}")
        elif status == APIStatus.ERROR_AUTHORIZATION:
            raise AuthorizationError(f"user not authorized for the action '{command}'")
        elif status == APIStatus.ERROR_INACTIVE_SESSION:
            raise SessionClosed("the session is closed on server")
        elif status in [APIStatus.ERROR_PROTOCOL, APIStatus.ERROR_SYNTAX]:
            raise InternalError(f"protocol error: {status}")
        elif status in [APIStatus.ERROR_SERVER_CONNECTION]:
            raise ConnectionError(f"cannot connect to server: {status}")
        elif status != APIStatus.SUCCESS:
            details = result.get(ResultKey.DETAILS, "")
            raise RuntimeError(f"runtime error encountered: {status}: {details}")

        if enforce_meta and not meta:
            raise InternalError("missing meta from result")

        # both API Status and Meta are okay
        return result

    @staticmethod
    def _validate_job_id(job_id: str):
        if not job_id:
            raise JobNotFound("job_id is required but not specified.")

        if not isinstance(job_id, str):
            raise JobNotFound(f"invalid job_id {job_id}")

    def clone_job(self, job_id: str) -> str:
        """Create a new job by cloning a specified job.

        Args:
            job_id: job to be cloned

        Returns: ID of the new job

        """
        self._validate_job_id(job_id)
        result = self._do_command(AdminCommandNames.CLONE_JOB + " " + job_id)
        meta = result[ResultKey.META]
        job_id = meta.get(MetaKey.JOB_ID, None)
        info = meta.get(MetaKey.INFO, "")
        if not job_id:
            raise InternalError(f"server failed to return job id: {info}")
        return job_id

    def submit_job(self, job_definition_path: str) -> str:
        """Submit a predefined job to the NVFLARE system.

        Args:
            job_definition_path: path to the folder that defines a NVFLARE job

        Returns: the job id if accepted by the system

        If the submission fails, an exception will be raised.

        """
        if not job_definition_path:
            raise InvalidJobDefinition("job_definition_path is required but not specified.")

        if not isinstance(job_definition_path, str):
            raise InvalidJobDefinition(f"job_definition_path must be str but got {type(job_definition_path)}.")

        if not os.path.isdir(job_definition_path):
            if os.path.isdir(os.path.join(self.upload_dir, job_definition_path)):
                job_definition_path = os.path.join(self.upload_dir, job_definition_path)
                job_definition_path = os.path.abspath(job_definition_path)
            else:
                raise InvalidJobDefinition(f"job_definition_path '{job_definition_path}' is not a valid folder")

        result = self._do_command(AdminCommandNames.SUBMIT_JOB + " " + job_definition_path)
        meta = result[ResultKey.META]
        job_id = meta.get(MetaKey.JOB_ID, None)
        if not job_id:
            raise InternalError("server failed to return job id")
        return job_id

    def get_job_meta(self, job_id: str) -> dict:
        """Get the meta info of the specified job.

        Args:
            job_id: ID of the job

        Returns: a dict of job metadata

        """
        self._validate_job_id(job_id)
        result = self._do_command(AdminCommandNames.GET_JOB_META + " " + job_id)
        meta = result[ResultKey.META]
        job_meta = meta.get(MetaKey.JOB_META, None)
        if not job_meta:
            raise InternalError("server failed to return job meta")
        return job_meta

    def list_jobs(
        self,
        detailed: bool = False,
        limit: Optional[int] = None,
        id_prefix: str = None,
        name_prefix: str = None,
        reverse: bool = False,
    ) -> List[dict]:
        """Get the job info from the server.

        Args:
            detailed (bool): True to get the detailed information for each job, False by default
            limit (int, optional): maximum number of jobs to show, with 0 or None to show all (defaults to None to show all)
            id_prefix (str): if included, only return jobs with the beginning of the job ID matching the id_prefix
            name_prefix (str): if included, only return jobs with the beginning of the job name matching the name_prefix
            reverse (bool): if specified, list jobs in the reverse order of submission times

        Returns: a list of job metadata

        """
        if not isinstance(detailed, bool):
            raise ValueError(f"detailed must be bool but got {type(detailed)}")
        if not isinstance(reverse, bool):
            raise ValueError(f"reverse must be bool but got {type(reverse)}")
        if limit is not None and not isinstance(limit, int):
            raise ValueError(f"limit must be None or int but got {type(limit)}")
        if id_prefix is not None and not isinstance(id_prefix, str):
            raise ValueError(f"id_prefix must be None or str but got {type(id_prefix)}")
        if name_prefix is not None and not isinstance(name_prefix, str):
            raise ValueError(f"name_prefix must be None or str but got {type(name_prefix)}")

        command = AdminCommandNames.LIST_JOBS
        if detailed:
            command = command + " -d"
        if reverse:
            command = command + " -r"
        if limit:
            if not isinstance(limit, int):
                raise InvalidArgumentError(f"limit must be int but got {type(limit)}")
            command = command + " -m " + str(limit)
        if name_prefix:
            if not isinstance(name_prefix, str):
                raise InvalidArgumentError("name_prefix must be str but got {}.".format(type(name_prefix)))
            else:
                command = command + " -n " + name_prefix
        if id_prefix:
            if not isinstance(id_prefix, str):
                raise InvalidArgumentError("id_prefix must be str but got {}.".format(type(id_prefix)))
            else:
                command = command + " " + id_prefix
        result = self._do_command(command)
        meta = result[ResultKey.META]
        jobs_list = meta.get(MetaKey.JOBS, [])
        return jobs_list

    def download_job_result(self, job_id: str) -> str:
        """Download result of the job.

        Args:
            job_id (str): ID of the job

        Returns: folder path to the location of the job result

        If the job size is smaller than the maximum size set on the server, the job will download to the download_dir
        set in Session through the admin config, and the path to the downloaded result will be returned. If the size
        of the job is larger than the maximum size, the location to download the job will be returned.

        """
        self._validate_job_id(job_id)
        result = self._do_command(AdminCommandNames.DOWNLOAD_JOB + " " + job_id)
        meta = result[ResultKey.META]
        location = meta.get(MetaKey.LOCATION)
        return location

    def abort_job(self, job_id: str):
        """Abort the specified job.

        Args:
            job_id (str): job to be aborted

        Returns: dict of (status, info)

        If the job is already done, no effect;
        If job is not started yet, it will be cancelled and won't be scheduled.
        If the job is being executed, it will be aborted.

        """
        self._validate_job_id(job_id)
        # result = self._do_command(AdminCommandNames.ABORT_JOB + " " + job_id)
        # return result.get(ResultKey.META, None)
        self._do_command(AdminCommandNames.ABORT_JOB + " " + job_id)

    def delete_job(self, job_id: str):
        """Delete the specified job completely from the system.

        Args:
            job_id (str): job to be deleted

        Returns: None

        The job will be deleted from the job store if the job is not currently running.

        """
        self._validate_job_id(job_id)
        self._do_command(AdminCommandNames.DELETE_JOB + " " + job_id)

    def get_system_info(self):
        """Get general system information.

        Returns: a SystemInfo object

        """
        return self._do_get_system_info(AdminCommandNames.CHECK_STATUS)

    def _do_get_system_info(self, cmd: str):
        result = self._do_command(f"{cmd} {TargetType.SERVER}")
        meta = result[ResultKey.META]
        server_info = ServerInfo(status=meta.get(MetaKey.SERVER_STATUS), start_time=meta.get(MetaKey.SERVER_START_TIME))

        clients = []
        client_meta_list = meta.get(MetaKey.CLIENTS, None)
        if client_meta_list:
            for c in client_meta_list:
                client_info = ClientInfo(
                    name=c.get(MetaKey.CLIENT_NAME), last_connect_time=c.get(MetaKey.CLIENT_LAST_CONNECT_TIME)
                )
                clients.append(client_info)

        jobs = []
        job_meta_list = meta.get(MetaKey.JOBS, None)
        if job_meta_list:
            for j in job_meta_list:
                job_info = JobInfo(app_name=j.get(MetaKey.APP_NAME), job_id=j.get(MetaKey.JOB_ID))
                jobs.append(job_info)

        return SystemInfo(server_info=server_info, client_info=clients, job_info=jobs)

    def get_client_job_status(self, client_names: List[str] = None) -> List[dict]:
        """Get job status info of specified FL clients.

        Args:
            client_names (List[str]): names of the clients to get status info

        Returns: A list of jobs running on the clients. Each job is described by a dict of: id, app name and status.
        If there are multiple jobs running on one client, the list contains one entry for each job for that client.
        If no FL clients are connected or the server failed to communicate to them, this method returns None.

        """
        parts = [AdminCommandNames.CHECK_STATUS, TargetType.CLIENT]
        if client_names:
            processed_targets_str = process_targets_into_str(client_names)
            parts.append(processed_targets_str)

        command = " ".join(parts)
        result = self._do_command(command)
        meta = result[ResultKey.META]
        return meta.get(MetaKey.CLIENT_STATUS, None)

    def restart(self, target_type: str, client_names: Optional[List[str]] = None) -> dict:
        """Restart specified system target(s).

        Args:
            target_type (str): what system target (server, client, or all) to restart
            client_names (List[str]): clients to be restarted if target_type is client. If not specified, all clients.

        Returns: a dict that contains detailed info about the restart request:
            status - the overall status of the result.
            server_status - whether the server is restarted successfully - only if target_type is "all" or "server".
            client_status - a dict (keyed on client name) that specifies status of each client - only if target_type
            is "all" or "client".

        """
        if target_type not in _VALID_TARGET_TYPES:
            raise ValueError(f"invalid target_type {target_type} - must be in {_VALID_TARGET_TYPES}")

        parts = [AdminCommandNames.RESTART, target_type]
        if target_type == TargetType.CLIENT and client_names:
            processed_targets_str = process_targets_into_str(client_names)
            parts.append(processed_targets_str)

        command = " ".join(parts)
        result = self._do_command(command)
        return result[ResultKey.META]

    def shutdown(self, target_type: TargetType, client_names: Optional[List[str]] = None):
        """Shut down specified system target(s).

        Args:
            target_type: what system target (server, client, or all) to shut down
            client_names: clients to be shut down if target_type is client. If not specified, all clients.

        Returns: None
        """
        if target_type not in _VALID_TARGET_TYPES:
            raise ValueError(f"invalid target_type {target_type} - must be in {_VALID_TARGET_TYPES}")

        parts = [AdminCommandNames.SHUTDOWN, target_type]
        if target_type == TargetType.CLIENT and client_names:
            processed_targets_str = process_targets_into_str(client_names)
            parts.append(processed_targets_str)

        command = " ".join(parts)
        self._do_command(command)

    def set_timeout(self, value: float):
        """Set a session-specific command timeout.

        This is the amount of time the server will wait for responses after sending commands to FL clients.

        Note that this value is only effective for the current API session.

        Args:
            value (float): a positive float number for the timeout in seconds

        Returns: None

        """
        self.api.set_command_timeout(value)

    def unset_timeout(self):
        """Unset the session-specific command timeout.

        Once unset, the FL Admin Server's default timeout will be used.

        Returns: None

        """
        self.api.unset_command_timeout()

    def list_sp(self) -> dict:
        """List available service providers.

        Returns: a dict that contains information about the primary SP and others

        """
        reply = self._do_command("list_sp", enforce_meta=False)
        return reply.get(ResultKey.DETAILS)

    def get_active_sp(self) -> dict:
        """Get the current active service provider (SP).

        Returns: a dict that describes the current active SP. If no SP is available currently, the 'name' attribute of
        the result is empty.
        """
        reply = self._do_command("get_active_sp", enforce_meta=False)
        return reply.get(ResultKey.META)

    def promote_sp(self, sp_end_point: str):
        """Promote the specified endpoint to become the active SP.

        Args:
            sp_end_point: the endpoint of the SP. It's string in this format: <url>:<server_port>:<admin_port>

        Returns: None

        """
        sp_end_point = validate_sp_string(sp_end_point)
        self._do_command("promote_sp " + sp_end_point)

    def get_available_apps_to_upload(self):
        """Get defined FLARE app folders from the upload folder on the machine the FLARE API is running.

        Returns: a list of app folders

        """
        dir_list = []
        for item in os.listdir(self.upload_dir):
            if os.path.isdir(os.path.join(self.upload_dir, item)):
                dir_list.append(item)
        return dir_list

    def shutdown_system(self):
        """Shutdown the whole NVFLARE system including the overseer, FL server(s), and all FL clients.

        Returns: None

        Note: the user must be a Project Admin to use this method; otherwise the NOT_AUTHORIZED exception will be raised.

        """
        sys_info = self._do_get_system_info(AdminCommandNames.ADMIN_CHECK_STATUS)
        if sys_info.server_info.status != "stopped":
            raise JobNotDone("there are still running jobs")

        resp = self.overseer_agent.set_state("shutdown")
        err = json.loads(resp.text).get("Error")
        if err:
            raise RuntimeError(err)

    def ls_target(self, target: str, options: str = None, path: str = None) -> str:
        """Run the "ls" command on the specified target and return the result.

        Args:
            target: the target (server or a client name) the command will be run on
            options: options of the "ls" command
            path: the optional file path

        Returns: result of "ls" command

        """
        return self._shell_command_on_target("ls", target, options, path)

    def cat_target(self, target: str, options: str = None, file: str = None) -> str:
        """Run the "cat" command on the specified target and return the result.

        Args:
            target: the target (server or a client name) the command will be run on
            options: options of the "cat" command
            file: the file that the "cat" command will run against

        Returns: result of "cat" command

        """
        return self._shell_command_on_target("cat", target, options, file, fp_required=True, fp_type="file")

    def tail_target(self, target: str, options: str = None, file: str = None) -> str:
        """Run the "tail" command on the specified target and return the result.

        Args:
            target: the target (server or a client name) the command will be run on
            options: options of the "tail" command
            file: the file that the "tail" command will run against

        Returns: result of "tail" command

        """
        return self._shell_command_on_target("tail", target, options, file, fp_required=True, fp_type="file")

    def tail_target_log(self, target: str, options: str = None) -> str:
        """Run the "tail log.txt" command on the specified target and return the result.

        Args:
            target: the target (server or a client name) the command will be run on
            options: options of the "tail" command

        Returns: result of "tail" command

        """
        return self.tail_target(target, options, file="log.txt")

    def head_target(self, target: str, options: str = None, file: str = None) -> str:
        """Run the "head" command on the specified target and return the result.

        Args:
            target: the target (server or a client name) the command will be run on
            options: options of the "head" command
            file: the file that the "head" command will run against

        Returns: result of "head" command

        """
        return self._shell_command_on_target("head", target, options, file, fp_required=True, fp_type="file")

    def head_target_log(self, target: str, options: str = None) -> str:
        """Run the "head log.txt" command on the specified target and return the result.

        Args:
            target: the target (server or a client name) the command will be run on
            options: options of the "head" command

        Returns: result of "head" command

        """
        return self.head_target(target, options, file="log.txt")

    def grep_target(self, target: str, options: str = None, pattern: str = None, file: str = None) -> str:
        """Run the "grep" command on the specified target and return the result.

        Args:
            target: the target (server or a client name) the command will be run on
            options: options of the "grep" command
            pattern: the grep pattern
            file: the file that the "grep" command will run against

        Returns: result of "grep" command

        """
        return self._shell_command_on_target(
            "grep", target, options, file, pattern=pattern, pattern_required=True, fp_required=True, fp_type="file"
        )

    def get_working_directory(self, target: str) -> str:
        """Get the working directory of the specified target.

        Args:
            target (str): the target (server of a client name)

        Returns: current working directory of the specified target

        """
        return self._shell_command_on_target("pwd", target, options=None, fp=None)

    def _shell_command_on_target(
        self,
        cmd: str,
        target: str,
        options,
        fp,
        pattern=None,
        pattern_required=False,
        fp_required=False,
        fp_type="path",
    ) -> str:
        target = validate_required_target_string(target)
        parts = [cmd, target]
        if options:
            options = validate_options_string(options)
            parts.append(options)

        if pattern_required:
            if not pattern:
                raise SyntaxError("pattern is required but not specified.")
            if not isinstance(pattern, str):
                raise ValueError("pattern is not str.")
            parts.append('"' + pattern + '"')

        if fp_required and not fp:
            raise SyntaxError(f"{fp_type} is required but not specified.")

        if fp:
            if fp_type == "path":
                validate_path_string(fp)
            else:
                validate_file_string(fp)
            parts.append(fp)
        command = " ".join(parts)
        reply = self._do_command(command, enforce_meta=False)
        return self._get_string_data(reply)

    @staticmethod
    def _get_string_data(reply: dict) -> str:
        result = ""
        data_items = reply.get(ProtoKey.DATA, [])
        for it in data_items:
            if isinstance(it, dict):
                if it.get(ProtoKey.TYPE) == ProtoKey.STRING:
                    result += it.get(ProtoKey.DATA, "")
        return result

    @staticmethod
    def _get_dict_data(reply: dict) -> dict:
        result = {}
        data_items = reply.get(ProtoKey.DATA, [])
        for it in data_items:
            if isinstance(it, dict):
                if it.get(ProtoKey.TYPE) == ProtoKey.DICT:
                    return it.get(ProtoKey.DATA, {})
        return result

    def show_stats(self, job_id: str, target_type: str, targets: Optional[List[str]] = None) -> dict:
        """Show processing stats of specified job on specified targets.

        Args:
            job_id (str): ID of the job
            target_type (str): type of target (server or client)
            targets: list of client names if target type is "client". All clients if not specified.

        Returns: a dict that contains job stats on specified targets. The key of the dict is target name. The value is
        a dict of stats reported by different system components (ServerRunner or ClientRunner).

        """
        return self._collect_info(AdminCommandNames.SHOW_STATS, job_id, target_type, targets)

    def show_errors(self, job_id: str, target_type: str, targets: Optional[List[str]] = None) -> dict:
        """Show processing errors of specified job on specified targets.

        Args:
            job_id (str): ID of the job
            target_type (str): type of target (server or client)
            targets: list of client names if target type is "client". All clients if not specified.

        Returns: a dict that contains job errors (if any) on specified targets. The key of the dict is target name.
        The value is a dict of errors reported by different system components (ServerRunner or ClientRunner).

        """
        return self._collect_info(AdminCommandNames.SHOW_ERRORS, job_id, target_type, targets)

    def reset_errors(self, job_id: str):
        """Clear errors for all system targets for the specified job.

        Args:
            job_id (str): ID of the job

        Returns: None

        """
        self._collect_info(AdminCommandNames.RESET_ERRORS, job_id, TargetType.ALL)

    def _collect_info(self, cmd: str, job_id: str, target_type: str, targets=None) -> dict:
        if not job_id:
            raise ValueError("job_id is required but not specified.")

        if not isinstance(job_id, str):
            raise TypeError("job_id must be str but got {}.".format(type(job_id)))

        if target_type not in _VALID_TARGET_TYPES:
            raise ValueError(f"invalid target_type {target_type}: must be one of {_VALID_TARGET_TYPES}")

        parts = [cmd, job_id, target_type]
        if target_type == TargetType.CLIENT and targets:
            processed_targets_str = process_targets_into_str(targets)
            parts.append(processed_targets_str)

        command = " ".join(parts)
        reply = self._do_command(command, enforce_meta=False)
        return self._get_dict_data(reply)

    def do_app_command(self, job_id: str, topic: str, cmd_data) -> dict:
        """Ask a running job to execute an app command

        Args:
            job_id: the ID of the running job
            topic: topic of the command
            cmd_data: the data of the command. Must be JSON serializable.

        Returns: result of the app command

        If the job is not currently running, an exception will occur. User must make sure that the job is running when
        calling this method.

        """
        command = f"{AdminCommandNames.APP_COMMAND} {job_id} {topic}"
        if cmd_data:
            # cmd_data must be JSON serializable!
            try:
                json.dumps(cmd_data)
            except Exception as ex:
                raise ValueError(f"cmd_data cannot be JSON serialized: {ex}")
        reply = self._do_command(command, enforce_meta=False, props=cmd_data)
        return self._get_dict_data(reply)

    def get_connected_client_list(self) -> List[ClientInfo]:
        """Get the list of connected clients.

        Returns: a list of ClientInfo objects

        """
        sys_info = self.get_system_info()
        return sys_info.client_info

    def get_client_env(self, client_names=None):
        """Get running environment values for specified clients. The env includes values of client name,
        workspace directory, root url of the FL server, and secure mode or not.

        These values can be used for 3rd-party system configuration (e.g. CellPipe to connect to the FLARE system).

        Args:
            client_names: clients to get env from. None means all clients.

        Returns: list of env info for specified clients.

        Raises: InvalidTarget exception, if no clients are connected or an invalid client name is specified

        """
        if not client_names:
            command = AdminCommandNames.REPORT_ENV
        else:
            if isinstance(client_names, str):
                client_names = [client_names]
            elif not isinstance(client_names, list):
                raise ValueError(f"client_names must be str or list of str but got {type(client_names)}")
            command = AdminCommandNames.REPORT_ENV + " " + " ".join(client_names)

        result = self._do_command(command)
        meta = result[ResultKey.META]
        client_envs = meta.get(MetaKey.CLIENTS)
        if not client_envs:
            raise RuntimeError(f"missing {MetaKey.CLIENTS} from meta")
        return client_envs

    def monitor_job(
        self, job_id: str, timeout: float = 0.0, poll_interval: float = 2.0, cb=None, *cb_args, **cb_kwargs
    ) -> MonitorReturnCode:
        """Monitor the job progress.

        Monitors until one of the conditions occurs:
            - job is done
            - timeout
            - the status_cb returns False

        Args:
            job_id (str): the job to be monitored
            timeout (float): how long to monitor. If 0, never time out.
            poll_interval (float): how often to poll job status
            cb: if provided, callback to be called after each status poll

        Returns: a MonitorReturnCode

        Every time the cb is called, it must return a bool indicating whether the monitor
        should continue. If False, this method ends.

        """
        start_time = time.time()
        while True:
            if 0 < timeout < time.time() - start_time:
                return MonitorReturnCode.TIMEOUT

            job_meta = self.get_job_meta(job_id)
            if cb is not None:
                should_continue = cb(self, job_id, job_meta, *cb_args, **cb_kwargs)
                if not should_continue:
                    return MonitorReturnCode.ENDED_BY_CB

            # check whether the job is finished
            job_status = job_meta.get(JobMetaKey.STATUS.value, None)
            if not job_status:
                raise InternalError(f"missing status in job {job_id}")

            if job_status.startswith("FINISHED"):
                return MonitorReturnCode.JOB_FINISHED

            time.sleep(poll_interval)


def basic_cb_with_print(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    """This is a sample callback to use with monitor_job.

    This demonstrates how a custom callback can be used.

    """
    if job_meta["status"] == "RUNNING":
        if cb_kwargs["cb_run_counter"]["count"] < 3:
            print(job_meta)
        else:
            print(".", end="")
    else:
        print("\n" + str(job_meta))

    cb_kwargs["cb_run_counter"]["count"] += 1
    return True


def new_secure_session(username: str, startup_kit_location: str, debug: bool = False, timeout: float = 10.0) -> Session:
    """Create a new secure FLARE API session with the NVFLARE system.

    Args:
        username (str): username assigned to the user
        startup_kit_location (str): path to the provisioned startup folder, the root admin dir containing the startup folder
        debug (bool): enable debug mode
        timeout (float): how long to try to establish the session, in seconds

    Returns: a Session object

    """
    session = Session(username=username, startup_path=startup_kit_location, secure_mode=True, debug=debug)

    session.try_connect(timeout)
    return session


def new_insecure_session(startup_kit_location: str, debug: bool = False, timeout: float = 10.0) -> Session:
    """Create a new insecure FLARE API session with the NVFLARE system.

    Args:
        startup_kit_location (str): path to the provisioned startup folder
        debug (bool): enable debug mode
        timeout (float): how long to try to establish the session, in seconds

    Returns: a Session object

    The username for insecure session is always "admin".

    """
    session = Session(username="admin", startup_path=startup_kit_location, secure_mode=False, debug=debug)

    session.try_connect(timeout)
    return session
