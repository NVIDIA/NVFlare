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

import os
import time
from typing import List, Optional

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.api import AdminAPI, APIStatus, ResultKey
from nvflare.fuel.hci.client.overseer_service_finder import ServiceFinderByOverseer
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue

from .api_spec import (
    AuthenticationError,
    AuthorizationError,
    ClientInfo,
    InternalError,
    InvalidArgumentError,
    InvalidJobDefinition,
    JobInfo,
    JobNotDone,
    JobNotFound,
    MonitorReturnCode,
    NoConnection,
    ServerInfo,
    SessionClosed,
    SessionSpec,
    SystemInfo,
)
from .config import FLAdminClientStarterConfigurator


class Session(SessionSpec):
    def __init__(self, username: str = None, startup_path: str = None, secure_mode: bool = True, debug: bool = False):
        """Initializes a session with the NVFLARE system

        Args:
            username: string of username to log in with
            startup_path: path to the provisioned startup kit, which contains endpoint of the system
            secure_mode: whether to run in secure mode or not
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

        # Connect with admin client
        if conf.overseer_agent:
            service_finder = ServiceFinderByOverseer(conf.overseer_agent)
        else:
            service_finder = None

        self.api = AdminAPI(
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            upload_dir=upload_dir,
            download_dir=download_dir,
            service_finder=service_finder,
            user_name=username,
            poc=(not self.secure_mode),
            debug=debug,
        )
        self.upload_dir = upload_dir
        self.download_dir = download_dir

    def close(self):
        """Close the session

        Returns:

        """
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

    def _do_command(self, command: str):
        if self.api.closed:
            raise SessionClosed("session closed")

        result = self.api.do_command(command)

        if not isinstance(result, dict):
            raise InternalError(f"result from server must be dict but got {type(result)}")

        # check meta status first
        meta = result.get(ResultKey.META, None)
        if not meta:
            raise InternalError("missing meta from result")

        if not isinstance(meta, dict):
            raise InternalError(f"meta must be dict but got {type(meta)}")

        cmd_status = meta.get(MetaKey.STATUS)
        info = meta.get(MetaKey.INFO, "")
        if cmd_status == MetaStatusValue.INVALID_JOB_DEFINITION:
            raise InvalidJobDefinition(f"invalid job definition: {info}")
        elif cmd_status == MetaStatusValue.NOT_AUTHORIZED:
            raise AuthorizationError(f"user not authorized for the action '{command}: {info}'")
        elif cmd_status == MetaStatusValue.SYNTAX_ERROR:
            raise InternalError(f"protocol error: {info}")
        elif cmd_status == MetaStatusValue.INVALID_JOB_ID:
            raise JobNotFound(f"no such job: {info}")
        elif cmd_status == MetaStatusValue.JOB_RUNNING:
            raise JobNotDone(f"job {info} is still running")
        elif cmd_status != MetaStatusValue.OK:
            raise InternalError(f"server internal error ({cmd_status}): {info}")

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

        return result

    def _validate_job_id(self, job_id: str):
        if not job_id:
            raise JobNotFound("job_id is required but not specified.")

        if not isinstance(job_id, str):
            raise JobNotFound(f"invalid job_id {job_id}")

    def clone_job(self, job_id: str) -> str:
        """Create a new job by cloning a specified job

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
        """Submit a predefined job to the NVFLARE system

        Args:
            job_definition_path: path to the folder that defines a NVFLARE job

        Returns: the job id if accepted by the system

        If the submission fails, exception will be raised:

        """
        if not job_definition_path:
            raise InvalidJobDefinition("job_definition_path is required but not specified.")

        if not isinstance(job_definition_path, str):
            raise InvalidJobDefinition(f"job_definition_path must be str but got {type(job_definition_path)}.")

        if not os.path.isdir(job_definition_path):
            if os.path.isdir(os.path.join(self.upload_dir, job_definition_path)):
                job_definition_path = os.path.join(self.upload_dir, job_definition_path)
            else:
                raise InvalidJobDefinition(f"job_definition_path '{job_definition_path}' is not a valid folder")

        result = self._do_command(AdminCommandNames.SUBMIT_JOB + " " + job_definition_path)
        meta = result[ResultKey.META]
        job_id = meta.get(MetaKey.JOB_ID, None)
        if not job_id:
            raise InternalError("server failed to return job id")
        return job_id

    def get_job_meta(self, job_id: str) -> dict:
        """Get the meta info of the specified job

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
        """Get the job info from the server

        Args:
            detailed: True to get the detailed information for each job, False by default
            limit: maximum number of jobs to show, with 0 or None to show all (defaults to None to show all)
            id_prefix: if included, only return jobs with the beginning of the job ID matching the id_prefix
            name_prefix: if included, only return jobs with the beginning of the job name matching the name_prefix
            reverse: if specified, list jobs in the reverse order of submission times
        Returns: a dict of job metadata

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
        """
        Download result of the job

        Args:
            job_id: ID of the job

        Returns: folder path to the location of the job result

        If the job size is smaller than the maximum size set on the server, the job will download to the download_dir
        set in Session through the admin config, and the path to the downloaded result will be returned. If the size
        of the job is larger than the maximum size, the location to download the job will be returned.

        """
        self._validate_job_id(job_id)
        result = self._do_command(AdminCommandNames.DOWNLOAD_JOB + " " + job_id)
        meta = result[ResultKey.META]
        download_job_id = meta.get(MetaKey.JOB_ID, None)
        job_download_url = meta.get(MetaKey.JOB_DOWNLOAD_URL, None)
        if not job_download_url:
            return os.path.join(self.download_dir, download_job_id)
        else:
            return job_download_url

    def abort_job(self, job_id: str):
        """Abort the specified job

        Args:
            job_id: job to be aborted

        Returns: dict of (status, info)

        If the job is already done, no effect;
        If job is not started yet, it will be cancelled and won't be scheduled
        If the job is being executed, it will be aborted

        """
        self._validate_job_id(job_id)
        # result = self._do_command(AdminCommandNames.ABORT_JOB + " " + job_id)
        # return result.get(ResultKey.META, None)
        self._do_command(AdminCommandNames.ABORT_JOB + " " + job_id)

    def delete_job(self, job_id: str):
        """Delete the specified job completely from the system

        Args:
            job_id: job to be deleted

        Returns: None

        The job will be deleted from the job store if the job is not currently running

        """
        self._validate_job_id(job_id)
        self._do_command(AdminCommandNames.DELETE_JOB + " " + job_id)

    def get_system_info(self):
        result = self._do_command(AdminCommandNames.CHECK_STATUS + " server")
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

    def monitor_job(
        self, job_id: str, timeout: float = 0.0, poll_interval: float = 2.0, cb=None, *cb_args, **cb_kwargs
    ) -> MonitorReturnCode:
        """Monitor the job progress until one of the conditions occurs:
         - job is done
         - timeout
         - the status_cb returns False

        Args:
            job_id: the job to be monitored
            timeout: how long to monitor. If 0, never time out.
            poll_interval: how often to poll job status
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
    """This is a sample callback to use with monitor_job that demonstrates how a custom callback can
    be used

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
    """Create a new secure session with NVFLARE system

    Args:
        username: username assigned to the user
        startup_kit_location: path to the provisioned startup folder, the root admin dir containing the startup folder
        debug: enable debug mode
        timeout: how long to try to establish the session

    Returns: a Session object

    """
    session = Session(username=username, startup_path=startup_kit_location, secure_mode=True, debug=debug)

    session.try_connect(timeout)
    return session


def new_insecure_session(startup_kit_location: str, debug: bool = False, timeout: float = 10.0) -> Session:
    """Create a new secure session with NVFLARE system

    Args:
        startup_kit_location: path to the provisioned startup folder
        debug: enable debug mode
        timeout: how long to try to establish the session

    Returns: a Session object

    The username for insecure session is always "admin"

    """
    session = Session(username="admin", startup_path=startup_kit_location, secure_mode=False, debug=debug)

    session.try_connect(timeout)
    return session
