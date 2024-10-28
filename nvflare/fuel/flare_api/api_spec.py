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

import enum
import time
from abc import ABC, abstractmethod
from typing import List, Optional


class MonitorReturnCode(int, enum.Enum):

    JOB_FINISHED = 0
    TIMEOUT = 1
    ENDED_BY_CB = 2


class NoConnection(Exception):
    pass


class SessionClosed(Exception):
    pass


class InvalidArgumentError(Exception):
    pass


class InvalidJobDefinition(Exception):
    pass


class JobNotFound(Exception):
    pass


class JobNotRunning(Exception):
    pass


class JobNotDone(Exception):
    pass


class InternalError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class NoClientsAvailable(Exception):
    pass


class ClientsStillRunning(Exception):
    pass


class InvalidTarget(Exception):
    pass


class NoReply(Exception):
    pass


class TargetType:
    ALL = "all"
    SERVER = "server"
    CLIENT = "client"


class ServerInfo:
    def __init__(self, status, start_time):
        self.status = status
        self.start_time = start_time

    def __str__(self) -> str:
        return f"status: {self.status}, start_time: {time.asctime(time.localtime(self.start_time))}"


class ClientInfo:
    def __init__(self, name: str, last_connect_time):
        self.name = name
        self.last_connect_time = last_connect_time

    def __str__(self) -> str:
        return f"{self.name}(last_connect_time: {time.asctime(time.localtime(self.last_connect_time))})"


class JobInfo:
    def __init__(self, job_id: str, app_name: str):
        self.job_id = job_id
        self.app_name = app_name

    def __str__(self) -> str:
        return f"JobInfo:\n  job_id: {self.job_id}\n  app_name: {self.app_name}"


class SystemInfo:
    def __init__(self, server_info: ServerInfo, client_info: List[ClientInfo], job_info: List[JobInfo]):
        self.server_info = server_info
        self.client_info = client_info
        self.job_info = job_info

    def __str__(self) -> str:
        client_info_str = "\n".join(map(str, self.client_info))
        job_info_str = "\n".join(map(str, self.job_info))
        return (
            f"SystemInfo\nserver_info:\n{self.server_info}\nclient_info:\n{client_info_str}\njob_info:\n{job_info_str}"
        )


class SessionSpec(ABC):
    @abstractmethod
    def submit_job(self, job_definition_path: str) -> str:
        """Submit a predefined job to the NVFLARE system

        Args:
            job_definition_path: path to the folder that defines a NVFLARE job

        Returns: the job id if accepted by the system

        If the submission fails, exception will be raised:

        """
        pass

    @abstractmethod
    def clone_job(self, job_id: str) -> str:
        """Create a new job by cloning a specified job

        Args:
            job_id: job to be cloned

        Returns: ID of the new job

        """
        pass

    @abstractmethod
    def get_job_meta(self, job_id: str) -> dict:
        """Get the meta info of the specified job

        Args:
            job_id: ID of the job

        Returns: a dict of job metadata

        """
        pass

    @abstractmethod
    def list_jobs(self, detailed: bool = False, all: bool = False) -> List[dict]:
        """Get the job info from the server

        Args:
            detailed: True to get the detailed information for each job, False by default
            all: True to get jobs submitted by all users (default is to only list jobs submitted by the same user)

        Returns: a list of job metadata

        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def abort_job(self, job_id: str):
        """Abort the specified job

        Args:
            job_id: job to be aborted

        Returns: None

        If the job is already done, no effect;
        If job is not started yet, it will be cancelled and won't be scheduled.
        If the job is being executed, it will be aborted

        """
        pass

    @abstractmethod
    def delete_job(self, job_id: str):
        """Delete the specified job completely from the system

        Args:
            job_id: job to be deleted

        Returns: None

        If the job is being executed, the job will be stopped first.
        Everything of the job will be deleted from the job store, as well as workspaces on
        the FL server and clients.

        """
        pass

    @abstractmethod
    def get_system_info(self) -> SystemInfo:
        """Get general info of the FLARE system"""
        pass

    @abstractmethod
    def get_client_job_status(self, client_names: List[str] = None) -> List[dict]:
        """Get job status info of specified FL clients

        Args:
            client_names: names of the clients to get status info

        Returns: A list of jobs running on the clients. Each job is described by a dict of: id, app name and status.
        If there are multiple jobs running on one client, the list contains one entry for each job for that client.
        If no FL clients are connected or the server failed to communicate to them, this method returns None.

        """
        pass

    @abstractmethod
    def restart(self, target_type: str, client_names: Optional[List[str]] = None) -> dict:
        """
        Restart specified system target(s)

        Args:
            target_type: what system target (server, client, or all) to restart
            client_names: clients to be restarted if target_type is client. If not specified, all clients.

        Returns: a dict that contains detailed info about the restart request:
        status - the overall status of the result.
        server_status - whether the server is restarted successfully - only if target_type is "all" or "server".
        client_status - a dict (keyed on client name) that specifies status of each client - only if target_type
        is "all" or "client".

        """
        pass

    @abstractmethod
    def shutdown(self, target_type: TargetType, client_names: Optional[List[str]] = None):
        """Shut down specified system target(s)

        Args:
            target_type: what system target (server, client, or all) to shut down
            client_names: clients to be shut down if target_type is client. If not specified, all clients.

        Returns: None
        """
        pass

    @abstractmethod
    def set_timeout(self, value: float):
        """
        Set a session-specific command timeout. This is the amount of time the server will wait for responses
        after sending commands to FL clients.

        Note that this value is only effective for the current API session.

        Args:
            value: a positive float number

        Returns: None

        """
        pass

    @abstractmethod
    def unset_timeout(self):
        """
        Unset the session-specific command timeout. Once unset, the FL Admin Server's default will be used.

        Returns: None

        """
        pass

    @abstractmethod
    def list_sp(self) -> dict:
        """List available service providers

        Returns: a dict that contains information about the primary SP and others

        """
        pass

    @abstractmethod
    def get_active_sp(self) -> dict:
        """Get the current active service provider (SP).

        Returns: a dict that describes the current active SP. If no SP is available currently, the 'name' attribute of
        the result is empty.
        """
        pass

    @abstractmethod
    def promote_sp(self, sp_end_point: str):
        """Promote the specified endpoint to become the active SP.

        Args:
            sp_end_point: the endpoint of the SP. It's string in this format: <url>:<server_port>:<admin_port>

        Returns: None

        """
        pass

    @abstractmethod
    def get_available_apps_to_upload(self):
        """Get defined FLARE app folders from the upload folder on the machine the FLARE API is running

        Returns: a list of app folders

        """
        pass

    @abstractmethod
    def shutdown_system(self):
        """Shut down the whole NVFLARE system including the overseer, FL server(s), and all FL clients.

        Returns: None

        Note: the user must be a Project Admin to use this method; otherwise the NOT_AUTHORIZED exception will raise.

        """
        pass

    @abstractmethod
    def ls_target(self, target: str, options: str = None, path: str = None) -> str:
        """Run the "ls" command on the specified target and return result

        Args:
            target: the target (server or a client name) the command will run
            options: options of the "ls" command
            path: the optional file path

        Returns: result of "ls" command

        """
        pass

    @abstractmethod
    def cat_target(self, target: str, options: str = None, file: str = None) -> str:
        """Run the "cat" command on the specified target and return result

        Args:
            target: the target (server or a client name) the command will run
            options: options of the "cat" command
            file: the file that the "cat" command will run against

        Returns: result of "cat" command

        """
        pass

    @abstractmethod
    def tail_target(self, target: str, options: str = None, file: str = None) -> str:
        """Run the "tail" command on the specified target and return result

        Args:
            target: the target (server or a client name) the command will run
            options: options of the "tail" command
            file: the file that the "tail" command will run against

        Returns: result of "tail" command

        """
        pass

    @abstractmethod
    def tail_target_log(self, target: str, options: str = None) -> str:
        """Run the "tail log.txt" command on the specified target and return result

        Args:
            target: the target (server or a client name) the command will run
            options: options of the "tail" command

        Returns: result of "tail" command

        """
        pass

    @abstractmethod
    def head_target(self, target: str, options: str = None, file: str = None) -> str:
        """Run the "head" command on the specified target and return result

        Args:
            target: the target (server or a client name) the command will run
            options: options of the "head" command
            file: the file that the "head" command will run against

        Returns: result of "head" command

        """
        pass

    @abstractmethod
    def head_target_log(self, target: str, options: str = None) -> str:
        """Run the "head log.txt" command on the specified target and return result

        Args:
            target: the target (server or a client name) the command will run
            options: options of the "head" command

        Returns: result of "head" command

        """
        pass

    @abstractmethod
    def grep_target(self, target: str, options: str = None, pattern: str = None, file: str = None) -> str:
        """Run the "grep" command on the specified target and return result

        Args:
            target: the target (server or a client name) the command will run
            options: options of the "grep" command
            pattern: the grep pattern
            file: the file that the "grep" command will run against

        Returns: result of "grep" command

        """
        pass

    @abstractmethod
    def get_working_directory(self, target: str) -> str:
        """Get the working directory of the specified target

        Args:
            target: the target (server of a client name)

        Returns: current working directory of the specified target

        """
        pass

    @abstractmethod
    def show_stats(self, job_id: str, target_type: str, targets: Optional[List[str]] = None) -> dict:
        """Show processing stats of specified job on specified targets

        Args:
            job_id: ID of the job
            target_type: type of target (server or client)
            targets: list of client names if target type is "client". All clients if not specified.

        Returns: a dict that contains job stats on specified targets. The key of the dict is target name. The value is
        a dict of stats reported by different system components (ServerRunner or ClientRunner).

        """
        pass

    @abstractmethod
    def show_errors(self, job_id: str, target_type: str, targets: Optional[List[str]] = None) -> dict:
        """Show processing errors of specified job on specified targets

        Args:
            job_id: ID of the job
            target_type: type of target (server or client)
            targets: list of client names if target type is "client". All clients if not specified.

        Returns: a dict that contains job errors (if any) on specified targets. The key of the dict is target name.
        The value is a dict of errors reported by different system components (ServerRunner or ClientRunner).

        """
        pass

    @abstractmethod
    def reset_errors(self, job_id: str):
        """Clear errors for all system targets for the specified job

        Args:
            job_id: ID of the job

        Returns: None

        """
        pass

    @abstractmethod
    def get_connected_client_list(self) -> List[ClientInfo]:
        """Get the list of connected clients

        Returns: a list of ClientInfo objects

        """
        pass

    @abstractmethod
    def monitor_job(
        self, job_id: str, timeout: int = 0, poll_interval: float = 2.0, cb=None, *cb_args, **cb_kwargs
    ) -> MonitorReturnCode:
        """Monitor the job progress until one of the conditions occurs:
         - job is done
         - timeout
         - the status_cb returns False

        Args:
            job_id: the job to be monitored
            timeout: how long to monitor. If 0, never time out.
            poll_interval: how often to poll job status
            cb: if provided, callback to be called after each poll

        Returns: a MonitorReturnCode

        Every time the cb is called, it must return a bool indicating whether the monitor
        should continue. If False, this method ends.

        """
        pass

    @abstractmethod
    def close(self):
        """Close the session

        Returns:

        """
        pass


def job_monitor_cb_signature(session: SessionSpec, job_id: str, job_mea: dict, *args, **kwargs) -> bool:
    """

    Args:
        session: the session
        job_id: ID of the job being monitored
        job_mea: meta info of the job
        *args:
        **kwargs:

    Returns:

    """
    pass
