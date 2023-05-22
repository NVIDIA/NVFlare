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
from typing import List


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


class JobNotDone(Exception):
    pass


class InternalError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


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

        Returns: a dict of job meta data

        """
        pass

    @abstractmethod
    def list_jobs(self, detailed: bool = False, all: bool = False) -> List[dict]:
        """Get the job info from the server

        Args:
            detailed: True to get the detailed information for each job, False by default
            all: True to get jobs submitted by all users (default is to only list jobs submitted by the same user)

        Returns: a list of of job meta data

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
