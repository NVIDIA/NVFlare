# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Optional

from nvflare.fuel.hci.client.api_status import APIStatus


class FLAdminAPIResponse(dict):
    def __init__(self, status: APIStatus, details: dict = None, raw: dict = None):
        """Structure containing the response of calls to the api as key value pairs.

        The status key is the primary indicator of the success of a call and can contain APIStatus.SUCCESS or another
        APIStatus. Most calls will return additional information in the details key, which is also a dictionary of key
        value pairs. The raw key can optionally have the underlying response from AdminAPI when relevant, particularly
        when data is received from the server and the status of a call is APIStatus.ERROR_RUNTIME to provide additional
        information.

        Note that the status in this response primarily indicates that the command submitted successfully. Depending on
        the command and especially for calls to multiple clients, the contents of details or the raw response should be
        examined to determine if the execution of the command was successful for each specific client.

        Args:
            status: APIStatus for primary indicator of the success of a call
            details: response details
            raw: raw response from server
        """
        super().__init__()
        self["status"] = status  # todo: status.value but it may break existing code
        if details is not None:
            self["details"] = details
        if raw is not None:
            self["raw"] = raw


class APISyntaxError(Exception):
    pass


class TargetType(str, Enum):
    ALL = "all"
    SERVER = "server"
    CLIENT = "client"


class FLAdminAPISpec(ABC):
    @abstractmethod
    def check_status(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """Checks and returns the FL status.

        If target_type is server, the call does not wait for the server to retrieve
        information on the clients but returns the last information the server had at the time this call is made.

        If target_type is client, specific clients can be specified in targets, and this call generally takes longer
        than the function to just check the FL server status because this one waits for communication from the server to
        client then back.

        Note that this is still the previous training check_status, and there will be a new call to get status through
        InfoCollector, which will be able to get information from components.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def submit_job(self, job_folder: str) -> FLAdminAPIResponse:
        """Submit a job.

        Assumes job folder is in the upload_dir set in API init.

        Args:
            job_folder (str): name of the job folder in upload_dir to submit

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def clone_job(self, job_id: str) -> FLAdminAPIResponse:
        """Clone a job that exists by copying the job contents and providing a new job_id.

        Args:
            job_id (str): job id of the job to clone

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def list_jobs(self, options: str = None) -> FLAdminAPIResponse:
        """List the jobs in the system.

        Args:
            options (str): the options string as provided to the list_jobs command for admin client.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def download_job(self, job_id: str) -> FLAdminAPIResponse:
        """Download the specified job in the system.

        Args:
            job_id (str): Job id for the job to download

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def abort_job(self, job_id: str) -> FLAdminAPIResponse:
        """Abort a job that is running.

        Args:
            job_id (str): the job id to abort

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def delete_job(self, job_id: str) -> FLAdminAPIResponse:
        """Delete the specified job and workspace from the permanent store.

        Args:
            job_id (str): the job id to delete

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def abort(self, job_id: str, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """Issue a command to abort training.

        Args:
            job_id (str): job id
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def restart(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """Issue a command to restart the specified target.

        If the target is server, all FL clients will be restarted as well.

        Args:
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def shutdown(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """Issue a command to stop FL entirely for a specific FL client or specific FL clients.

        Note that the targets will not be able to start with an API command after shutting down.

        Args:
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def remove_client(self, targets: List[str]) -> FLAdminAPIResponse:
        """Issue a command to remove a specific FL client or FL clients.

        Note that the targets will not be able to start with an API command after shutting down. Also, you will not be
        able to issue admin commands through the server to that client until the client is restarted (this includes
        being able to issue the restart command through the API).

        Args:
            targets: a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def set_timeout(self, timeout: float) -> FLAdminAPIResponse:
        """Sets the timeout for admin commands on the server in seconds.

        This timeout is the maximum amount of time the server will wait for replies from clients. If the timeout is too
        short, the server may not receive a response because clients may not have a chance to reply.

        Args:
            timeout: timeout in seconds of admin commands to set on the server

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def list_sp(self) -> FLAdminAPIResponse:
        """Gets the information on the available servers (service providers).

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def get_active_sp(self) -> FLAdminAPIResponse:
        """Gets the active server (service provider).

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def promote_sp(self, sp_end_point: str) -> FLAdminAPIResponse:
        """Sends command through overseer_agent to promote the specified sp_end_point to become the active server.

        Args:
            sp_end_point: service provider end point to promote to active in the form of server:fl_port:admin_port like example.com:8002:8003

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def get_available_apps_to_upload(self):
        pass

    @abstractmethod
    def ls_target(self, target: str, options: str = None, path: str = None) -> FLAdminAPIResponse:
        """Issue ls command to retrieve the contents of the path.

        Sends the shell command to get the directory listing of the target allowing for options that the ls command
        of admin client allows. If no path is specified, the contents of the working directory are returned. The target
        can be "server" or a specific client name for example "site2". The allowed options are: "-a" for all, "-l" to
        use a long listing format, "-t" to sort by modification time newest first, "-S" to sort by file size largest
        first, "-R" to list subdirectories recursively, "-u" with -l to show access time otherwise sort by access time.

        Args:
            target (str):  either server or single client's client name.
            options (str): the options string as provided to the ls command for admin client.
            path (str):    optionally, the path to specify (relative to the working directory of the specified target)

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def cat_target(self, target: str, options: str = None, file: str = None) -> FLAdminAPIResponse:
        """Issue cat command.

        Sends the shell command to get the contents of the target's specified file allowing for options that the cat
        command of admin client allows. The target can be "server" or a specific client name for example "site2". The
        file is required and should contain the relative path to the file from the working directory of the target. The
        allowed options are "-n" to number all output lines, "-b" to number nonempty output lines, "-s" to suppress
        repeated empty output lines, and "-T" to display TAB characters as ^I.

        Args:
            target (str):  either server or single client's client name.
            options (str): the options string as provided to the ls command for admin client.
            file (str):    the path to the file to return the contents of

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def tail_target_log(self, target: str, options: str = None) -> FLAdminAPIResponse:
        """Returns the end of target's log allowing for options that the tail of admin client allows.

        The option "-n" can be used to specify the number of lines for example "-n 100", or "-c" can specify the
        number of bytes.

        Args:
            target (str):  either server or single client's client name.
            options (str): the options string as provided to the tail command for admin client. For this command, "-n" can be
                     used to specify the number of lines for example "-n 100", or "-c" can specify the number of bytes.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def get_working_directory(self, target: str) -> FLAdminAPIResponse:
        """Gets the workspace root directory of the specified target.

        Args:
            target (str):  either server or single client's client name.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def grep_target(
        self, target: str, options: str = None, pattern: str = None, file: str = None
    ) -> FLAdminAPIResponse:
        """Issue grep command.

        Sends the shell command to grep the contents of the target's specified file allowing for options that the grep
        command of admin client allows. The target can be "server" or a specific client name for example "site2". The
        file is required and should contain the relative path to the file from the working directory of the target. The
        pattern is also required. The allowed options are "-n" to print line number with output lines, "-i" to ignore
        case distinctions, and "-b" to print the byte offset with output lines.

        Args:
            target (str):  either server or single client's client name.
            options (str): the options string as provided to the grep command for admin client.
            pattern (str): the pattern to search for
            file (str):    the path to the file to grep

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def show_stats(
        self, job_id: str, target_type: TargetType, targets: Optional[List[str]] = None
    ) -> FLAdminAPIResponse:
        """Gets and shows stats from the Info Collector.

        Args:
            job_id (str): job id
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """

    @abstractmethod
    def show_errors(
        self, job_id: str, target_type: TargetType, targets: Optional[List[str]] = None
    ) -> FLAdminAPIResponse:
        """Gets and shows errors from the Info Collector.

        Args:
            job_id (str): job id
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """

    @abstractmethod
    def reset_errors(self, job_id: str) -> FLAdminAPIResponse:
        """Resets the collector errors.

        Args:
            job_id (str): job id

        Returns: FLAdminAPIResponse

        """

    @abstractmethod
    def get_connected_client_list(self) -> FLAdminAPIResponse:
        """A convenience function to get a list of the clients currently connected to the FL server.

        Operates through the check status server call. Note that this returns the client list based on the last known
        statuses on the server, so it can be possible for a client to be disconnected and not yet removed from the list
        of connected clients.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def wait_until_server_status(
        self,
        interval: int = 20,
        timeout: int = None,
        callback: Callable[[FLAdminAPIResponse], bool] = None,
        fail_attempts: int = 3,
    ) -> FLAdminAPIResponse:
        """Wait until provided callback returns True.

        There is the option to specify a timeout and interval to check the server status. If no callback function is
        provided, the default callback returns True when the server
        status is "training stopped". A custom callback can be provided to add logic to handle checking for other
        conditions. A timeout should be set in case there are any error conditions that result in the system being stuck
        in a state where the callback never returns True.

        Args:
            interval (int): in seconds, the time between consecutive checks of the server
            timeout (int): if set, the amount of time this function will run until before returning a response message
            callback: the reply from check_status_server() will be passed to the callback, along with any additional kwargs
            which can go on to perform additional logic.
            fail_attempts (int): number of consecutive failed attempts of getting the server status before returning with ERROR_RUNTIME.

        Returns: FLAdminAPIResponse

        """
        pass
