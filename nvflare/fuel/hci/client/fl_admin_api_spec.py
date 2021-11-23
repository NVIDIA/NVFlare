# Copyright (c) 2021, NVIDIA CORPORATION.
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
    """
    Structure containing the response of calls to the api as key value pairs. The status key is the primary indicator of
    the success of a call and can contain APIStatus.SUCCESS or another APIStatus. Most calls will return additional
    information in the details key, which is also a dictionary of key value pairs. The raw key can optionally have
    the underlying response from AdminAPI when relevant, particularly when data is received from the server and the
    status of a call is APIStatus.ERROR_RUNTIME to provide additional information.

    Note that the status in this response primarily indicates that the command submitted successfully. Depending on the
    command and especially for calls to multiple clients, the contents of details or the raw response should be examined
    to determine if the execution of the command was successful for each specific client.
    """

    def __init__(self, status: APIStatus, details: dict = None, raw: dict = None):
        super().__init__()
        self["status"] = status
        if details:
            self["details"] = details
        if raw:
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
        """
        Checks and returns the FL status. If target_type is server, the call does not wait for the server to retrieve
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
    def set_run_number(self, run_number: int) -> FLAdminAPIResponse:
        """
        Sets a current run number.

        Args:
            run_number: run number in order to set for the current experiment, must be integer greater than 0

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def delete_run_number(self, run_number: int) -> FLAdminAPIResponse:
        """
        Deletes a specified run number. This deletes the run folder corresponding to the run number on the server and
        all connected clients. This is not reversible.

        Args:
            run_number: run number for the run folder to delete

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def upload_app(self, app: str) -> FLAdminAPIResponse:
        """
        Uploads specified app to the upload directory of FL server. Currently assumes app is in the upload_dir set in
        API init.

        Args:
            app: name of the folder in upload_dir to upload

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def deploy_app(self, app: str, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """
        Issues a command to deploy the specified app to the specified target for the current run number. The app must
        be already uploaded and available on the server.

        Args:
            app: name of app to deploy
            target_type: server | client | all
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def start_app(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """
        Issue a command to start the deployed app for the current run number at the specified target.

        Args:
            target_type: server | client | all
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def abort(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """Issue a command to abort training.

        Args:
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def restart(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """Issue a command to restart the specified target. If the target is server, all FL clients will be restarted
        as well.

        Args:
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def shutdown(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        """Issue a command to stop FL entirely for a specific FL client or specific FL clients. Note that the targets
        will not be able to start with an API command after shutting down.

        Args:
            target_type: server | client
            targets: if target_type is client, targets can optionally be a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def remove_client(self, targets: List[str]) -> FLAdminAPIResponse:
        """Issue a command to remove a specific FL client or FL clients. Note that the targets
        will not be able to start with an API command after shutting down.

        Args:
            targets: a list of client names

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def set_timeout(self, timeout: float) -> FLAdminAPIResponse:
        """
        Sets the timeout for admin commands on the server.

        Args:
            timeout: timeout of admin commands to set on the server

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def get_available_apps_to_upload(self):
        pass

    @abstractmethod
    def ls_target(self, target: str, options: str = None, path: str = None) -> FLAdminAPIResponse:
        """
        Sends the shell command to get the directory listing of the target allowing for options that the ls command
        of admin client allows.

        Args:
            target:  either server or single client's client name.
            options: the options string as provided to the ls command for admin client.
            path:    optionally, the path to specify

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def cat_target(self, target: str, options: str = None, file: str = None) -> FLAdminAPIResponse:
        """
        Sends the shell command to get the contents of the target's specified file allowing for options that the cat
        command of admin client allows.

        Args:
            target:  either server or single client's client name.
            options: the options string as provided to the ls command for admin client.
            file:    the path to the file to return the contents of

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def tail_target_log(self, target: str, options: str = None) -> FLAdminAPIResponse:
        """Returns the end of target's log allowing for options that the tail of admin client allows.

        Args:
            target:  either server or single client's client name.
            options: the options string as provided to the tail command for admin client. For this command, "-n" can be
                     used to specify the number of lines for example "-n 100", or "-c" can specify the number of bytes.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def env_target(self, target: str) -> FLAdminAPIResponse:
        """Get the environment variables of the specified target.

        Args:
            target:  either server or single client's client name.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def get_working_directory(self, target: str) -> FLAdminAPIResponse:
        """Gets the workspace root directory of the specified target.

        Args:
            target:  either server or single client's client name.

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def grep_target(
        self, target: str, options: str = None, pattern: str = None, file: str = None
    ) -> FLAdminAPIResponse:
        """
        Sends the shell command to grep the contents of the target's specified file allowing for options that the grep
        command of admin client allows.

        Args:
            target:  either server or single client's client name.
            options: the options string as provided to the grep command for admin client.
            pattern: the pattern to search for
            file:    the path to the file to grep

        Returns: FLAdminAPIResponse

        """
        pass

    @abstractmethod
    def show_stats(self) -> FLAdminAPIResponse:
        """Gets and shows stats from the Info Collector.

        Returns: FLAdminAPIResponse

        """

    @abstractmethod
    def show_errors(self) -> FLAdminAPIResponse:
        """Gets and shows errors from the Info Collector.

        Returns: FLAdminAPIResponse

        """

    @abstractmethod
    def reset_errors(self) -> FLAdminAPIResponse:
        """Resets the collector errors.

        Returns: FLAdminAPIResponse

        """

    @abstractmethod
    def get_connected_client_list(self) -> FLAdminAPIResponse:
        """A convenience function to get a list of the clients currently connected to the FL server through the check
        status server call. Note that this returns the client list based on the last known statuses on the server, so it
        can be possible for a client to be disconnected and not yet removed from the list of connected clients.

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
        """Wait until provided callback returns True, with the option to specify a timeout and interval to
        check the server status. If no callback function is provided, the default callback returns True when the server
        status is "training stopped". A custom callback can be provided to add logic to handle checking for other
        conditions. A timeout should be set in case there are any error conditions that result in the system being stuck
        in a state where the callback never returns True.

        Args:
            interval: in seconds, the time between consecutive checks of the server
            timeout: if set, the amount of time this function will run until before returning a response message
            callback: the reply from check_status_server() will be passed to the callback, along with any additional kwargs
            which can go on to perform additional logic.
            fail_attempts: number of consecutive failed attempts of getting the server status before returning with ERROR_RUNTIME.

        Returns: FLAdminAPIResponse

        """
        pass
