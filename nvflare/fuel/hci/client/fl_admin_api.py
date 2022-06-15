# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.apis.overseer_spec import OverseerAgent
from nvflare.apis.utils.format_check import type_pattern_mapping
from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import APISyntaxError, FLAdminAPIResponse, FLAdminAPISpec, TargetType


def wrap_with_return_exception_responses(func):
    """Decorator on all FLAdminAPI calls to handle any raised exceptions and return the fitting error status."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            reply = func(self, *args, **kwargs)
            if reply:
                return reply
            else:
                return FLAdminAPIResponse(
                    APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not generate reply."}
                )
        except ConnectionRefusedError as e:
            return FLAdminAPIResponse(APIStatus.ERROR_AUTHENTICATION, {"message": "Error: " + str(e)})
        except PermissionError as e:
            return FLAdminAPIResponse(APIStatus.ERROR_AUTHORIZATION, {"message": "Error: " + str(e)})
        except LookupError as e:
            return FLAdminAPIResponse(APIStatus.ERROR_INVALID_CLIENT, {"message": "Error: " + str(e)})
        except APISyntaxError as e:
            return FLAdminAPIResponse(APIStatus.ERROR_SYNTAX, {"message": "Error: " + str(e)})
        except TimeoutError as e:
            return FLAdminAPIResponse(
                APIStatus.ERROR_RUNTIME,
                {"message": "TimeoutError: possibly unable to communicate with server. " + str(e)},
            )
        except Exception as e:
            return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": "Exception: " + str(e)})

    return wrapper


def default_server_status_handling_cb(reply: FLAdminAPIResponse, **kwargs) -> bool:
    if reply["details"][FLDetailKey.SERVER_ENGINE_STATUS] == "stopped":
        return True
    else:
        return False


def default_client_status_handling_cb(reply: FLAdminAPIResponse) -> bool:
    client_statuses = reply.get("details").get("client_statuses")
    stopped_client_count = 0
    for i in range(1, len(client_statuses)):
        if client_statuses[i][3] == "No Jobs":
            stopped_client_count = stopped_client_count + 1
    if stopped_client_count == len(client_statuses) - 1:
        return True
    else:
        return False


def default_stats_handling_cb(reply: FLAdminAPIResponse) -> bool:
    if reply.get("details").get("message").get("ServerRunner").get("status") == "done":
        return True
    else:
        return False


class FLAdminAPI(AdminAPI, FLAdminAPISpec):
    def __init__(
        self,
        ca_cert: str = "",
        client_cert: str = "",
        client_key: str = "",
        upload_dir: str = "",
        download_dir: str = "",
        server_cn=None,
        cmd_modules: Optional[List] = None,
        overseer_agent: OverseerAgent = None,
        user_name: str = None,
        poc=False,
        debug=False,
    ):
        """FLAdminAPI serves as foundation for communications to FL server through the AdminAPI.

        Upon initialization, FLAdminAPI will start the overseer agent to get the active server and then try to log in.
        This happens in a thread, so code that executes after should check that the FLAdminAPI is successfully logged in.

        Args:
            ca_cert: path to CA Cert file, by default provisioned rootCA.pem
            client_cert: path to admin client Cert file, by default provisioned as client.crt
            client_key: path to admin client Key file, by default provisioned as client.key
            upload_dir: File transfer upload directory. Folders uploaded to the server to be deployed must be here. Folder must already exist and be accessible.
            download_dir: File transfer download directory. Can be same as upload_dir. Folder must already exist and be accessible.
            server_cn: server cn (only used for validating server cn)
            cmd_modules: command modules to load and register. Note that FileTransferModule is initialized here with upload_dir and download_dir if cmd_modules is None.
            overseer_agent: initialized OverseerAgent to obtain the primary service provider to set the host and port of the active server
            user_name: Username to authenticate with FL server
            poc: Whether to enable poc mode for using the proof of concept example without secure communication.
            debug: Whether to print debug messages. False by default.
        """
        super().__init__(
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            upload_dir=upload_dir,
            download_dir=download_dir,
            server_cn=server_cn,
            cmd_modules=cmd_modules,
            overseer_agent=overseer_agent,
            auto_login=True,
            user_name=user_name,
            poc=poc,
            debug=debug,
        )
        self.upload_dir = upload_dir
        self.download_dir = download_dir
        self._error_buffer = None

    def _process_targets_into_str(self, targets: List[str]) -> str:
        if not isinstance(targets, list):
            raise APISyntaxError("targets is not a list.")
        if not all(isinstance(t, str) for t in targets):
            raise APISyntaxError("all targets in the list of targets must be strings.")
        for t in targets:
            try:
                self._validate_required_target_string(t)
            except APISyntaxError:
                raise APISyntaxError("each target in targets must be a string of only valid characters and no spaces.")
        return " ".join(targets)

    def _validate_required_target_string(self, target: str) -> str:
        """Returns the target string if it exists and is valid."""
        if not target:
            raise APISyntaxError("target is required but not specified.")
        if not isinstance(target, str):
            raise APISyntaxError("target is not str.")
        if not re.match("^[A-Za-z0-9._-]*$", target):
            raise APISyntaxError("target must be a string of only valid characters and no spaces.")
        return target

    def _validate_options_string(self, options: str) -> str:
        """Returns the options string if it is valid."""
        if not isinstance(options, str):
            raise APISyntaxError("options is not str.")
        if not re.match("^[A-Za-z0-9- ]*$", options):
            raise APISyntaxError("options must be a string of only valid characters.")
        return options

    def _validate_path_string(self, path: str) -> str:
        """Returns the path string if it is valid."""
        if not isinstance(path, str):
            raise APISyntaxError("path is not str.")
        if not re.match("^[A-Za-z0-9-._/]*$", path):
            raise APISyntaxError("unsupported characters in path {}".format(path))
        if path.startswith("/"):
            raise APISyntaxError("absolute path is not allowed")
        paths = path.split("/")
        for p in paths:
            if p == "..":
                raise APISyntaxError(".. in path name is not allowed")
        return path

    def _validate_file_string(self, file: str) -> str:
        """Returns the file string if it is valid."""
        if not isinstance(file, str):
            raise APISyntaxError("file is not str.")
        if not re.match("^[A-Za-z0-9-._/]*$", file):
            raise APISyntaxError("unsupported characters in file {}".format(file))
        if file.startswith("/"):
            raise APISyntaxError("absolute path for file is not allowed")
        paths = file.split("/")
        for p in paths:
            if p == "..":
                raise APISyntaxError(".. in file path is not allowed")
        basename, file_extension = os.path.splitext(file)
        if file_extension not in [".txt", ".log", ".json", ".csv", ".sh", ".config", ".py"]:
            raise APISyntaxError(
                "this command cannot be applied to file {}. Only files with the following extensions are "
                "permitted: .txt, .log, .json, .csv, .sh, .config, .py".format(file)
            )
        return file

    def _validate_sp_string(self, sp_string) -> str:
        if re.match(
            type_pattern_mapping.get("sp_end_point"),
            sp_string,
        ):
            return sp_string
        else:
            raise APISyntaxError("sp_string must be of the format example.com:8002:8003")

    def _get_processed_cmd_reply_data(self, command) -> Tuple[bool, str, Dict[str, Any]]:
        """Executes the specified command through the underlying AdminAPI's do_command() and checks the response to
        raise common errors.

        Returns:
            Tuple of bool to indicate if success is in reply data, str with full response of the reply data, and the raw
            reply.
        """
        success_in_data = False
        reply = self.do_command(command)
        # handle errors from write_error (these can be from FileTransferModule)
        if self._error_buffer:
            err = self._error_buffer
            self._error_buffer = None
            raise RuntimeError(err)
        if reply.get("status") == APIStatus.SUCCESS:
            success_in_data = True
        reply_data_list = []
        reply_data_full_response = ""
        if reply.get("data"):
            for data in reply["data"]:
                if isinstance(data, dict):
                    if data.get("type") == "success":
                        success_in_data = True
                    if data.get("type") == "string" or data.get("type") == "error":
                        reply_data_list.append(data["data"])
            reply_data_full_response = "\n".join(reply_data_list)
            if "session_inactive" in reply_data_full_response:
                raise ConnectionRefusedError(reply_data_full_response)
            if "Failed to communicate" in reply_data_full_response:
                raise ConnectionError(reply_data_full_response)
            if "invalid client" in reply_data_full_response:
                raise LookupError(reply_data_full_response)
            if "unknown site" in reply_data_full_response:
                raise LookupError(reply_data_full_response)
            if "Authorization Error" in reply_data_full_response:
                raise PermissionError(reply_data_full_response)
        if reply.get("status") != APIStatus.SUCCESS:
            raise RuntimeError(reply.get("details"))
        return success_in_data, reply_data_full_response, reply

    def _parse_section_of_response_text(
        self, data, start_string: str, offset: int = None, end_string: str = None, end_index=None
    ) -> str:
        """Convenience method to get portion of string based on parameters."""
        if not offset:
            offset = len(start_string) + 1
        if end_string:
            return data[data.find(start_string) + offset : data.find(end_string)]
        if end_index:
            return data[data.find(start_string) + offset : end_index]
        return data[data.find(start_string) + offset :]

    def _parse_section_of_response_text_as_int(
        self, data, start_string: str, offset: int = None, end_string: str = None, end_index=None
    ) -> int:
        try:
            return int(
                self._parse_section_of_response_text(
                    data=data, start_string=start_string, offset=offset, end_string=end_string, end_index=end_index
                )
            )
        except ValueError:
            return -1

    def write_error(self, error: str) -> None:
        """Internally used to handle errors from FileTransferModule"""
        self._error_buffer = error

    @wrap_with_return_exception_responses
    def check_status(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        if target_type == TargetType.SERVER:
            return self._check_status_server()
        elif target_type == TargetType.CLIENT:
            return self._check_status_client(targets)
        else:
            raise APISyntaxError("target_type must be server or client.")

    def _check_status_server(self) -> FLAdminAPIResponse:
        """
        Checks the server status and returns the details. This call does not wait for the server to retrieve information
        on the clients but returns the last information the server had at the time this call is made.

        """
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(
            AdminCommandNames.CHECK_STATUS + " server"
        )
        details = {}
        if reply.get("data"):
            for data in reply["data"]:
                if data["type"] == "string":
                    if data["data"].find("Engine status:") != -1:
                        details[FLDetailKey.SERVER_ENGINE_STATUS] = self._parse_section_of_response_text(
                            data=data["data"], start_string="Engine status:"
                        )
                    if data["data"].find("Registered clients:") != -1:
                        details[FLDetailKey.REGISTERED_CLIENTS] = self._parse_section_of_response_text_as_int(
                            data=data["data"], start_string="Registered clients:"
                        )
                if data["type"] == "table":
                    details[FLDetailKey.STATUS_TABLE] = data["rows"]
            return FLAdminAPIResponse(APIStatus.SUCCESS, details, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    def _check_status_client(self, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        if targets:
            processed_targets_str = self._process_targets_into_str(targets)
            command = AdminCommandNames.CHECK_STATUS + " client " + processed_targets_str
        else:
            command = AdminCommandNames.CHECK_STATUS + " client"
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        details = {}
        if reply.get("data"):
            for data in reply["data"]:
                if data["type"] == "table":
                    details["client_statuses"] = data["rows"]
            return FLAdminAPIResponse(APIStatus.SUCCESS, details, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def submit_job(self, job_folder: str) -> FLAdminAPIResponse:
        if not job_folder:
            raise APISyntaxError("job_folder is required but not specified.")
        if not isinstance(job_folder, str):
            raise APISyntaxError("job_folder must be str but got {}.".format(type(job_folder)))
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(
            AdminCommandNames.SUBMIT_JOB + " " + job_folder
        )
        if reply_data_full_response:
            if "Submitted job" in reply_data_full_response:
                # TODO:: this is a hack to get job id
                return FLAdminAPIResponse(
                    APIStatus.SUCCESS,
                    {"message": reply_data_full_response, "job_id": reply_data_full_response.split(":")[-1].strip()},
                    reply,
                )
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def clone_job(self, job_id: str) -> FLAdminAPIResponse:
        if not job_id:
            raise APISyntaxError("job_folder is required but not specified.")
        if not isinstance(job_id, str):
            raise APISyntaxError("job_folder must be str but got {}.".format(type(job_id)))
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(
            AdminCommandNames.CLONE_JOB + " " + job_id
        )
        if reply_data_full_response:
            if "Cloned job" in reply_data_full_response:
                return FLAdminAPIResponse(
                    APIStatus.SUCCESS,
                    {"message": reply_data_full_response, "job_id": reply_data_full_response.split(":")[-1].strip()},
                    reply,
                )
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def list_jobs(self, options: str = None) -> FLAdminAPIResponse:
        command = AdminCommandNames.LIST_JOBS
        if options:
            options = self._validate_options_string(options)
            command = command + " " + options
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def download_job(self, job_id: str) -> FLAdminAPIResponse:
        if not job_id:
            raise APISyntaxError("job_id is required but not specified.")
        if not isinstance(job_id, str):
            raise APISyntaxError("job_id must be str but got {}.".format(type(job_id)))
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(
            AdminCommandNames.DOWNLOAD_JOB + " " + job_id
        )
        if success:
            return FLAdminAPIResponse(
                APIStatus.SUCCESS,
                {"message": reply.get("details")},
                reply,
            )
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def abort_job(self, job_id: str) -> FLAdminAPIResponse:
        if not job_id:
            raise APISyntaxError("job_id is required but not specified.")
        if not isinstance(job_id, str):
            raise APISyntaxError("job_id must be str but got {}.".format(type(job_id)))
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(
            AdminCommandNames.ABORT_JOB + " " + job_id
        )
        if reply_data_full_response:
            if "Abort signal has been sent" in reply_data_full_response:
                return FLAdminAPIResponse(
                    APIStatus.SUCCESS,
                    {"message": reply_data_full_response},
                    reply,
                )
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def delete_job(self, job_id: str) -> FLAdminAPIResponse:
        if not isinstance(job_id, str):
            raise APISyntaxError("job_id must be str but got {}.".format(type(job_id)))
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(
            AdminCommandNames.DELETE_JOB + " " + str(job_id)
        )
        if reply_data_full_response:
            if "can not be deleted" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response})
        if success:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def abort(self, job_id: str, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        if not job_id:
            raise APISyntaxError("job_id is required but not specified.")
        if not isinstance(job_id, str):
            raise APISyntaxError("job_id must be str but got {}.".format(type(job_id)))
        if target_type == TargetType.ALL:
            command = AdminCommandNames.ABORT + " " + job_id + " all"
        elif target_type == TargetType.SERVER:
            command = AdminCommandNames.ABORT + " " + job_id + " server"
        elif target_type == TargetType.CLIENT:
            if targets:
                processed_targets_str = self._process_targets_into_str(targets)
                command = AdminCommandNames.ABORT + " " + job_id + " client " + processed_targets_str
            else:
                command = AdminCommandNames.ABORT + " " + job_id + " client"
        else:
            raise APISyntaxError("target_type must be server, client, or all.")
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            if "Server app has not started" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response}, reply)
            if "No clients to abort" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response}, reply)
            if "please wait for started before abort" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response}, reply)
        if success:
            return_details = {}
            if reply_data_full_response:
                return_details["message"] = reply_data_full_response
            if reply.get("data"):
                for data in reply["data"]:
                    if data["type"] == "table":
                        return_details[FLDetailKey.RESPONSES] = data["rows"]
            return FLAdminAPIResponse(APIStatus.SUCCESS, return_details, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def restart(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        if target_type == TargetType.ALL:
            command = AdminCommandNames.RESTART + " " + "all"
        elif target_type == TargetType.SERVER:
            command = AdminCommandNames.RESTART + " " + "server"
        elif target_type == TargetType.CLIENT:
            if targets:
                processed_targets_str = self._process_targets_into_str(targets)
                command = AdminCommandNames.RESTART + " client " + processed_targets_str
            else:
                command = AdminCommandNames.RESTART + " " + "client"
        else:
            raise APISyntaxError("target_type must be server, client, or all.")
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            if "no clients available" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response})
            if "Server is starting, please wait for started before restart" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response})
        if success:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def shutdown(self, target_type: TargetType, targets: Optional[List[str]] = None) -> FLAdminAPIResponse:
        if target_type == TargetType.ALL:
            command = AdminCommandNames.SHUTDOWN + " " + "all"
        elif target_type == TargetType.SERVER:
            command = AdminCommandNames.SHUTDOWN + " " + "server"
        elif target_type == TargetType.CLIENT:
            if targets:
                processed_targets_str = self._process_targets_into_str(targets)
                command = AdminCommandNames.SHUTDOWN + " client " + processed_targets_str
            else:
                command = AdminCommandNames.SHUTDOWN + " " + "client"
        else:
            raise APISyntaxError("target_type must be server, client, or all.")
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            if "There are still active clients. Shutdown all clients first." in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response})
            if "no clients to shutdown" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response})
            if "Server is starting, please wait for started before shutdown" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response})
        if success:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def remove_client(self, targets: List[str]) -> FLAdminAPIResponse:
        if not targets:
            raise APISyntaxError("targets needs to be provided as a list of client names.")
        processed_targets_str = self._process_targets_into_str(targets)
        command = AdminCommandNames.REMOVE_CLIENT + " " + processed_targets_str
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if success:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def set_timeout(self, timeout: float) -> FLAdminAPIResponse:
        if not isinstance(timeout, (float, int)):
            raise APISyntaxError("timeout is not float.")
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data("set_timeout " + str(timeout))
        if success:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def list_sp(self) -> FLAdminAPIResponse:
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data("list_sp")
        if reply.get("data"):
            return FLAdminAPIResponse(APIStatus.SUCCESS, reply.get("data"), reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def get_active_sp(self) -> FLAdminAPIResponse:
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data("get_active_sp")
        if reply.get("details"):
            return FLAdminAPIResponse(APIStatus.SUCCESS, reply.get("details"), reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def promote_sp(self, sp_end_point: str) -> FLAdminAPIResponse:
        sp_end_point = self._validate_sp_string(sp_end_point)
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data("promote_sp " + sp_end_point)
        if success:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply.get("details")}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def shutdown_system(self) -> FLAdminAPIResponse:
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data("shutdown_system")
        if success:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply.get("details")}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def get_available_apps_to_upload(self):
        dir_list = []
        for item in os.listdir(self.upload_dir):
            if os.path.isdir(os.path.join(self.upload_dir, item)):
                dir_list.append(item)
        return FLAdminAPIResponse(APIStatus.SUCCESS, {"app_list": dir_list})

    @wrap_with_return_exception_responses
    def ls_target(self, target: str, options: str = None, path: str = None) -> FLAdminAPIResponse:
        target = self._validate_required_target_string(target)
        command = "ls " + target
        if options:
            options = self._validate_options_string(options)
            command = command + " " + options
        if path:
            path = self._validate_path_string(path)
            command = command + " " + path
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response})
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def cat_target(self, target: str, options: str = None, file: str = None) -> FLAdminAPIResponse:
        if not file:
            raise APISyntaxError("file is required but not specified.")
        file = self._validate_file_string(file)
        target = self._validate_required_target_string(target)
        command = "cat " + target
        if options:
            options = self._validate_options_string(options)
            command = command + " " + options
        if file:
            command = command + " " + file
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response})
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def tail_target_log(self, target: str, options: str = None) -> FLAdminAPIResponse:
        target = self._validate_required_target_string(target)
        command = "tail " + target
        if options:
            options = self._validate_options_string(options)
            command = command + " " + options
        command = command + " log.txt"
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response})
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def env_target(self, target: str) -> FLAdminAPIResponse:
        target = self._validate_required_target_string(target)
        command = "env " + target
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            details = {}
            environment = reply_data_full_response.split("\n")
            for e in environment:
                # set key and value to contents of each line with first = as separator
                details[e[0 : e.find("=")]] = e[e.find("=") + 1 :]
            return FLAdminAPIResponse(APIStatus.SUCCESS, details)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def get_working_directory(self, target: str) -> FLAdminAPIResponse:
        target = self._validate_required_target_string(target)
        command = "pwd " + target
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response})
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def grep_target(
        self, target: str, options: str = None, pattern: str = None, file: str = None
    ) -> FLAdminAPIResponse:
        if not file:
            raise APISyntaxError("file is required but not specified.")
        file = self._validate_file_string(file)
        if not pattern:
            raise APISyntaxError("pattern is required but not specified.")
        if not isinstance(pattern, str):
            raise APISyntaxError("pattern is not str.")
        target = self._validate_required_target_string(target)
        command = "grep " + target
        if options:
            options = self._validate_options_string(options)
            command = command + " " + options
        command = command + ' "' + pattern + '" ' + file
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply_data_full_response:
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response})
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def show_stats(
        self, job_id: str, target_type: TargetType, targets: Optional[List[str]] = None
    ) -> FLAdminAPIResponse:
        if not job_id:
            raise APISyntaxError("job_id is required but not specified.")
        if not isinstance(job_id, str):
            raise APISyntaxError("job_id must be str but got {}.".format(type(job_id)))
        if target_type == TargetType.SERVER:
            command = AdminCommandNames.SHOW_STATS + " " + job_id + " server"
        elif target_type == TargetType.CLIENT:
            if targets:
                processed_targets_str = self._process_targets_into_str(targets)
                command = AdminCommandNames.SHOW_STATS + " " + job_id + " client " + processed_targets_str
            else:
                command = AdminCommandNames.SHOW_STATS + " " + job_id + " client"
        else:
            raise APISyntaxError("target_type must be server or client.")
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply.get("data"):
            for data in reply["data"]:
                if data["type"] == "dict":
                    stats_result = data["data"]
                    return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": stats_result}, reply)
        if reply_data_full_response:
            if "App is not running" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def show_errors(
        self, job_id: str, target_type: TargetType, targets: Optional[List[str]] = None
    ) -> FLAdminAPIResponse:
        if not job_id:
            raise APISyntaxError("job_id is required but not specified.")
        if not isinstance(job_id, str):
            raise APISyntaxError("job_id must be str but got {}.".format(type(job_id)))
        if target_type == TargetType.SERVER:
            command = AdminCommandNames.SHOW_ERRORS + " " + job_id + " server"
        elif target_type == TargetType.CLIENT:
            if targets:
                processed_targets_str = self._process_targets_into_str(targets)
                command = AdminCommandNames.SHOW_ERRORS + " " + job_id + " client " + processed_targets_str
            else:
                command = AdminCommandNames.SHOW_ERRORS + " " + job_id + " client"
        else:
            raise APISyntaxError("target_type must be server or client.")
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(command)
        if reply.get("data"):
            for data in reply["data"]:
                if data["type"] == "dict":
                    errors_result = data["data"]
                    return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": errors_result}, reply)
        if reply_data_full_response:
            if "App is not running" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": "No errors."}, reply)

    @wrap_with_return_exception_responses
    def reset_errors(self, job_id: str) -> FLAdminAPIResponse:
        if not job_id:
            raise APISyntaxError("job_id is required but not specified.")
        if not isinstance(job_id, str):
            raise APISyntaxError("job_id must be str but got {}.".format(type(job_id)))
        success, reply_data_full_response, reply = self._get_processed_cmd_reply_data(
            AdminCommandNames.RESET_ERRORS + " " + job_id
        )
        if reply_data_full_response:
            if "App is not running" in reply_data_full_response:
                return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": reply_data_full_response}, reply)
            return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": reply_data_full_response}, reply)
        return FLAdminAPIResponse(
            APIStatus.ERROR_RUNTIME, {"message": "Runtime error: could not handle server reply."}, reply
        )

    @wrap_with_return_exception_responses
    def get_connected_client_list(self) -> FLAdminAPIResponse:
        reply = self._check_status_server()
        if reply["status"] == APIStatus.SUCCESS:
            status_table = reply["details"][FLDetailKey.STATUS_TABLE]
            list_of_connected_clients = []
            for row in status_table:
                if row[0] != "CLIENT NAME":
                    list_of_connected_clients.append(row[0])
            return FLAdminAPIResponse(APIStatus.SUCCESS, {FLDetailKey.CONNECTED_CLIENTS: list_of_connected_clients})
        else:
            return FLAdminAPIResponse(APIStatus.ERROR_RUNTIME, {"message": "runtime error"}, reply)

    @wrap_with_return_exception_responses
    def wait_until_server_status(
        self,
        interval: int = 20,
        timeout: int = None,
        callback: Callable[[FLAdminAPIResponse, Optional[List]], bool] = default_server_status_handling_cb,
        fail_attempts: int = 3,
        **kwargs,
    ) -> FLAdminAPIResponse:
        failed_attempts = 0
        start = time.time()
        while True:
            reply = self._check_status_server()
            if reply["details"].get(FLDetailKey.SERVER_ENGINE_STATUS):
                met = callback(reply, **kwargs)
                if met:
                    return FLAdminAPIResponse(APIStatus.SUCCESS, {}, None)
                fail_attempts = 0
            else:
                print("Could not get reply from check status server, trying again later")
                failed_attempts += 1

            now = time.time()
            if timeout is not None:
                if now - start >= timeout:
                    return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": "Waited until timeout."}, None)
            if failed_attempts > fail_attempts:
                return FLAdminAPIResponse(
                    APIStatus.ERROR_RUNTIME,
                    {
                        "message": "FL server status was not obtainable for more than the specified number of "
                        "fail_attempts. "
                    },
                    None,
                )
            time.sleep(interval)

    @wrap_with_return_exception_responses
    def wait_until_client_status(
        self,
        interval: int = 10,
        timeout: int = None,
        callback: Callable[[FLAdminAPIResponse, Optional[List]], bool] = default_client_status_handling_cb,
        fail_attempts: int = 6,
        **kwargs,
    ) -> FLAdminAPIResponse:
        """This is similar to wait_until_server_status() and is an example for using other information from a repeated
        call, in this case check_status(TargetType.CLIENT). Custom code can be written to use any data available from
        any call to make decisions for how to proceed. Take caution that the conditions will be met at some point, or
        timeout should be set with logic outside this function to handle checks for potential errors or this may loop
        indefinitely.

        Args:
            interval: in seconds, the time between consecutive checks of the server
            timeout: if set, the amount of time this function will run until before returning a response message
            callback: the reply from show_stats(TargetType.SERVER) will be passed to the callback, along with any additional kwargs
            which can go on to perform additional logic.
            fail_attempts: number of consecutive failed attempts of getting the server status before returning with ERROR_RUNTIME.

        Returns: FLAdminAPIResponse

        """
        failed_attempts = 0
        start = time.time()
        while True:
            try:
                reply = self.check_status(TargetType.CLIENT)
                if reply:
                    met = callback(reply, **kwargs)
                    if met:
                        return FLAdminAPIResponse(APIStatus.SUCCESS, {}, None)
                    fail_attempts = 0
                else:
                    print("Could not get reply from check status client, trying again later")
                    failed_attempts += 1
            except BaseException as e:
                print("Could not get clients stats, trying again later. Exception: ", e)
                failed_attempts += 1

            now = time.time()
            if timeout is not None:
                if now - start >= timeout:
                    return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": "Waited until timeout."}, None)
            if failed_attempts > fail_attempts:
                return FLAdminAPIResponse(
                    APIStatus.ERROR_RUNTIME,
                    {
                        "message": "FL client status was not obtainable for more than the specified number of "
                        "fail_attempts. "
                    },
                    None,
                )
            time.sleep(interval)

    @wrap_with_return_exception_responses
    def wait_until_server_stats(
        self,
        interval: int = 10,
        timeout: int = None,
        callback: Callable[[FLAdminAPIResponse, Optional[List]], bool] = default_stats_handling_cb,
        fail_attempts: int = 6,
        **kwargs,
    ) -> FLAdminAPIResponse:
        """This is similar to wait_until_server_status() and is an example for using other information from a repeated
        call, in this case show_stats(TargetType.SERVER). Custom code can be written to use any data available from any
        call to make decisions for how to proceed. Take caution that the conditions will be met at some point, or
        timeout should be set with logic outside this function to handle checks for potential errors or this may loop
        indefinitely.

        Args:
            interval: in seconds, the time between consecutive checks of the server
            timeout: if set, the amount of time this function will run until before returning a response message
            callback: the reply from show_stats(TargetType.SERVER) will be passed to the callback, along with any additional kwargs
            which can go on to perform additional logic.
            fail_attempts: number of consecutive failed attempts of getting the server status before returning with ERROR_RUNTIME.

        Returns: FLAdminAPIResponse

        """
        failed_attempts = 0
        start = time.time()
        while True:
            try:
                reply = self.show_stats(TargetType.SERVER)
                try:
                    if reply:
                        met = callback(reply, **kwargs)
                        if met:
                            return FLAdminAPIResponse(APIStatus.SUCCESS, {}, None)
                        fail_attempts = 0
                    else:
                        print("Could not get reply from show stats server, trying again later")
                        failed_attempts += 1
                except AttributeError:
                    # if attribute cannot be found, check if app is no longer running to return APIStatus.SUCCESS
                    if reply.get("details").get("message") == "App is not running":
                        return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": "Waited until app not running."}, None)
            except BaseException as e:
                print("Could not get server stats, trying again later. Exception: ", e)
                failed_attempts += 1

            now = time.time()
            if timeout is not None:
                if now - start >= timeout:
                    return FLAdminAPIResponse(APIStatus.SUCCESS, {"message": "Waited until timeout."}, None)
            if failed_attempts > fail_attempts:
                return FLAdminAPIResponse(
                    APIStatus.ERROR_RUNTIME,
                    {
                        "message": "FL server stats was not obtainable for more than the specified number of "
                        "fail_attempts. "
                    },
                    None,
                )
            time.sleep(interval)

    def login(self, username: str):
        result = super().login(username=username)
        return FLAdminAPIResponse(status=result["status"], details=result["details"])

    def login_with_poc(self, username: str, poc_key: str):
        result = super().login_with_poc(username=username, poc_key=poc_key)
        return FLAdminAPIResponse(status=result["status"], details=result["details"])
