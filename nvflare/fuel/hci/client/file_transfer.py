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

import os
import traceback

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.fuel.hci.base64_utils import (
    b64str_to_binary_file,
    b64str_to_bytes,
    b64str_to_text_file,
    binary_file_to_b64str,
    bytes_to_b64str,
    text_file_to_b64str,
)
from nvflare.fuel.hci.cmd_arg_utils import join_args
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.table import Table
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes

from .lib import AdminClient, ReplyProcessor


def _server_cmd_name(name: str):
    return ftd.SERVER_MODULE_NAME + "." + name


class _DownloadProcessor(ReplyProcessor):
    """Reply processor to handle downloads."""

    def __init__(self, download_dir: str, str_to_file_func):
        self.download_dir = download_dir
        self.str_to_file_func = str_to_file_func
        self.data_received = False
        self.table = None

    def reply_start(self, client, reply_json):
        self.data_received = False
        self.table = Table(["file", "size"])

    def reply_done(self, client):
        if not self.data_received:
            client.write_error("protocol error - no data received")
        else:
            client.write_table(self.table)

    def process_table(self, client, table: Table):
        try:
            rows = table.rows
            if len(rows) < 1:
                # no data
                client.write_error("protocol error - no file data")
                return

            for i in range(len(rows)):
                if i == 0:
                    # this is header
                    continue

                row = rows[i]
                if len(row) < 1:
                    client.write_error("protocol error - missing file name")
                    return

                if len(row) < 2:
                    client.write_error("protocol error - missing file data")
                    return

                file_name = row[0]
                encoded_str = row[1]
                full_path = os.path.join(self.download_dir, file_name)
                num_bytes = self.str_to_file_func(encoded_str, full_path)
                self.table.add_row([file_name, str(num_bytes)])
                self.data_received = True
        except Exception as ex:
            traceback.print_exc()
            client.write_error("exception processing file: {}".format(ex))


class _DownloadFolderProcessor(ReplyProcessor):
    """
    Reply processor for handling downloading directories.
    """

    def __init__(self, download_dir: str):
        self.download_dir = download_dir
        self.data_received = False
        self.table = None

    def reply_start(self, client, reply_json):
        self.data_received = False

    def reply_done(self, client):
        if not self.data_received:
            client.write_error("protocol error - no data received")

    def process_string(self, client, item: str):
        try:
            self.data_received = True
            data_bytes = b64str_to_bytes(item)
            unzip_all_from_bytes(data_bytes, self.download_dir)
        except Exception as ex:
            traceback.print_exc()
            client.write_error("exception processing reply: {}".format(ex))


class FileTransferModule(CommandModule):
    """
    Command module with commands relevant to file transfer.
    """

    def __init__(self, upload_dir: str, download_dir: str, upload_folder_cmd_name="upload_app"):
        assert os.path.isdir(upload_dir), "upload_dir {} is not a valid dir".format(upload_dir)

        assert os.path.isdir(download_dir), "download_dir {} is not a valid dir".format(download_dir)

        self.upload_dir = upload_dir
        self.download_dir = download_dir
        self.upload_folder_cmd_name = upload_folder_cmd_name

    def get_spec(self):
        return CommandModuleSpec(
            name="file_transfer",
            cmd_specs=[
                CommandSpec(
                    name="upload_text",
                    description="upload one or more text files in the upload_dir",
                    usage="upload_text file_name ...",
                    handler_func=self.upload_text_file,
                    visible=False,
                ),
                CommandSpec(
                    name="download_text",
                    description="download one or more text files in the download_dir",
                    usage="download_text file_name ...",
                    handler_func=self.download_text_file,
                ),
                CommandSpec(
                    name="upload_binary",
                    description="upload one or more binary files in the upload_dir",
                    usage="upload_binary file_name ...",
                    handler_func=self.upload_binary_file,
                    visible=False,
                ),
                CommandSpec(
                    name="download_binary",
                    description="download one or more binary files in the download_dir",
                    usage="download_binary file_name ...",
                    handler_func=self.download_binary_file,
                ),
                CommandSpec(
                    name=self.upload_folder_cmd_name,
                    description="upload application to the server",
                    usage=self.upload_folder_cmd_name + " application_folder",
                    handler_func=self.upload_folder,
                ),
                CommandSpec(
                    name="download_folder",
                    description="download a folder from the server",
                    usage="download_folder folder_name",
                    handler_func=self.download_folder,
                ),
                CommandSpec(
                    name="info",
                    description="show folder setup info",
                    usage="info",
                    handler_func=self.info,
                ),
            ],
        )

    def upload_file(self, args, admin: AdminClient, cmd_name, file_to_str_func):
        full_cmd_name = _server_cmd_name(cmd_name)
        if len(args) < 2:
            admin.write_error("syntax error: missing file names")
            return

        parts = [full_cmd_name]
        for i in range(1, len(args)):
            file_name = args[i]
            full_path = os.path.join(self.upload_dir, file_name)
            if not os.path.isfile(full_path):
                admin.write_error("no such file: {}".format(full_path))
                return

            encoded_string = file_to_str_func(full_path)
            parts.append(file_name)
            parts.append(encoded_string)

        command = join_args(parts)
        admin.server_execute(command)

    def upload_text_file(self, args, admin: AdminClient):
        self.upload_file(args, admin, ftd.SERVER_CMD_UPLOAD_TEXT, text_file_to_b64str)

    def upload_binary_file(self, args, admin: AdminClient):
        self.upload_file(args, admin, ftd.SERVER_CMD_UPLOAD_BINARY, binary_file_to_b64str)

    def download_file(self, args, admin: AdminClient, cmd_name, str_to_file_func):
        full_cmd_name = _server_cmd_name(cmd_name)
        if len(args) < 2:
            admin.write_error("syntax error: missing file names")
            return

        parts = [full_cmd_name]
        for i in range(1, len(args)):
            file_name = args[i]
            parts.append(file_name)

        command = join_args(parts)
        reply_processor = _DownloadProcessor(self.download_dir, str_to_file_func)
        admin.server_execute(command, reply_processor)

    def download_text_file(self, args, admin: AdminClient):
        self.download_file(args, admin, ftd.SERVER_CMD_DOWNLOAD_TEXT, b64str_to_text_file)

    def download_binary_file(self, args, admin: AdminClient):
        self.download_file(args, admin, ftd.SERVER_CMD_DOWNLOAD_BINARY, b64str_to_binary_file)

    def upload_folder(self, args, admin: AdminClient):
        if len(args) != 2:
            admin.write_error("usage: " + self.upload_folder_cmd_name + " folder_name")
            return

        folder_name = args[1]
        if folder_name.endswith("/"):
            folder_name = folder_name.rstrip("/")

        full_path = os.path.join(self.upload_dir, folder_name)
        if not os.path.isdir(full_path):
            admin.write_error('"{}" is not a valid folder'.format(full_path))
            return

        # zip the data
        data = zip_directory_to_bytes(self.upload_dir, folder_name)

        # prepare for upload
        folder_name = os.path.split(folder_name)[1]
        rel_path = os.path.relpath(full_path, self.upload_dir)
        folder_name = self._remove_loading_dotdot(rel_path)

        b64str = bytes_to_b64str(data)
        parts = [_server_cmd_name(ftd.SERVER_CMD_UPLOAD_FOLDER), folder_name, b64str]
        command = join_args(parts)
        admin.server_execute(command)

    def download_folder(self, args, admin: AdminClient):
        if len(args) != 2:
            admin.write_error("usage: download_folder folder_name")
            return

        parts = [_server_cmd_name(ftd.SERVER_CMD_DOWNLOAD_FOLDER), args[1]]
        command = join_args(parts)
        reply_processor = _DownloadFolderProcessor(self.download_dir)
        admin.server_execute(command, reply_processor)

    def info(self, args, admin: AdminClient):
        admin.write_string("Local Upload Source: {}".format(self.upload_dir))
        admin.write_string("Local Download Destination: {}".format(self.download_dir))
        admin.server_execute(_server_cmd_name(ftd.SERVER_CMD_INFO))

    def _remove_loading_dotdot(self, path):
        while path.startswith("../"):
            path = path[3:]
        return path
