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
import shutil
import tempfile
import traceback
from typing import List

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.fuel.hci.base64_utils import (
    b64str_to_binary_file,
    b64str_to_bytes,
    b64str_to_text_file,
    binary_file_to_b64str,
    bytes_to_b64str,
    text_file_to_b64str,
)
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import AuthzContext
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes


def upload_folder_authz_func_signature(folder_path: str) -> (str, AuthzContext):
    pass


class FileTransferModule(CommandModule):
    """
    Command module for file transfers.
    """

    def __init__(self, upload_dir: str, download_dir: str, upload_folder_authz_func=None):
        assert os.path.isdir(upload_dir), "upload_dir {} is not a valid dir".format(upload_dir)

        assert os.path.isdir(download_dir), "download_dir {} is not a valid dir".format(download_dir)

        self.upload_dir = upload_dir
        self.download_dir = download_dir
        self.upload_folder_authz_func = upload_folder_authz_func

    def get_spec(self):
        return CommandModuleSpec(
            name=ftd.SERVER_MODULE_NAME,
            cmd_specs=[
                CommandSpec(
                    name=ftd.SERVER_CMD_UPLOAD_TEXT,
                    description="upload one or more text files",
                    usage="_upload name1 data1 name2 data2 ...",
                    handler_func=self.upload_text_file,
                    visible=False,
                ),
                CommandSpec(
                    name=ftd.SERVER_CMD_DOWNLOAD_TEXT,
                    description="download one or more text files",
                    usage="download file_name ...",
                    handler_func=self.download_text_file,
                    visible=False,
                ),
                CommandSpec(
                    name=ftd.SERVER_CMD_UPLOAD_BINARY,
                    description="upload one or more binary files",
                    usage="upload name1 data1 name2 data2 ...",
                    handler_func=self.upload_binary_file,
                    visible=False,
                ),
                CommandSpec(
                    name=ftd.SERVER_CMD_DOWNLOAD_BINARY,
                    description="download one or more binary files",
                    usage="download file_name ...",
                    handler_func=self.download_binary_file,
                    visible=False,
                ),
                CommandSpec(
                    name=ftd.SERVER_CMD_UPLOAD_FOLDER,
                    description="upload a folder from client",
                    usage="upload_folder folder_name",
                    handler_func=self.upload_folder,
                    authz_func=self._authorize_upload_folder,
                    visible=False,
                ),
                CommandSpec(
                    name=ftd.SERVER_CMD_DOWNLOAD_FOLDER,
                    description="download a folder to client",
                    usage="download folder_name",
                    handler_func=self.download_folder,
                    visible=False,
                ),
                CommandSpec(
                    name=ftd.SERVER_CMD_INFO,
                    description="show info",
                    usage="info",
                    handler_func=self.info,
                    visible=False,
                ),
            ],
        )

    def upload_file(self, conn: Connection, args: List[str], str_to_file_func):
        if len(args) < 3:
            conn.append_error("syntax error: missing files")
            return

        if len(args) % 2 != 1:
            conn.append_error("syntax error: file name/data not paired")
            return

        table = conn.append_table(["file", "size"])
        i = 1
        while i < len(args):
            name = args[i]
            data = args[i + 1]
            i += 2

            full_path = os.path.join(self.upload_dir, name)
            num_bytes = str_to_file_func(b64str=data, file_name=full_path)
            table.add_row([name, str(num_bytes)])

    def upload_text_file(self, conn: Connection, args: List[str]):
        self.upload_file(conn, args, b64str_to_text_file)

    def upload_binary_file(self, conn: Connection, args: List[str]):
        self.upload_file(conn, args, b64str_to_binary_file)

    def download_file(self, conn: Connection, args: List[str], file_to_str_func):
        if len(args) < 2:
            conn.append_error("syntax error: missing file names")
            return

        table = conn.append_table(["name", "data"])
        for i in range(1, len(args)):
            file_name = args[i]
            full_path = os.path.join(self.download_dir, file_name)
            if not os.path.exists(full_path):
                conn.append_error("no such file: {}".format(file_name))
                continue

            if not os.path.isfile(full_path):
                conn.append_error("not a file: {}".format(file_name))
                continue

            encoded_str = file_to_str_func(full_path)
            table.add_row([file_name, encoded_str])

    def download_text_file(self, conn: Connection, args: List[str]):
        self.download_file(conn, args, text_file_to_b64str)

    def download_binary_file(self, conn: Connection, args: List[str]):
        self.download_file(conn, args, binary_file_to_b64str)

    def _authorize_upload_folder(self, conn: Connection, args: List[str]):
        if len(args) != 3:
            conn.append_error("syntax error: require data")
            return False, None

        folder_name = args[1]
        zip_b64str = args[2]
        tmp_dir = tempfile.mkdtemp()

        try:
            data_bytes = b64str_to_bytes(zip_b64str)
            unzip_all_from_bytes(data_bytes, tmp_dir)
            tmp_folder_path = os.path.join(tmp_dir, folder_name)

            if not os.path.isdir(tmp_folder_path):
                conn.append_error("logic error: unzip failed to create folder {}".format(tmp_folder_path))
                return False, None

            if self.upload_folder_authz_func:
                err, authz_ctx = self.upload_folder_authz_func(tmp_folder_path)
                if err is None:
                    err = ""
                elif not isinstance(err, str):
                    # the validator failed to follow signature
                    # assuming things are bad
                    err = "folder validation failed"

                if len(err) > 0:
                    conn.append_error(err)
                    return False, None
                else:
                    return True, authz_ctx
            else:
                return True, None
        except BaseException:
            traceback.print_exc()
            conn.append_error("exception occurred")
            return False, None
        finally:
            shutil.rmtree(tmp_dir)

    def upload_folder(self, conn: Connection, args: List[str]):
        folder_name = args[1]
        zip_b64str = args[2]
        folder_path = os.path.join(self.upload_dir, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        data_bytes = b64str_to_bytes(zip_b64str)
        unzip_all_from_bytes(data_bytes, self.upload_dir)
        conn.set_prop("upload_folder_path", folder_path)
        conn.append_string("Created folder {}".format(folder_path))

    def download_folder(self, conn: Connection, args: List[str]):
        if len(args) != 2:
            conn.append_error("syntax error: require folder name")
            return

        folder_name = args[1]
        full_path = os.path.join(self.download_dir, folder_name)
        if not os.path.exists(full_path):
            conn.append_error("no such folder: {}".format(full_path))
            return

        if not os.path.isdir(full_path):
            conn.append_error("'{}' is not a valid folder".format(full_path))
            return

        try:
            data = zip_directory_to_bytes(self.download_dir, folder_name)
            b64str = bytes_to_b64str(data)
            conn.append_string(b64str)
        except BaseException:
            traceback.print_exc()
            conn.append_error("exception occurred")

    def info(self, conn: Connection, args: List[str]):
        conn.append_string("Server Upload Destination: {}".format(self.upload_dir))
        conn.append_string("Server Download Source: {}".format(self.download_dir))
