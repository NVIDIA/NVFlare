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

import logging
import os

from nvflare.fuel.hci.chunk import MAX_CHUNK_SIZE, Sender
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, make_meta
from nvflare.fuel.hci.server.constants import ConnProps


class _BytesSender:
    def __init__(self, conn: Connection):
        self.conn = conn

    def send(self, data):
        self.conn.flush_bytes(data)


class BinaryTransfer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_file(self, conn: Connection, file_name):
        download_dir = conn.get_prop(ConnProps.DOWNLOAD_DIR)
        conn.binary_mode = True
        full_path = os.path.join(download_dir, file_name)
        if not os.path.exists(full_path):
            self.logger.error(f"no such file: {full_path}")
            return

        if not os.path.isfile(full_path):
            self.logger.error(f"not a file: {full_path}")
            return

        self.logger.debug(f"called to send {full_path} ...")
        bytes_sender = _BytesSender(conn)
        sender = Sender(send_data_func=bytes_sender.send)
        buffer_size = MAX_CHUNK_SIZE
        bytes_sent = 0
        with open(full_path, mode="rb") as f:
            chunk = f.read(buffer_size)
            while chunk:
                sender.send(chunk)
                bytes_sent += len(chunk)
                chunk = f.read(buffer_size)
            sender.close()
        self.logger.debug(f"finished sending {full_path}: {bytes_sent} bytes sent")

    def download_folder(self, conn: Connection, folder_name: str, download_file_cmd_name: str, control_id: str):
        download_dir = conn.get_prop(ConnProps.DOWNLOAD_DIR)
        folder_path = os.path.join(download_dir, folder_name)
        self.logger.debug(f"download_folder called for {folder_name}")

        # return list of the files
        files = []
        for (dir_path, dir_names, file_names) in os.walk(folder_path):
            for f in file_names:
                p = os.path.join(dir_path, f)
                p = os.path.relpath(p, folder_path)
                p = os.path.join(folder_name, p)
                files.append(p)

        self.logger.debug(f"files of the folder: {files}")
        conn.append_string(
            "OK",
            meta=make_meta(
                MetaStatusValue.OK,
                extra={MetaKey.FILES: files, MetaKey.CONTROL_ID: control_id, MetaKey.CMD_NAME: download_file_cmd_name},
            ),
        )

        user_name = conn.get_prop(ConnProps.USER_NAME, "?")
        self.logger.info(f"downloaded {control_id} to user {user_name}")
