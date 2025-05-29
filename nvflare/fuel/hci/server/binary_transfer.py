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

from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, StreamChannel, StreamCtxKey, make_meta
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.log_utils import get_obj_logger


class BinaryTransfer:
    def __init__(self):
        self.logger = get_obj_logger(self)

    def download_file(self, conn: Connection, tx_id: str, folder_name: str, file_name: str):
        conn.binary_mode = True
        tx_path = self.tx_path(conn, tx_id, folder_name)
        full_path = os.path.join(tx_path, file_name)
        if not os.path.exists(full_path):
            self.logger.error(f"no such file: {full_path}")
            return

        if not os.path.isfile(full_path):
            self.logger.error(f"not a file: {full_path}")
            return

        request = conn.get_prop(ConnProps.REQUEST)
        assert isinstance(request, CellMessage)
        target = request.get_header(MessageHeaderKey.ORIGIN)

        self.logger.info(f"streaming {full_path} to {target} ...")
        engine = conn.get_prop(ConnProps.ENGINE)
        stream_ctx = {StreamCtxKey.TX_ID: tx_id}
        with engine.new_context() as fl_ctx:
            rc, _ = FileStreamer.stream_file(
                channel=StreamChannel.DOWNLOAD,
                topic="file",
                stream_ctx=stream_ctx,
                targets=[target],
                file_name=full_path,
                fl_ctx=fl_ctx,
            )
            if rc != ReturnCode.OK:
                self.logger.error(f"Failed to stream file {full_path} to {target}")
                return
        bytes_sent = FileStreamer.get_file_size(stream_ctx)
        self.logger.info(f"sent {full_path} to {target}: {bytes_sent} bytes")

    @staticmethod
    def tx_path(conn: Connection, tx_id: str, folder_name=None):
        download_dir = conn.get_prop(ConnProps.DOWNLOAD_DIR)
        if not folder_name:
            return os.path.join(download_dir, tx_id)
        else:
            return os.path.join(download_dir, tx_id, folder_name)

    def download_folder(self, conn: Connection, tx_id: str, folder_name: str, download_file_cmd_name: str):
        self.logger.debug(f"download_folder called for {folder_name}")
        tx_path = self.tx_path(conn, tx_id, folder_name)

        # return list of the files
        files = []
        for dir_path, dir_names, file_names in os.walk(tx_path):
            for f in file_names:
                p = os.path.join(dir_path, f)
                p = os.path.relpath(p, tx_path)
                files.append(p)

        self.logger.debug(f"files of the folder: {files}")
        if len(files) > 0:
            conn.append_string(
                "OK",
                meta=make_meta(
                    MetaStatusValue.OK,
                    extra={
                        MetaKey.FILES: files,
                        MetaKey.TX_ID: tx_id,
                        MetaKey.FOLDER_NAME: folder_name,
                        MetaKey.CMD_NAME: download_file_cmd_name,
                    },
                ),
            )
        else:
            conn.append_error(
                "No data to download",
                meta=make_meta(
                    MetaStatusValue.ERROR,
                    info="No data to download",
                ),
            )
