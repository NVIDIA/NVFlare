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
import shutil

from nvflare.fuel.f3.streaming.file_downloader import FileDownloader
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, make_meta
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.log_utils import get_obj_logger


class BinaryTransfer:
    def __init__(self):
        self.logger = get_obj_logger(self)

    @staticmethod
    def tx_path(conn: Connection, tx_id: str, folder_name=None):
        download_dir = conn.get_prop(ConnProps.DOWNLOAD_DIR)
        if not folder_name:
            return os.path.join(download_dir, tx_id)
        else:
            return os.path.join(download_dir, tx_id, folder_name)

    def download_folder(self, conn: Connection, tx_id: str, folder_name: str):
        self.logger.debug(f"download_folder called for {folder_name}")
        tx_path = self.tx_path(conn, tx_id, folder_name)

        engine = conn.get_prop(ConnProps.ENGINE)
        cell = engine.get_cell()
        source_fqcn = cell.get_fqcn()
        download_tid = FileDownloader.new_transaction(
            cell=engine.get_cell(),
            timeout=5,
            timeout_cb=self._cleanup_tx,
            tx_path=tx_path,
        )

        # return list of the files
        files = []
        for dir_path, dir_names, file_names in os.walk(tx_path):
            for f in file_names:
                full_path = os.path.join(dir_path, f)

                ref_id = FileDownloader.add_file(
                    transaction_id=download_tid,
                    file_name=full_path,
                )

                p = os.path.relpath(full_path, tx_path)
                files.append([p, ref_id])

        self.logger.debug(f"files of the folder to download: {files}")
        if len(files) > 0:
            conn.append_string(
                "OK",
                meta=make_meta(
                    MetaStatusValue.OK,
                    extra={
                        MetaKey.SOURCE_FQCN: source_fqcn,
                        MetaKey.FILES: files,
                        MetaKey.TX_ID: tx_id,
                        MetaKey.FOLDER_NAME: folder_name,
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

    def _cleanup_tx(self, tx_id: str, files, tx_path):
        """
        Remove the job download folder
        """
        shutil.rmtree(tx_path, ignore_errors=True)
        self.logger.debug(f"deleted download path: {tx_id=} {tx_path=} {files=}")
