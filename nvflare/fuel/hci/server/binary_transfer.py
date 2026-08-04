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

from nvflare.fuel.f3.streaming.file_downloader import ObjectDownloader, add_file
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ReplyKeyword, make_meta
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
        admin = conn.get_prop(ConnProps.ADMIN_SERVER)
        session = conn.get_prop(ConnProps.SESSION)
        session_mgr = admin.sess_mgr
        session_id = session.sess_id

        def _download_progress(tx_id, bytes_done, state, **kwargs):
            if bytes_done > 0 and state == TransferProgressState.ACTIVE:
                session_mgr.mark_download_active(session_id, tx_id)

        cell = engine.get_cell()
        source_fqcn = cell.get_fqcn()
        downloader = ObjectDownloader(
            num_receivers=1,
            cell=cell,
            timeout=admin.timeout,
            transaction_done_cb=self._cleanup_tx,
            progress_cb=_download_progress,
            progress_interval=min(30.0, max(0.0, session_mgr.idle_timeout / 3.0)),
            tx_path=tx_path,
            session_mgr=session_mgr,
            session_id=session_id,
        )
        if not session_mgr.bind_download(session_id, downloader.tx_id, downloader.delete_transaction):
            downloader.delete_transaction()
            conn.append_error(ReplyKeyword.SESSION_INACTIVE)
            return

        # return list of the files
        files = []
        for dir_path, dir_names, file_names in os.walk(tx_path):
            for f in file_names:
                full_path = os.path.join(dir_path, f)
                ref_id = add_file(downloader, file_name=full_path)
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
            downloader.delete_transaction()
            conn.append_error(
                "No data to download",
                meta=make_meta(
                    MetaStatusValue.ERROR,
                    info="No data to download",
                ),
            )

    def _cleanup_tx(self, tx_id: str, status, files, tx_path, session_mgr, session_id):
        """
        Remove the job download folder
        """
        session_mgr.end_download(session_id, tx_id)
        shutil.rmtree(tx_path, ignore_errors=True)
        self.logger.debug(f"deleted download path: {tx_id=} {status=} {tx_path=} {files=}")
