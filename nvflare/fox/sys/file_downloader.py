# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.fox.api.ctx import Context
from nvflare.fox.sys.backend import SysBackend
from nvflare.fuel.f3.streaming.file_downloader import FileDownloader


def prepare_file_for_download(
    file_name: str,
    timeout: float,
    ctx: Context,
    file_downloaded_cb=None,
    **cb_kwargs,
):
    backend = ctx.backend
    if not isinstance(backend, SysBackend):
        raise ValueError(f"backend must be SysBackend but got {type(backend)}")

    cell = backend.cell
    tx_id = FileDownloader.new_transaction(
        cell=cell,
        timeout=timeout,
        timeout_cb=None,
    )
    rid = FileDownloader.add_file(
        tx_id,
        file_name,
        file_downloaded_cb=file_downloaded_cb,
        **cb_kwargs,
    )
    return {"source": cell.get_fqcn(), "rid": rid}


def download_file(ref: dict, per_request_timeout: float, ctx: Context):
    backend = ctx.backend
    if not isinstance(backend, SysBackend):
        raise ValueError(f"backend must be SysBackend but got {type(backend)}")

    err, file_path = FileDownloader.download_file(
        from_fqcn=ref.get("source"),
        ref_id=ref.get("rid"),
        per_request_timeout=per_request_timeout,
        cell=backend.cell,
        abort_signal=ctx.abort_signal,
    )
    return err, file_path
