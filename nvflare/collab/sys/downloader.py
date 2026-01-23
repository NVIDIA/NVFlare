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
import numpy as np
import torch

from nvflare.app_common.np.np_downloader import add_arrays
from nvflare.app_common.np.np_downloader import download_arrays as pull_arrays
from nvflare.app_opt.pt.tensor_downloader import add_tensors
from nvflare.app_opt.pt.tensor_downloader import download_tensors as pull_tensors
from nvflare.collab import collab
from nvflare.collab.sys.backend import FlareBackend
from nvflare.fuel.f3.streaming.file_downloader import add_file
from nvflare.fuel.f3.streaming.file_downloader import download_file as pull_file
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader


class DownloadRefKey:
    SOURCE = "source"
    REF_ID = "ref_id"
    OBJECT_TYPE = "object_type"


class ObjectType:
    FILE = "file"
    TENSORS = "tensors"
    ARRAYS = "arrays"


class Downloader(ObjectDownloader):

    def __init__(
        self,
        num_receivers: int,
        timeout: float,
    ):
        ctx = collab.context
        backend = ctx.backend
        if not isinstance(backend, FlareBackend):
            raise ValueError(f"backend must be FlareBackend but got {type(backend)}")

        super().__init__(
            cell=backend.cell,
            timeout=timeout,
            num_receivers=num_receivers,
        )

    def _to_ref(self, obj_type, ref_id):
        return {
            DownloadRefKey.OBJECT_TYPE: obj_type,
            DownloadRefKey.REF_ID: ref_id,
            DownloadRefKey.SOURCE: self.cell.get_fqcn(),
        }

    def add_file(
        self,
        file_name: str,
        chunk_size=None,
        file_downloaded_cb=None,
        **cb_kwargs,
    ):
        rid = add_file(self, file_name, chunk_size=chunk_size, file_downloaded_cb=file_downloaded_cb, **cb_kwargs)
        return self._to_ref(ObjectType.FILE, rid)

    def add_tensors(self, tensors: dict[str, torch.Tensor], max_chunk_size: int = 0):
        rid = add_tensors(self, tensors, max_chunk_size=max_chunk_size)
        return self._to_ref(ObjectType.TENSORS, rid)

    def add_arrays(self, arrays: dict[str, np.ndarray], max_chunk_size: int = 0):
        rid = add_arrays(self, arrays, max_chunk_size=max_chunk_size)
        return self._to_ref(ObjectType.ARRAYS, rid)


def download_file(ref: dict, per_request_timeout: float):
    ctx = collab.context
    backend = ctx.backend
    if not isinstance(backend, FlareBackend):
        raise ValueError(f"backend must be FlareBackend but got {type(backend)}")

    obj_type = ref.get(DownloadRefKey.OBJECT_TYPE)
    if obj_type != ObjectType.FILE:
        raise ValueError(f"obj_type must be {ObjectType.FILE} but got {obj_type}")

    return pull_file(
        from_fqcn=ref.get(DownloadRefKey.SOURCE),
        ref_id=ref.get(DownloadRefKey.REF_ID),
        per_request_timeout=per_request_timeout,
        cell=backend.cell,
        abort_signal=ctx.abort_signal,
    )


def download_tensors(ref: dict, per_request_timeout: float, tensors_received_cb=None, **cb_kwargs):
    ctx = collab.context
    backend = ctx.backend
    if not isinstance(backend, FlareBackend):
        raise ValueError(f"backend must be FlareBackend but got {type(backend)}")

    obj_type = ref.get(DownloadRefKey.OBJECT_TYPE)
    if obj_type != ObjectType.TENSORS:
        raise ValueError(f"obj_type must be {ObjectType.TENSORS} but got {obj_type}")

    return pull_tensors(
        from_fqcn=ref.get(DownloadRefKey.SOURCE),
        ref_id=ref.get(DownloadRefKey.REF_ID),
        per_request_timeout=per_request_timeout,
        cell=backend.cell,
        abort_signal=ctx.abort_signal,
        tensors_received_cb=tensors_received_cb,
        **cb_kwargs,
    )


def download_arrays(ref: dict, per_request_timeout: float, arrays_received_cb=None, **cb_kwargs):
    ctx = collab.context
    backend = ctx.backend
    if not isinstance(backend, FlareBackend):
        raise ValueError(f"backend must be FlareBackend but got {type(backend)}")

    obj_type = ref.get(DownloadRefKey.OBJECT_TYPE)
    if obj_type != ObjectType.ARRAYS:
        raise ValueError(f"obj_type must be {ObjectType.ARRAYS} but got {obj_type}")

    return pull_arrays(
        from_fqcn=ref.get(DownloadRefKey.SOURCE),
        ref_id=ref.get(DownloadRefKey.REF_ID),
        per_request_timeout=per_request_timeout,
        cell=backend.cell,
        abort_signal=ctx.abort_signal,
        arrays_received_cb=arrays_received_cb,
        **cb_kwargs,
    )
