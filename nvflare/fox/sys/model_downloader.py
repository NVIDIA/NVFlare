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
import torch

from nvflare.app_opt.pt.tensor_downloader import TensorDownloader
from nvflare.fox.api.ctx import Context
from nvflare.fox.sys.backend import SysBackend

SOURCE = "source"
REF_ID = "ref_id"


class ModelDownloader:

    def __init__(
        self,
        num_receivers: int,
        timeout: float,
        ctx: Context,
    ):
        backend = ctx.backend
        if not isinstance(backend, SysBackend):
            raise ValueError(f"backend must be SysBackend but got {type(backend)}")

        self.cell = backend.cell
        self.tx_id = TensorDownloader.new_transaction(
            cell=self.cell,
            timeout=timeout,
            num_receivers=num_receivers,
        )

    def add_model(self, model: dict[str, torch.Tensor], num_tensors_per_chunk: int = 1):
        rid = TensorDownloader.add_tensors(
            transaction_id=self.tx_id,
            tensors=model,
            num_tensors_per_chunk=num_tensors_per_chunk,
        )
        return {SOURCE: self.cell.get_fqcn(), REF_ID: rid}


def download_model(ref: dict, per_request_timeout: float, ctx: Context, model_received_cb=None, **cb_kwargs):
    backend = ctx.backend
    if not isinstance(backend, SysBackend):
        raise ValueError(f"backend must be SysBackend but got {type(backend)}")

    return TensorDownloader.download_tensors(
        from_fqcn=ref.get(SOURCE),
        ref_id=ref.get(REF_ID),
        per_request_timeout=per_request_timeout,
        cell=backend.cell,
        abort_signal=ctx.abort_signal,
        tensors_received_cb=model_received_cb,
        **cb_kwargs,
    )
