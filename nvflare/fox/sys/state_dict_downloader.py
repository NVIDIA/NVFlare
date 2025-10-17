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


class StateDictDownloader:

    def __init__(
        self,
        timeout: float,
        ctx: Context,
        state_dict: dict[str, torch.Tensor],
        num_tensors_per_chunk: int = 1,
        state_dict_downloaded_cb=None,
        **cb_kwargs,
    ):
        self.state_dict_downloaded_cb = state_dict_downloaded_cb
        self.cb_kwargs = cb_kwargs
        self.ctx = ctx

        backend = ctx.backend
        if not isinstance(backend, SysBackend):
            raise ValueError(f"backend must be SysBackend but got {type(backend)}")

        self.cell = backend.cell
        self.tx_id = TensorDownloader.new_transaction(
            cell=self.cell,
            num_tensors_per_chunk=num_tensors_per_chunk,
            timeout=timeout,
            timeout_cb=None,
        )

        self.rid = TensorDownloader.add_state_dict(
            transaction_id=self.tx_id,
            state_dict=state_dict,
            state_dict_downloaded_cb=self._state_dict_downloaded_cb,
        )

    def _state_dict_downloaded_cb(self, rid, to_site, status, obj):
        if self.state_dict_downloaded_cb:
            self.state_dict_downloaded_cb(self, to_site, status, self.ctx, **self.cb_kwargs)

    def get_ref(self):
        return {"source": self.cell.get_fqcn(), "rid": self.rid}

    def clear(self):
        TensorDownloader.delete_transaction(self.tx_id)


def download_state_dict(ref: dict, per_request_timeout: float, ctx: Context, tensors_received_cb=None, **cb_kwargs):
    backend = ctx.backend
    if not isinstance(backend, SysBackend):
        raise ValueError(f"backend must be SysBackend but got {type(backend)}")

    err, result = TensorDownloader.download_state_dict(
        from_fqcn=ref.get("source"),
        ref_id=ref.get("rid"),
        per_request_timeout=per_request_timeout,
        cell=backend.cell,
        abort_signal=ctx.abort_signal,
        tensors_received_cb=tensors_received_cb,
        **cb_kwargs,
    )
    return err, result
