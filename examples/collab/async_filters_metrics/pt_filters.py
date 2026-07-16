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
from typing import Any

from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable, download_tensors
from nvflare.collab.api import CallFilter, Context, ResultFilter
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.utils.log_utils import get_obj_logger


def _add_tensors(tensors, num_receivers: int, context: Context):
    cell = context.backend.cell
    tx_id = DownloadService.new_transaction(cell=cell, timeout=5.0, num_receivers=num_receivers)
    ref_id = DownloadService.add_object(tx_id, TensorDownloadable(tensors, max_chunk_size=0))
    return cell.get_fqcn(), ref_id


def _download_tensors(ref, context: Context):
    source_fqcn, ref_id = ref
    return download_tensors(
        from_fqcn=source_fqcn,
        ref_id=ref_id,
        per_request_timeout=5.0,
        cell=context.backend.cell,
        abort_signal=context.abort_signal,
    )


class OutgoingModelCallFilter(CallFilter):

    def __init__(self, model_arg_name: str):
        super().__init__()
        self.model_arg_name = model_arg_name
        self.logger = get_obj_logger(self)

    def filter_call(self, func_kwargs: dict, context: Context):
        arg_value = func_kwargs.get(self.model_arg_name)
        if not arg_value:
            return func_kwargs

        num_receivers = context.target_group_size
        self.logger.info(f"target group size={num_receivers}")

        func_kwargs[self.model_arg_name] = _add_tensors(arg_value, num_receivers, context)
        return func_kwargs


class IncomingModelCallFilter(CallFilter):

    def __init__(self, model_arg_name: str):
        super().__init__()
        self.model_arg_name = model_arg_name
        self.logger = get_obj_logger(self)

    def filter_call(self, func_kwargs: dict, context: Context):
        arg_value = func_kwargs.get(self.model_arg_name)
        if not arg_value:
            return func_kwargs

        err, model = _download_tensors(arg_value, context)
        if err:
            self.logger.error(f"error filtering call arg {arg_value}: {err}")
        else:
            func_kwargs[self.model_arg_name] = model
        return func_kwargs


class OutgoingModelResultFilter(ResultFilter):

    def filter_result(self, result: Any, context: Context):
        if not isinstance(result, dict):
            return result

        return _add_tensors(result, num_receivers=1, context=context)


class IncomingModelResultFilter(ResultFilter):

    def __init__(self):
        super().__init__()
        self.logger = get_obj_logger(self)

    def filter_result(self, result: Any, context: Context):
        err, model = _download_tensors(result, context)
        if err:
            self.logger.error(f"error filtering result {result}: {err}")
            return result
        else:
            return model
