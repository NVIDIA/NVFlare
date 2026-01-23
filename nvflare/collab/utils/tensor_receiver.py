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
from nvflare.collab import collab
from nvflare.collab.api.gcc import GroupCallContext
from nvflare.collab.sys.downloader import download_tensors
from nvflare.fuel.utils.log_utils import get_obj_logger


class TensorReceiver:
    """This class implements the callback function to add partially received tensors to the result queue.
    This will enable the semi-in-time tensor aggregation by iterating through result queue.
    To use, the application simply sets the process_resp_cb to an instance of this class when making group call.

    Example:
        collab.clients(
            blocking=False,
            process_resp_cb=TensorReceiver(),
        ).train(...)
    """

    def __init__(self):
        self.logger = get_obj_logger(self)

    def __call__(self, gcc: GroupCallContext, result):
        self.logger.info(f"[{collab.call_info}] got train result from {collab.caller}: {result}")
        model, model_type = result
        if model_type == "ref":
            err, _ = download_tensors(
                ref=model,
                per_request_timeout=5.0,
                tensors_received_cb=self._receive_tensors,
                gcc=gcc,
            )
            if err:
                raise RuntimeError(f"failed to download model {model}: {err}")
            else:
                return None
        else:
            return model

    def _receive_tensors(self, tensors, gcc: GroupCallContext):
        self.logger.info(f"adding partial result: {tensors}")
        gcc.add_partial_result(tensors)
        return None
