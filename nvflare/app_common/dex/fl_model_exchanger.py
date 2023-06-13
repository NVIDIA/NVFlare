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

from typing import Optional, Tuple

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.dex.dxo_exchanger import DXOExchanger
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class FLModelExchanger:
    def __init__(self, exchanger: DXOExchanger):
        self.exchanger = exchanger

    def send_request(self, model: FLModel, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """Sends a request.

        Args:
            model: The FLModel object to be sent.
            timeout: how long to wait for the peer to read the data.

        Returns:
            A tuple of (has_been_read, request message id).
            has_been_read indicates whether the peer has read the data.
        """
        dxo = FLModelUtils.to_dxo(model)
        return self.exchanger.send_request(dxo, timeout)

    def receive_request(self, timeout: Optional[float] = None) -> Tuple[FLModel, str]:
        """Receives a request.

        Args:
            timeout: how long to wait for the request to come.

        Returns:
            A tuple of (FLModel, request id).
        """
        dxo, req_id = self.exchanger.receive_request(timeout=timeout)
        return FLModelUtils.from_dxo(dxo), req_id

    def send_reply(self, model: FLModel, req_id: str, timeout: Optional[float] = None) -> bool:
        """Sends a reply.

        Args:
            model: The FLModel object to be sent.
            req_id: request ID.
            timeout: how long to wait for the peer to read the data.

        Returns:
            A bool indicates whether the peer has read the data.
        """
        dxo = FLModelUtils.to_dxo(model)
        return self.exchanger.send_reply(dxo, req_id, timeout)

    def receive_reply(self, req_id: str, timeout: Optional[float] = None) -> FLModel:
        """Receives a reply.

        Args:
            req_id: request ID.
            timeout: how long to wait for the request to come.

        Returns:
            A data of type FLModel.
        """
        dxo = self.exchanger.receive_reply(req_id, timeout)
        return FLModelUtils.from_dxo(dxo)
