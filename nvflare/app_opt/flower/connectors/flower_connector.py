# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from abc import abstractmethod

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.tie.connector import Connector
from nvflare.app_opt.flower.defs import Constant
from nvflare.fuel.utils.validation_utils import check_positive_int, check_positive_number


class FlowerServerConnector(Connector):
    """
    FlowerServerConnector specifies commonly required methods for server connector implementations.
    """

    def __init__(self):
        Connector.__init__(self)
        self.num_rounds = None

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by Flower Controller to configure the site.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        if num_rounds is None:
            raise RuntimeError("num_rounds is not configured")

        check_positive_int(Constant.CONF_KEY_NUM_ROUNDS, num_rounds)
        self.num_rounds = num_rounds

    @abstractmethod
    def send_request_to_flower(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Send request to the Flower server.
        Subclass must implement this method to send this request to the Flower server.

        Args:
            request: the request received from FL client
            fl_ctx: the FL context

        Returns: reply from the Flower server converted to Shareable

        """
        pass

    def process_app_request(self, op: str, request: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """This method is called by the FL Server when the request is received from a FL client.

        Args:
            op: the op code of the request.
            request: the request received from FL client
            fl_ctx: FL context
            abort_signal: abort signal that could be triggered during the process

        Returns: response from the Flower server converted to Shareable

        """
        stopped, ec = self._is_stopped()
        if stopped:
            self.log_warning(fl_ctx, f"dropped request '{op}' since connector is already stopped {ec=}")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        reply = self.send_request_to_flower(request, fl_ctx)
        self.log_info(fl_ctx, f"received reply for '{op}'")
        return reply


class FlowerClientConnector(Connector):
    """
    FlowerClientConnector defines commonly required methods for client connector implementations.
    """

    def __init__(self, per_msg_timeout: float, tx_timeout: float):
        """Constructor of FlowerClientConnector

        Args:
            per_msg_timeout: per-msg timeout to be used when sending request to server via ReliableMessage
            tx_timeout: tx timeout to be used when sending request to server via ReliableMessage
        """
        check_positive_number("per_msg_timeout", per_msg_timeout)
        check_positive_number("tx_timeout", tx_timeout)

        Connector.__init__(self)
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.stopped = False
        self.num_rounds = None

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by Flower Executor to configure the target.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        if num_rounds is None:
            raise RuntimeError("num_rounds is not configured")

        check_positive_int(Constant.CONF_KEY_NUM_ROUNDS, num_rounds)
        self.num_rounds = num_rounds

    def _send_flower_request(self, request: Shareable) -> Shareable:
        """Send Flower request to the FL server via FLARE message.

        Args:
            request: shareable that contains flower msg

        Returns: operation result

        """
        op = "request"
        reply = self.send_request(
            op=op,
            target=None,  # server
            request=request,
            per_msg_timeout=self.per_msg_timeout,
            tx_timeout=self.tx_timeout,
            fl_ctx=None,
        )
        if not isinstance(reply, Shareable):
            raise RuntimeError(f"invalid reply for op {op}: expect Shareable but got {type(reply)}")
        return reply
