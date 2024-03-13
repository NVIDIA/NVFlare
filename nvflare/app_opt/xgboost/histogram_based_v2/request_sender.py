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
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.sender import Sender
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.obj_utils import get_logger


class RequestSender:
    """
    A wrapper class for Sender to handle reply
    """

    def __init__(self, engine, sender: Sender, timeout: float = 30):
        """Constructor

        Args:
            engine: the client engine that can send aux messages
            sender: Sender component
        """
        self.engine = engine
        self.sender = sender
        self.timeout = timeout
        self.logger = get_logger(self)

    def _extract_result(self, reply, expected_op):
        if not reply:
            return None
        if not isinstance(reply, dict):
            self.logger.error(f"expect reply to be a dict but got {type(reply)}")
            return None
        result = reply.get(FQCN.ROOT_SERVER)
        if not result:
            self.logger.error(f"no reply from {FQCN.ROOT_SERVER} for request {expected_op}")
            return None
        if not isinstance(result, Shareable):
            self.logger.error(f"expect result to be a Shareable but got {type(result)}")
            return None
        rc = result.get_return_code()
        if rc != ReturnCode.OK:
            self.logger.error(f"server failed to process request: {rc=}")
            return None
        reply_op = result.get_header(Constant.MSG_KEY_XGB_OP)
        if reply_op != expected_op:
            self.logger.error(f"received op {reply_op} != expected op {expected_op}")
            return None
        return result

    def send_to_server(self, op: str, req: Shareable, abort_signal: Signal):
        """Send an XGB request to the server.

        Args:
            op: the XGB operation code
            req: the XGB request
            abort_signal: used for checking whether the job is aborted.

        Returns: reply from the server
        """
        req.set_header(Constant.MSG_KEY_XGB_OP, op)

        reply = self.sender.send_to_server(self.engine, Constant.TOPIC_XGB_REQUEST, req, self.timeout, abort_signal)

        return self._extract_result(reply, op)
