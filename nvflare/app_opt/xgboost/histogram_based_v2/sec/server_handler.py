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
import os
import threading

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.sec.sec_handler import SecurityHandler

try:
    from nvflare.app_opt.he import decomposers

    tenseal_imported = True
except Exception:
    tenseal_imported = False


class ServerSecurityHandler(SecurityHandler):
    def __init__(self):
        FLComponent.__init__(self)
        self.encrypted_gh = None
        self.gh_source_rank = 0
        self.gh_seq = 0
        self.gh_original_buf_size = 0
        self.aggr_seq = 0
        self.aggr_result_dict = None
        self.aggr_result_to_send = None
        self.aggr_result_lock = threading.Lock()
        self.world_size = 0
        self.size_dict = None

        if tenseal_imported:
            decomposers.register()

    def _process_before_broadcast(self, fl_ctx: FLContext):
        self.info(fl_ctx, "start")
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        seq = fl_ctx.get_prop(Constant.PARAM_KEY_SEQ)

        request = fl_ctx.get_prop(Constant.PARAM_KEY_REQUEST)
        assert isinstance(request, Shareable)
        has_encrypted_gh = request.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        self.info(fl_ctx, f"{has_encrypted_gh=}")
        if not has_encrypted_gh:
            self.info(fl_ctx, "not for gh broadcast - ignore")
            return

        self.encrypted_gh = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        self.gh_source_rank = rank
        self.gh_seq = seq
        self.gh_original_buf_size = request.get_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE)
        self.info(fl_ctx, f"got gh bcst: encrypted_gh={len(self.encrypted_gh)} orig_buf={self.gh_original_buf_size}")
        # only need to send a small dummy buffer to the server
        dummy_buf = os.urandom(Constant.DUMMY_BUFFER_SIZE)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=dummy_buf, private=True, sticky=False)

    def _process_after_broadcast(self, fl_ctx: FLContext):
        # this is called after the Server already received broadcast calls from all clients of the same sequence
        self.info(fl_ctx, "start")
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        seq = fl_ctx.get_prop(Constant.PARAM_KEY_SEQ)

        if seq != self.gh_seq:
            # this is not a gh broadcast
            self.info(fl_ctx, "not gh bcast - ignore")
            return

        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)
        reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
        reply.set_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE, self.gh_original_buf_size)

        if rank == self.gh_source_rank:
            # no need to send any data back to label client
            self.info(fl_ctx, f"return dummy to gh source {rank}")
            fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=None, private=True, sticky=False)
            return

        # send encrypted ghs
        self.info(fl_ctx, f"return {len(self.encrypted_gh)=} to non-label {rank}")
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=self.encrypted_gh, private=True, sticky=False)

    def _process_before_all_gather_v(self, fl_ctx: FLContext):
        request = fl_ctx.get_prop(Constant.PARAM_KEY_REQUEST)
        assert isinstance(request, Shareable)
        has_encrypted_data = request.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        self.info(fl_ctx, f"{has_encrypted_data=}")
        if not has_encrypted_data:
            self.info(fl_ctx, "start - non-secure data")
            return

        horizontal = request.get_header(Constant.HEADER_KEY_HORIZONTAL)
        training_mode = "horizontal" if horizontal else "vertical"
        self.info(fl_ctx, f"start - {training_mode}")

        fl_ctx.set_prop(key=Constant.HEADER_KEY_IN_AGGR, value=True, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.HEADER_KEY_HORIZONTAL, value=horizontal, private=True, sticky=False)

        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        if send_buf:
            if horizontal:
                length = send_buf.size()
            else:
                length = len(send_buf)
            # the send_buf contains encoded aggr result (str) or CKKS vector from this rank
            self.info(fl_ctx, f"got encrypted aggr data: {length} bytes")
            with self.aggr_result_lock:
                self.aggr_result_to_send = None
                if not self.aggr_result_dict:
                    self.aggr_result_dict = {}
                self.aggr_result_dict[rank] = send_buf
        else:
            self.info(fl_ctx, f"no aggr data from {rank=}")

        if self.size_dict is None:
            self.size_dict = {}

        self.size_dict[rank] = request.get_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE)
        # only send a dummy to the Server
        fl_ctx.set_prop(
            key=Constant.PARAM_KEY_SEND_BUF, value=os.urandom(Constant.DUMMY_BUFFER_SIZE), private=True, sticky=False
        )
        self.info(fl_ctx, "send dummy buf to XGB server")

    def _process_after_all_gather_v(self, fl_ctx: FLContext):
        # this is called after the Server has finished gathering
        # Note: this fl_ctx is the same as the one in _process_before_all_gather_v!
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        in_aggr = fl_ctx.get_prop(Constant.HEADER_KEY_IN_AGGR)
        self.info(fl_ctx, f"start {in_aggr=}")

        if not in_aggr:
            self.info(fl_ctx, "not in_aggr - ignore")
            return

        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)
        horizontal = fl_ctx.get_prop(Constant.HEADER_KEY_HORIZONTAL)
        reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
        reply.set_header(Constant.HEADER_KEY_HORIZONTAL, horizontal)

        with self.aggr_result_lock:
            if not self.aggr_result_to_send:
                if not self.aggr_result_dict:
                    return self._abort(f"Rank {rank}: no aggr result after AllGatherV!", fl_ctx)

                if horizontal:
                    self.aggr_result_to_send = self._histogram_sum(fl_ctx)
                else:
                    self.aggr_result_to_send = self.aggr_result_dict

                # reset aggr_result_dict for next gather
                self.aggr_result_dict = None

        self.world_size = len(self.size_dict)
        reply.set_header(Constant.HEADER_KEY_WORLD_SIZE, self.world_size)
        reply.set_header(Constant.HEADER_KEY_SIZE_DICT, self.size_dict)

        if horizontal:
            length = self.aggr_result_to_send.size()
        else:
            length = len(self.aggr_result_to_send)

        self.info(fl_ctx, f"aggr_result_to_send {length}")
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=self.aggr_result_to_send, private=True, sticky=False)

    def _histogram_sum(self, fl_ctx: FLContext):

        result = None

        for rank, vector in self.aggr_result_dict.items():
            if not result:
                result = vector
            else:
                result = result + vector

        return result
