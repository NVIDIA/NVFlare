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
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.xgboost.histogram_based_v2.aggr import Aggregator
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.mock_he.adder import Adder
from nvflare.app_opt.xgboost.histogram_based_v2.mock_he.decrypter import Decrypter
from nvflare.app_opt.xgboost.histogram_based_v2.mock_he.encryptor import Encryptor
from nvflare.app_opt.xgboost.histogram_based_v2.mock_he.util import (
    combine,
    decode_encrypted_data,
    decode_feature_aggregations,
    encode_encrypted_data,
    encode_feature_aggregations,
    generate_keys,
    split,
)
from nvflare.app_opt.xgboost.histogram_based_v2.sec.data_converter import FeatureAggregationResult
from nvflare.app_opt.xgboost.histogram_based_v2.sec.processor_data_converter import ProcessorDataConverter
from nvflare.app_opt.xgboost.histogram_based_v2.sec.sec_handler import SecurityHandler


class ClientSecurityHandler(SecurityHandler):
    def __init__(self, key_length=1024, num_workers=10):
        FLComponent.__init__(self)
        self.num_workers = num_workers
        self.key_length = key_length
        self.public_key = None
        self.private_key = None
        self.encryptor = None
        self.adder = None
        self.decrypter = None
        self.data_converter = ProcessorDataConverter()
        self.encrypted_ghs = None
        self.clear_ghs = None  # for label client: list of tuples (g, h)
        self.original_gh_buffer = None
        self.feature_masks = None
        self.aggregator = Aggregator()
        self.aggr_result = None  # for label client: computed aggr result based on clear-text clear_ghs

    def _process_before_broadcast(self, fl_ctx: FLContext):
        root = fl_ctx.get_prop(Constant.PARAM_KEY_ROOT)
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        self.info(fl_ctx, "start")
        if root != rank:
            # I am not the source of the broadcast
            self.info(fl_ctx, "not root - ignore")
            return

        buffer = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        clear_ghs = self.data_converter.decode_gh_pairs(buffer, fl_ctx)
        if clear_ghs is None:
            # the buffer does not contain (g, h) pairs
            self.info(fl_ctx, "no clear gh pairs - ignore")
            return

        self.info(fl_ctx, f"got gh {len(clear_ghs)} pairs; original buf len: {len(buffer)}")
        self.original_gh_buffer = buffer

        # encrypt clear-text gh pairs and send to server
        self.clear_ghs = [combine(clear_ghs[i][0], clear_ghs[i][1]) for i in range(len(clear_ghs))]
        t = time.time()
        encrypted_values = self.encryptor.encrypt(self.clear_ghs)
        self.info(fl_ctx, f"encrypted gh pairs: {len(encrypted_values)}, took {time.time() - t} secs")

        t = time.time()
        encoded = encode_encrypted_data(self.public_key, encrypted_values)
        self.info(fl_ctx, f"encoded msg: size={len(encoded)}, type={type(encoded)} time={time.time()-t} secs")

        # Remember the original buffer size, so we could send a dummy buffer of this size to other clients
        # This is important since all XGB clients already prepared a buffer of this size and expect the data
        # to be this size.
        headers = {Constant.HEADER_KEY_ENCRYPTED_DATA: True, Constant.HEADER_KEY_ORIGINAL_BUF_SIZE: len(buffer)}
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=encoded, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_HEADERS, value=headers, private=True, sticky=False)

    def _process_after_broadcast(self, fl_ctx: FLContext):
        # this is called when the bcst result is received from the server
        self.info(fl_ctx, "start")
        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)

        has_encrypted_data = reply.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        if not has_encrypted_data:
            self.info(fl_ctx, f"{has_encrypted_data=}")
            return

        if self.clear_ghs:
            # this is the root rank
            # TBD: assume MPI requires the original buffer to be sent back to it.
            fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=self.original_gh_buffer, private=True, sticky=False)
            self.info(fl_ctx, "has_encrypted_data: label client - send original buffer back to XGB")
            return

        # this is a receiving non-label client
        # the rcv_buf contains encrypted gh values
        encoded_gh_str = fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)
        self.info(fl_ctx, f"{len(encoded_gh_str)=} {type(encoded_gh_str)=}")
        self.public_key, self.encrypted_ghs = decode_encrypted_data(encoded_gh_str)

        original_buf_size = reply.get_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE)
        self.info(fl_ctx, f"{original_buf_size=}; encrypted gh pairs: {len(self.encrypted_ghs)}")

        # send a dummy buffer of original size to the XGB client since it is expecting data to be this size
        dummy_buf = os.urandom(original_buf_size)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=dummy_buf, private=True, sticky=False)

    def _process_before_all_gather_v(self, fl_ctx: FLContext):
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        self.info(fl_ctx, "start")
        buffer = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        aggr_ctx = self.data_converter.decode_aggregation_context(buffer, fl_ctx)

        if not aggr_ctx:
            # this AllGatherV is irrelevant to secure processing
            self.info(fl_ctx, "no aggr ctx - ignore")
            return

        if not self.feature_masks:
            # the feature contexts only need to be set once
            if not aggr_ctx.features and not self.clear_ghs:
                return self._abort("missing features in aggregation context from non-label client", fl_ctx)
            m = []
            if aggr_ctx.features:
                for f in aggr_ctx.features:
                    m.append((f.feature_id, f.sample_bin_assignment, f.num_bins))
            self.feature_masks = m
            self.info(fl_ctx, f"got feature ctx: {len(m)}")

        # compute aggregation
        groups = []
        if aggr_ctx.sample_groups:
            for gid, sample_ids in aggr_ctx.sample_groups.items():
                groups.append((gid, sample_ids))

        if not self.encrypted_ghs:
            if not self.clear_ghs:
                # this is non-label client
                return self._abort(f"no encrypted (g, h) values for aggregation in rank {rank}", fl_ctx)
            else:
                # label client - send a dummy of 4 bytes
                self.info(fl_ctx, "label client: _do_aggregation in clear text")
                self._do_aggregation(groups, fl_ctx)
                headers = {Constant.HEADER_KEY_ENCRYPTED_DATA: True, Constant.HEADER_KEY_ORIGINAL_BUF_SIZE: len(buffer)}
                fl_ctx.set_prop(key=Constant.PARAM_KEY_HEADERS, value=headers, private=True, sticky=False)
                fl_ctx.set_prop(
                    key=Constant.PARAM_KEY_SEND_BUF,
                    value=None,
                    private=True,
                    sticky=False,
                )
            return

        self.info(
            fl_ctx, f"_process_before_all_gather_v: non-label client - do encrypted aggr for {len(groups)} groups"
        )
        start = time.time()
        aggr_result = self.adder.add(self.encrypted_ghs, self.feature_masks, groups, encode_sum=True)
        self.info(fl_ctx, f"got aggr result for {len(aggr_result)} features in {time.time()-start} secs")
        start = time.time()
        encoded_str = encode_feature_aggregations(aggr_result)
        self.info(fl_ctx, f"encoded aggr result len {len(encoded_str)} in {time.time()-start} secs")
        headers = {Constant.HEADER_KEY_ENCRYPTED_DATA: True, Constant.HEADER_KEY_ORIGINAL_BUF_SIZE: len(buffer)}
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=encoded_str, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_HEADERS, value=headers, private=True, sticky=False)

    def _do_aggregation(self, groups, fl_ctx: FLContext):
        # this is only for the label-client to compute aggregation in clear-text!
        if not self.feature_masks:
            return

        t = time.time()
        aggr_result = []  # list of (fid, gid, GH_list)
        for fm in self.feature_masks:
            fid, masks, num_bins = fm
            if not groups:
                gid = 0
                GH_list = self.aggregator.aggregate(self.clear_ghs, masks, num_bins, None)
                aggr_result.append((fid, gid, GH_list))
            else:
                for grp in groups:
                    gid, sample_ids = grp
                    GH_list = self.aggregator.aggregate(self.clear_ghs, masks, num_bins, sample_ids)
                    aggr_result.append((fid, gid, GH_list))
        self.info(fl_ctx, f"aggregated clear-text in {time.time()-t} secs")
        self.aggr_result = aggr_result

    def _decrypt_aggr_result(self, encoded, fl_ctx: FLContext):
        # decrypt aggr result from a client
        if not isinstance(encoded, str):
            # this is dummy result of the label-client
            return encoded

        encoded_str = encoded
        t = time.time()
        decoded_aggrs = decode_feature_aggregations(self.public_key, encoded_str)
        self.info(fl_ctx, f"decode_feature_aggregations took {time.time()-t} secs")

        t = time.time()
        aggrs_to_decrypt = [decoded_aggrs[i][2] for i in range(len(decoded_aggrs))]
        decrypted_aggrs = self.decrypter.decrypt(aggrs_to_decrypt)  # this is a list of clear-text GH numbers
        self.info(fl_ctx, f"decrypted {len(aggrs_to_decrypt)} numbers in {time.time()-t} secs")

        aggr_result = []
        for i in range(len(decoded_aggrs)):
            fid, gid, _ = decoded_aggrs[i]
            clear_aggr = decrypted_aggrs[i]  # list of combined clear-text ints
            aggr_result.append((fid, gid, clear_aggr))
        return aggr_result

    def _process_after_all_gather_v(self, fl_ctx: FLContext):
        # called after AllGatherV result is received from the server
        self.info(fl_ctx, "start")
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)
        encrypted_data = reply.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        if not encrypted_data:
            self.info(fl_ctx, "no encrypted result - ignore")
            return

        rcv_buf = fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)

        # this rcv_buf is a list of replies from ALL clients!
        if not isinstance(rcv_buf, dict):
            return self._abort(f"rank {rank}: expect a dict of aggr result but got {type(rcv_buf)}", fl_ctx)
        rank_replies = rcv_buf
        self.info(fl_ctx, f"received rank replies: {len(rank_replies)}")

        if not self.clear_ghs:
            # this is non-label client - don't care about the results
            dummy = os.urandom(Constant.DUMMY_BUFFER_SIZE)
            fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=dummy, private=True, sticky=False)
            self.info(fl_ctx, "non-label client: return dummy buffer back to XGB")
            return

        # this is label client: rank_replies contain encrypted aggr result!
        for r, rr in rank_replies.items():
            if r != rank:
                # this is aggr result of a non-label client
                rank_replies[r] = self._decrypt_aggr_result(rr, fl_ctx)

        # add label client's result
        rank_replies[rank] = self.aggr_result

        combined_result = {}  # gid => dict[fid=>GH_list]
        for r, rr in rank_replies.items():
            # rr is a list of tuples: fid, gid, GHList
            if not rr:
                # label client may not have any features.
                continue

            for a in rr:
                fid, gid, combined_numbers = a
                GH_list = []
                for n in combined_numbers:
                    GH_list.append(split(n))
                grp_result = combined_result.get(gid)
                if not grp_result:
                    grp_result = {}
                    combined_result[gid] = grp_result
                grp_result[fid] = FeatureAggregationResult(fid, GH_list)
                self.info(fl_ctx, f"aggr from rank {r}: {fid=} {gid=} bins={len(GH_list)}")

        final_result = {}
        for gid, far in combined_result.items():
            sorted_far = sorted(far.items())

            # r is a tuple of (fid, FeatureAggregationResult)
            final_result[gid] = [r[1] for r in sorted_far]
            fid_list = [x.feature_id for x in final_result[gid]]
            self.info(fl_ctx, f"final aggr: {gid=} features={fid_list}")

        result = self.data_converter.encode_aggregation_result(final_result, fl_ctx)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=result, private=True, sticky=False)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.public_key, self.private_key = generate_keys(self.key_length)
            self.encryptor = Encryptor(self.public_key, self.num_workers)
            self.decrypter = Decrypter(self.private_key, self.num_workers)
            self.adder = Adder(self.num_workers)
        else:
            super().handle_event(event_type, fl_ctx)
