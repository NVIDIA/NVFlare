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

import logging
from typing import Dict

import tenseal as ts

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import (
    CURRENT_ROUND,
    DATA,
    MIN_RESPONSES,
    NUM_ROUNDS,
    START_ROUND,
)
from nvflare.app_common.workflows.wf_comm.wf_spec import WF

# Controller Workflow


class KM(WF):
    def __init__(self, min_clients: int):
        super(KM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_clients = min_clients
        self.num_rounds = 3

    def run(self):
        he_context, max_idx_results = self.distribute_he_context_collect_max_idx()
        global_res = self.aggr_max_idx(max_idx_results)
        enc_hist_results = self.distribute_max_idx_collect_enc_stats(global_res)
        hist_obs_global, hist_cen_global = self.aggr_he_hist(he_context, enc_hist_results)
        _ = self.distribute_global_hist(hist_obs_global, hist_cen_global)

    def distribute_he_context_collect_max_idx(self):
        self.logger.info("send kaplan-meier analysis command to all sites with HE context \n")

        context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
        context_serial = context.serialize(save_secret_key=True)
        # drop private key for server
        context.make_context_public()
        # payload data always needs to be wrapped into an FLModel
        model = FLModel(params={"he_context": context_serial}, params_type=ParamsType.FULL)

        msg_payload = {
            MIN_RESPONSES: self.min_clients,
            CURRENT_ROUND: 1,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: 1,
            DATA: model,
        }

        results = self.flare_comm.broadcast_and_wait(msg_payload)
        return context, results

    def aggr_max_idx(self, sag_result: Dict[str, Dict[str, FLModel]]):
        self.logger.info("aggregate max histogram index \n")

        if not sag_result:
            raise RuntimeError("input is None or empty")

        task_name, task_result = next(iter(sag_result.items()))

        if not task_result:
            raise RuntimeError("task_result None or empty ")

        max_idx_global = []
        for site, fl_model in task_result.items():
            max_idx = fl_model.params["max_idx"]
            print(max_idx)
            max_idx_global.append(max_idx)
        # actual time point as index, so plus 1 for storage
        return max(max_idx_global) + 1

    def distribute_max_idx_collect_enc_stats(self, result: int):
        self.logger.info("send global max_index to all sites \n")

        model = FLModel(params={"max_idx_global": result}, params_type=ParamsType.FULL)

        msg_payload = {
            MIN_RESPONSES: self.min_clients,
            CURRENT_ROUND: 2,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: 1,
            DATA: model,
        }

        results = self.flare_comm.broadcast_and_wait(msg_payload)
        return results

    def aggr_he_hist(self, he_context, sag_result: Dict[str, Dict[str, FLModel]]):
        self.logger.info("aggregate histogram within HE \n")

        if not sag_result:
            raise RuntimeError("input is None or empty")

        task_name, task_result = next(iter(sag_result.items()))

        if not task_result:
            raise RuntimeError("task_result None or empty ")

        hist_obs_global = None
        hist_cen_global = None
        for site, fl_model in task_result.items():
            hist_obs_he_serial = fl_model.params["hist_obs"]
            hist_obs_he = ts.bfv_vector_from(he_context, hist_obs_he_serial)
            hist_cen_he_serial = fl_model.params["hist_cen"]
            hist_cen_he = ts.bfv_vector_from(he_context, hist_cen_he_serial)

            if not hist_obs_global:
                print(f"assign global hist with result from {site}")
                hist_obs_global = hist_obs_he
            else:
                print(f"add to global hist with result from {site}")
                hist_obs_global += hist_obs_he

            if not hist_cen_global:
                print(f"assign global hist with result from {site}")
                hist_cen_global = hist_cen_he
            else:
                print(f"add to global hist with result from {site}")
                hist_cen_global += hist_cen_he

        # return the two accumulated vectors, serialized for transmission
        hist_obs_global_serial = hist_obs_global.serialize()
        hist_cen_global_serial = hist_cen_global.serialize()
        return hist_obs_global_serial, hist_cen_global_serial

    def distribute_global_hist(self, hist_obs_global_serial, hist_cen_global_serial):
        self.logger.info("send global accumulated histograms within HE to all sites \n")

        model = FLModel(
            params={"hist_obs_global": hist_obs_global_serial, "hist_cen_global": hist_cen_global_serial},
            params_type=ParamsType.FULL,
        )

        msg_payload = {
            MIN_RESPONSES: self.min_clients,
            CURRENT_ROUND: 3,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: 1,
            DATA: model,
        }

        results = self.flare_comm.broadcast_and_wait(msg_payload)
        return results
