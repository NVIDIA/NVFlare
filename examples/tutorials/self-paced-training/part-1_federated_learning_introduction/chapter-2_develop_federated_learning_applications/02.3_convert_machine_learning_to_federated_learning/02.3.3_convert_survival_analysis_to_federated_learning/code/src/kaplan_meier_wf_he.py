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

import base64
import logging
from typing import Dict

import tenseal as ts

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.workflows.model_controller import ModelController

# Controller Workflow


class KM_HE(ModelController):
    def __init__(self, min_clients: int, he_context_path: str):
        super(KM_HE, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_clients = min_clients
        self.he_context_path = he_context_path
        self.num_rounds = 3

    def run(self):
        max_idx_results = self.start_fl_collect_max_idx()
        global_res = self.aggr_max_idx(max_idx_results)
        enc_hist_results = self.distribute_max_idx_collect_enc_stats(global_res)
        hist_obs_global, hist_cen_global = self.aggr_he_hist(enc_hist_results)
        _ = self.distribute_global_hist(hist_obs_global, hist_cen_global)

    def read_data(self, file_name: str):
        with open(file_name, "rb") as f:
            data = f.read()
        return base64.b64decode(data)

    def start_fl_collect_max_idx(self):
        self.logger.info("send initial message to all sites to start FL \n")
        model = FLModel(params={}, start_round=1, current_round=1, total_rounds=self.num_rounds)

        results = self.send_model_and_wait(data=model)
        return results

    def aggr_max_idx(self, sag_result: Dict[str, Dict[str, FLModel]]):
        self.logger.info("aggregate max histogram index \n")

        if not sag_result:
            raise RuntimeError("input is None or empty")

        max_idx_global = []
        for fl_model in sag_result:
            max_idx = fl_model.params["max_idx"]
            max_idx_global.append(max_idx)
        # actual time point as index, so plus 1 for storage
        return max(max_idx_global) + 1

    def distribute_max_idx_collect_enc_stats(self, result: int):
        self.logger.info("send global max_index to all sites \n")

        model = FLModel(
            params={"max_idx_global": result},
            params_type=ParamsType.FULL,
            start_round=1,
            current_round=2,
            total_rounds=self.num_rounds,
        )

        results = self.send_model_and_wait(data=model)
        return results

    def aggr_he_hist(self, sag_result: Dict[str, Dict[str, FLModel]]):
        self.logger.info("aggregate histogram within HE \n")

        # Load HE context
        he_context_serial = self.read_data(self.he_context_path)
        he_context = ts.context_from(he_context_serial)

        if not sag_result:
            raise RuntimeError("input is None or empty")

        hist_obs_global = None
        hist_cen_global = None
        for fl_model in sag_result:
            site = fl_model.meta.get("client_name", None)
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
            start_round=1,
            current_round=3,
            total_rounds=self.num_rounds,
        )

        results = self.send_model_and_wait(data=model)
        return results
