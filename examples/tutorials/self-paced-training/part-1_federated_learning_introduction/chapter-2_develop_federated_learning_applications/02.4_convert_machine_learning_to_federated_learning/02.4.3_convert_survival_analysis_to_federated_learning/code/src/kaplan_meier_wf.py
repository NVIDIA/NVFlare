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

import logging
from typing import Dict

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.workflows.model_controller import ModelController


# Controller Workflow
class KM(ModelController):
    def __init__(self, min_clients: int):
        super(KM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_clients = min_clients
        self.num_rounds = 2

    def run(self):
        hist_local = self.start_fl_collect_hist()
        hist_obs_global, hist_cen_global = self.aggr_hist(hist_local)
        _ = self.distribute_global_hist(hist_obs_global, hist_cen_global)

    def start_fl_collect_hist(self):
        self.logger.info("send initial message to all sites to start FL \n")
        model = FLModel(params={}, start_round=1, current_round=1, total_rounds=self.num_rounds)

        results = self.send_model_and_wait(data=model)
        return results

    def aggr_hist(self, sag_result: Dict[str, Dict[str, FLModel]]):
        self.logger.info("aggregate histogram \n")

        if not sag_result:
            raise RuntimeError("input is None or empty")

        hist_idx_max = 0
        for fl_model in sag_result:
            hist = fl_model.params["hist_obs"]
            if hist_idx_max < max(hist.keys()):
                hist_idx_max = max(hist.keys())
        hist_idx_max += 1

        hist_obs_global = {}
        hist_cen_global = {}
        for idx in range(hist_idx_max + 1):
            hist_obs_global[idx] = 0
            hist_cen_global[idx] = 0

        for fl_model in sag_result:
            hist_obs = fl_model.params["hist_obs"]
            hist_cen = fl_model.params["hist_cen"]
            for i in hist_obs.keys():
                hist_obs_global[i] += hist_obs[i]
            for i in hist_cen.keys():
                hist_cen_global[i] += hist_cen[i]

        return hist_obs_global, hist_cen_global

    def distribute_global_hist(self, hist_obs_global, hist_cen_global):
        self.logger.info("send global accumulated histograms within HE to all sites \n")

        model = FLModel(
            params={"hist_obs_global": hist_obs_global, "hist_cen_global": hist_cen_global},
            params_type=ParamsType.FULL,
            start_round=1,
            current_round=2,
            total_rounds=self.num_rounds,
        )

        results = self.send_model_and_wait(data=model)
        return results
