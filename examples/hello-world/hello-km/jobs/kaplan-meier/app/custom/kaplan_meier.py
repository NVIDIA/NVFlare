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

import json
import logging
import os.path
from typing import Dict

from km_analysis import kaplan_meier_analysis

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common import wf_comm as flare
from nvflare.app_common.wf_comm.wf_comm_api_spec import (
    CURRENT_ROUND,
    DATA,
    MIN_RESPONSES,
    NUM_ROUNDS,
    START_ROUND,
    WFCommAPISpec,
)

# Controller Workflow


class KM:
    def __init__(self, min_clients: int, output_path: str):
        super(KM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_path = output_path
        self.min_clients = min_clients
        self.num_rounds = 1
        self.flare_comm: WFCommAPISpec = flare.get_wf_comm_api()

    def run(self):
        results = self.start_km_analysis()
        global_res = self.aggr_km_result(results)
        self.save(global_res, self.output_path)

    def start_km_analysis(self):
        self.logger.info("send kaplan-meier analysis command to all sites \n")

        msg_payload = {
            MIN_RESPONSES: self.min_clients,
            CURRENT_ROUND: 1,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: 1,
            DATA: {},
        }
        results = self.flare_comm.broadcast_and_wait(msg_payload)
        return results

    def aggr_km_result(self, sag_result: Dict[str, Dict[str, FLModel]]):

        self.logger.info("aggregate kaplan-meier analysis results \n")

        if not sag_result:
            raise RuntimeError("input is None or empty")

        task_name, task_result = next(iter(sag_result.items()))

        if not task_result:
            raise RuntimeError("task_result None or empty ")

        global_result: dict = {}
        all_result = {}
        for site, fl_model in task_result.items():
            result = fl_model.params
            all_result[site] = result
            timelines = result.get("timeline")
            event_counts = result.get("event_count")
            combined_arrays = list(zip(timelines, event_counts))
            g_timelines = global_result.get("timeline", [])
            g_event_counts = global_result.get("event_count", {})
            for t, count in combined_arrays:
                if t not in g_timelines:
                    g_timelines.append(t)
                    g_event_counts[t] = count
                else:
                    prev_count = g_event_counts.get(t)
                    g_event_counts[t] = prev_count + count
            global_result["event_count"] = g_event_counts
            global_result["timeline"] = g_timelines

        g_duration = global_result.get("timeline", [])
        g_event_counts = list(global_result.get("event_count").values())

        g_km_result = kaplan_meier_analysis(g_duration, g_event_counts)

        all_result["global"] = g_km_result
        return all_result

    def save(self, result: dict, file_path: str):
        self.logger.info(f"save the result to {file_path} \n")

        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(result, json_file, indent=4)
