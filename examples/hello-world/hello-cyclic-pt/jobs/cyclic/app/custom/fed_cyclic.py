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
import gc
import logging
import os
import random
from typing import List, Optional

import torch

from net import Net

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import (
    CURRENT_ROUND,
    DATA,
    MIN_RESPONSES,
    NUM_ROUNDS,
    START_ROUND,
    TARGET_SITES,
)
from nvflare.app_common.workflows.wf_comm.wf_spec import WF

update_model = FLModelUtils.update_model


# Fed Cyclic Weight Transfer Workflow


class RelayOrder:
    FIXED = "FIXED"
    RANDOM = "RANDOM"
    RANDOM_WITHOUT_SAME_IN_A_ROW = "RANDOM_WITHOUT_SAME_IN_A_ROW"


SUPPORTED_ORDERS = (RelayOrder.FIXED, RelayOrder.RANDOM, RelayOrder.RANDOM_WITHOUT_SAME_IN_A_ROW)


class FedCyclic(WF):
    def __init__(
        self,
        output_path: str,
        num_rounds: int = 5,
        start_round: int = 0,
        task_name="train",
        order: str = RelayOrder.FIXED,
    ):
        super(FedCyclic, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.output_path = output_path
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.task_name = task_name
        self.order = order
        self.last_site: Optional[str] = None
        self.last_model: Optional[FLModel] = None
        self.part_sites = None

        self.check_inputs()

    def run(self):

        self.last_model = self.init_model()

        # note: this one must be within run() method, not in the __init__() method
        # as some values are injected at runtime during run()
        self.part_sites = self.flare_comm.get_site_names()

        if len(self.part_sites) <= 1:
            raise ValueError(f"Not enough client sites. sites={self.part_sites}")

        start = self.start_round
        end = self.start_round + self.num_rounds
        for current_round in range(start, end):
            targets = self.get_relay_orders()
            relay_result = self.relay_and_wait(self.last_model, targets, current_round)

            self.logger.info(f"target sites ={targets}.")

            task_name, task_result = next(iter(relay_result.items()))
            self.last_site, self.last_model = next(iter(task_result.items()))

            self.logger.info(f"ending current round={current_round}.")
            gc.collect()

        self.save_model(self.last_model, self.output_path)
        self.logger.info("\n fed cyclic ended \n")

    def relay_and_wait(self, last_model: FLModel, targets: List[str], current_round):
        msg_payload = {
            MIN_RESPONSES: 1,
            CURRENT_ROUND: current_round,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: self.start_round,
            DATA: last_model,
            TARGET_SITES: targets,
        }
        # (2) relay_and_wait and wait
        results = self.flare_comm.relay_and_wait(msg_payload)
        return results

    def init_model(self):
        net = Net()
        model = FLModel(params=net.state_dict(), params_type=ParamsType.FULL)
        return model

    def check_inputs(self):
        if not isinstance(self.num_rounds, int):
            raise TypeError("num_rounds must be int but got {}".format(type(self.num_rounds)))
        if not isinstance(self.task_name, str):
            raise TypeError("task_name must be a string but got {}".format(type(self.task_name)))
        if self.order not in SUPPORTED_ORDERS:
            raise ValueError(f"order must be in {SUPPORTED_ORDERS}")

    def get_relay_orders(self):
        targets = list(self.part_sites)
        if len(targets) <= 1:
            raise ValueError("Not enough client sites.")

        if self.order == RelayOrder.RANDOM:
            random.shuffle(targets)
        elif self.order == RelayOrder.RANDOM_WITHOUT_SAME_IN_A_ROW:
            random.shuffle(targets)
            if self.last_site == targets[0]:
                targets = targets.append(targets.pop(0))
        self.last_site = targets[-1]
        return targets

    def save_model(self, model: FLModel, file_path: str):
        if not file_path:
            raise ValueError("invalid file path")

        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        self.logger.info(f"save best model to {file_path} \n")
        torch.save(model.params, file_path)
