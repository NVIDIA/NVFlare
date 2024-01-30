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
import os

import torch
from fedavg import FedAvg

from nvflare.app_common.abstract.fl_model import FLModel


class PTFedAvg(FedAvg):
    def __init__(
        self,
        min_clients: int,
        num_rounds: int,
        output_path: str,
        start_round: int = 1,
        stop_cond: str = None,
    ):
        super().__init__(min_clients, num_rounds, output_path, start_round, stop_cond)

    def save_model(self, model: FLModel, file_path: str):
        if not file_path:
            raise ValueError("invalid file path")

        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        self.logger.info(f"save best model to {file_path} \n")
        torch.save(model.params, file_path)
