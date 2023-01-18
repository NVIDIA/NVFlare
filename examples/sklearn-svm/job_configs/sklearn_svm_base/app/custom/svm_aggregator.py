# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

# We will move to this app_common when it gets matured
from typing import Dict

import numpy as np
from app_common.aggregators.assembler import Assembler
from nvflare.apis.dxo import DataKind
from sklearn.svm import SVC


class SVMAggregator(Assembler):
    def __init__(self):
        super().__init__(data_kind=DataKind.WEIGHTS)
        # Record the global support vectors
        # so that only 1 round of training is performed
        self.support_x = None
        self.support_y = None

    def get_model_params(self, data: dict):
        return {"support_x": data["support_x"], "support_y": data["support_y"]}

    def aggregate(self, current_round: int, data: Dict[str, dict]) -> dict:
        if current_round == 0:
            # Fist round, collect all support vectors and
            # perform one round of SVM to produce global model
            support_x = []
            support_y = []
            for client in self.accumulator:
                client_model = self.accumulator[client]
                support_x.append(client_model["support_x"])
                support_y.append(client_model["support_y"])
            global_x = np.concatenate(support_x)
            global_y = np.concatenate(support_y)
            svm_global = SVC(kernel="rbf")
            svm_global.fit(global_x, global_y)

            index = svm_global.support_
            self.support_x = global_x[index]
            self.support_y = global_y[index]
        # The following round directly returns the retained record
        # No further training
        params = {"support_x": self.support_x, "support_y": self.support_y}
        return params

    def reset(self) -> None:
        # Reset accumulator for next round,
        # # but not the center and count, which will be used as the starting point of the next round
        self.accumulator = {}
