# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import numpy as np
from sklearn.svm import SVC

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.app_constant import AppConstants


class SVMAssembler(Assembler):
    """Assembler for federated SVM using support vector aggregation.

    This assembler implements the aggregation logic for federated SVM training.
    The approach is to:
    1. Each client trains a local SVM on their data
    2. Each client sends their support vectors (and labels) to the server
    3. Server concatenates all support vectors from all clients
    4. Server trains a global SVM on the aggregated support vectors
    5. Server extracts final global support vectors and sends back to clients

    This approach only requires one round of training since SVM is not an
    iterative algorithm in the federated setting.

    Args:
        kernel: Kernel type to use in SVM. Options include 'linear', 'poly', 'rbf', 'sigmoid'.
            Default is 'rbf'.
    """

    def __init__(self, kernel: str = "rbf"):
        super().__init__(data_kind=DataKind.WEIGHTS)

        # Validate kernel parameter
        supported_kernels = ["linear", "poly", "rbf", "sigmoid"]
        if kernel not in supported_kernels:
            raise ValueError(f"Unsupported kernel '{kernel}'. Must be one of {supported_kernels}")

        # Record the global support vectors
        # so that only 1 round of training is performed
        self.support_x = None
        self.support_y = None
        self.kernel = kernel

    def get_model_params(self, dxo: DXO):
        """Extract support vectors from the DXO.

        Args:
            dxo: DXO containing model data with 'support_x' (support vectors)
                and 'support_y' (support vector labels)

        Returns:
            Dictionary with keys 'support_x' and 'support_y' containing
            the support vectors and their labels
        """
        data = dxo.data
        return {"support_x": data["support_x"], "support_y": data["support_y"]}

    def assemble(self, data: dict[str, dict], fl_ctx: FLContext) -> DXO:
        """Assemble the federated SVM model from client contributions.

        This method implements the core SVM aggregation logic:
        - Round 0: Collects support vectors from all clients, trains a global SVM
          on the concatenated support vectors, and extracts global support vectors
        - Round 1+: Returns the previously computed global support vectors

        Args:
            data: Dictionary mapping client names to their data (not used directly;
                self.collection contains the processed client models)
            fl_ctx: FLContext containing federated learning context information
                (e.g., current round number)

        Returns:
            DXO containing the global support vectors ('support_x' and 'support_y')

        Note:
            self.collection is populated by the parent Assembler class before this
            method is called. It contains the processed client models from get_model_params().
        """
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if current_round == 0:
            # First round, collect all support vectors from clients
            # Note: self.collection is populated by parent Assembler class
            support_x = []
            support_y = []
            for client in self.collection:
                client_model = self.collection[client]
                support_x.append(client_model["support_x"])
                support_y.append(client_model["support_y"])
            global_x = np.concatenate(support_x)
            global_y = np.concatenate(support_y)
            # perform one round of SVM to produce global model
            svm_global = SVC(kernel=self.kernel)
            svm_global.fit(global_x, global_y)
            # get global support vectors
            index = svm_global.support_
            self.support_x = global_x[index]
            self.support_y = global_y[index]
        params = {"support_x": self.support_x, "support_y": self.support_y}
        dxo = DXO(data_kind=self.expected_data_kind, data=params)
        return dxo
