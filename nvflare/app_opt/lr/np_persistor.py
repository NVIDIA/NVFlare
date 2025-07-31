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


import numpy as np

from nvflare.app_common.np.np_model_persistor import NPModelPersistor


class NewtonRaphsonModelPersistor(NPModelPersistor):
    """
    This class defines the persistor for Newton Raphson model.

    A persistor controls the logic behind initializing, loading
    and saving of the model / parameters for each round of a
    federated learning process.

    In the 2nd order Newton Raphson case, a model is just a
    1-D numpy vector containing the parameters for logistic
    regression. The length of the parameter vector is defined
    by the number of features in the dataset.

    """

    def __init__(self, model_dir="models", model_name="weights.npy", n_features=13):
        """
        Init function for NewtonRaphsonModelPersistor.

        Args:
            model_dir: sub-folder name to save and load the global model
                between rounds.
            model_name: name to save and load the global model.
            n_features: number of features for the logistic regression.
                For the UCI ML heart Disease dataset, this is 13.

        """

        super().__init__()

        self.model_dir = model_dir
        self.model_name = model_name
        self.n_features = n_features

        # A default model is loaded when no local model is available.
        # This happen when training starts.
        #
        # A `model` for a binary logistic regression is just a matrix,
        # with shape (n_features + 1, 1).
        # For the UCI ML Heart Disease dataset, the n_features = 13.
        #
        # A default matrix with value 0s is created.
        #
        self.default_data = np.zeros((self.n_features + 1, 1), dtype=np.float32)
