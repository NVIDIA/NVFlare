# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np

from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.np.constants import NPConstants

SERVER_MODEL_DIR = "models/server"
CLIENT_MODEL_DIR = "models/client"

if __name__ == "__main__":
    """
    This is the tool to generate the pre-trained models for demonstrating the cross-validation without training.
    """

    model_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    model_learnable = make_model_learnable(weights={NPConstants.NUMPY_KEY: model_data}, meta_props={})

    working_dir = os.getcwd()
    model_dir = os.path.join(working_dir, SERVER_MODEL_DIR)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "server_1.npy")
    np.save(model_path, model_learnable[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY])
    model_path = os.path.join(model_dir, "server_2.npy")
    np.save(model_path, model_learnable[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY])

    model_dir = os.path.join(working_dir, CLIENT_MODEL_DIR)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_save_path = os.path.join(model_dir, "best_numpy.npy")
    np.save(model_save_path, model_data)
