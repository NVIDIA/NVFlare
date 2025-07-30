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

import os

import numpy as np

SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"


def _save_model(model_data, model_dir: str, model_file: str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_file)
    np.save(model_path, model_data)


if __name__ == "__main__":
    """
    This is the tool to generate the pre-trained models for demonstrating the cross-validation without training.
    """

    model_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    _save_model(model_data=model_data, model_dir=SERVER_MODEL_DIR, model_file="server_1.npy")
    _save_model(model_data=model_data, model_dir=SERVER_MODEL_DIR, model_file="server_2.npy")
    _save_model(model_data=model_data, model_dir=CLIENT_MODEL_DIR, model_file="best_numpy.npy")
