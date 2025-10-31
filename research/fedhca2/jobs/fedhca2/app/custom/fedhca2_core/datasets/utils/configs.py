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

# Image scales for different datasets
TRAIN_SCALE = {
    'pascalcontext': (512, 512),
    'nyud': (480, 640),
}

TEST_SCALE = {
    'pascalcontext': (512, 512),
    'nyud': (480, 640),
}

# Number of training images
NUM_TRAIN_IMAGES = {
    'pascalcontext': 4998,
    'nyud': 795,
}


# Output channels for different tasks and datasets
def get_output_num(task, dataname):
    """Get number of output channels for task on dataset"""
    if dataname == 'pascalcontext':
        task_output = {
            'semseg': 21,
            'human_parts': 7,
            'normals': 3,
            'edge': 1,
            'sal': 2,
        }
    elif dataname == 'nyud':
        task_output = {
            'semseg': 40,
            'normals': 3,
            'edge': 1,
            'depth': 1,
        }
    else:
        raise NotImplementedError(f"Dataset {dataname} not supported")

    return task_output[task]
