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


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=/tmp/flare/dataset/heart_disease_data

# Install dependencies
#pip install wget
FLAMBY_INSTALL_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
# git clone https://github.com/owkin/FLamby.git && cd FLamby && pip install -e .

# Download data using FLamby
mkdir -p ${DATA_DIR}
python3 ${FLAMBY_INSTALL_DIR}/flamby/datasets/fed_heart_disease/dataset_creation_scripts/download.py --output-folder ${DATA_DIR}

# Convert data to numpy files
python3 ${SCRIPT_DIR}/utils/convert_data_to_np.py ${DATA_DIR}
