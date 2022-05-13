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

import os


def custom_client_datalist_json_path(datalist_json_path: str, client_id: str) -> str:
    """
    Customize datalist_json_path for each client
    Args:
         datalist_json_path: root path containing all jsons
         client_id: e.g., site-2
    """
    # Customize datalist_json_path for each client
    datalist_json_path_client = os.path.join(
        datalist_json_path,
        client_id + ".json",
    )
    return datalist_json_path_client
