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


from typing import Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel


class GenericTask(FLModel):
    def __init__(self, meta: Optional[Dict] = None):
        super().__init__(
            params_type=None,
            params={},
            optimizer_params=None,
            metrics=None,
            start_round=0,
            current_round=0,
            total_rounds=1,
            meta=meta,
        )
