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
from typing import List

from nvflare.app_common.psi.psi_spec import PSI
import random


class SamplePSI(PSI):

    def __init__(self, psi_writer_id: str):
        super().__init__(psi_writer_id)
        self.data = {}
        self.site_nums = 10
        scale = 1000
        for i in range(self.site_nums):
            start = int(random.randrange(1, 2) * scale)
            end = int(random.randrange(2, 4) * scale)
            step = i + 1
            self.data[f"site-{i + 1}"] = range(start, end, step)

    def load_items(self) -> List[str]:
        site = self.fl_ctx.get_identity_name()
        user_id_range = self.data[site]
        return [f"user_id-{i}" for i in user_id_range]
