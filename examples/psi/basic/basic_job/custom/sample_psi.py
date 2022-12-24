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
        scale = 10
        for i in range(self.site_nums):
            start = int(random.randrange(1, 2) * scale)
            end = int(random.randrange(2, 4) * scale)
            step = i + 1
            self.data[f"site-{i + 1}"] = range(start, end, step)

    def load_items(self) -> List[str]:
        site = self.fl_ctx.get_identity_name()
        user_id_range = self.data[site]
        return [f"user_id-{i}" for i in user_id_range]


#
# site: site-9 data =  { 'PSI_ITEM_SIZE': 111112}
# site: site-8 data =  { 'PSI_ITEM_SIZE': 125000}
# site: site-6 data =  { 'PSI_ITEM_SIZE': 166667}
# site: site-10 data = { 'PSI_ITEM_SIZE': 200000}
# site: site-7 data =  { 'PSI_ITEM_SIZE': 285715}
# site: site-3 data =  { 'PSI_ITEM_SIZE': 333334}
# site: site-5 data =  { 'PSI_ITEM_SIZE': 400000}
# site: site-4 data =  { 'PSI_ITEM_SIZE': 500000}
# site: site-2 data =  { 'PSI_ITEM_SIZE': 1000000}
# site: site-1 data =  { 'PSI_ITEM_SIZE': 2000000}


# site: site-7 data =  { 'PSI_ITEM_SIZE': 285715}
# site: site-3 data =  { 'PSI_ITEM_SIZE': 333334}
# The following combo will cause openmined PSI to fail during setup
# client_items_size = 333334
# server_items_size = 397
