# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os.path
from typing import List

import numpy as np

from nvflare.app_common.psi.psi_spec import PSI


class Cifar10LocalPSI(PSI):
    def __init__(self, psi_writer_id: str, data_path: str = "/tmp/data.csv"):
        super().__init__(psi_writer_id)
        self.data_path = data_path
        self.data = {}

        if not os.path.isfile(self.data_path):
            raise RuntimeError(f"invalid data path {data_path}")

    def load_items(self) -> List[str]:
        _ext = os.path.splitext(self.data_path)[1]

        items = np.load(self.data_path)

        return [str(i) for i in items]
