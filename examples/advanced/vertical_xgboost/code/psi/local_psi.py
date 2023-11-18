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

import pandas as pd

from nvflare.app_common.psi.psi_spec import PSI


class LocalPSI(PSI):
    def __init__(self, psi_writer_id: str, data_split_path: str, id_col: str):
        super().__init__(psi_writer_id)
        self.data_split_path = data_split_path
        self.id_col = id_col
        self.data = {}

    def load_items(self) -> List[str]:
        client_id = self.fl_ctx.get_identity_name()
        client_data_split_path = self.data_split_path.replace("site-x", client_id)
        if os.path.isfile(client_data_split_path):
            df = pd.read_csv(client_data_split_path, header=0)
        else:
            raise RuntimeError(f"invalid data path {client_data_split_path}")

        # Note: the PSI algorithm requires the items are unique
        items = list(df[self.id_col])
        return items
