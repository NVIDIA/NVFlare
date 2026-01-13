# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pandas as pd

from nvflare.app_common.psi.psi_spec import PSI


class LocalPSI(PSI):
    def __init__(
        self,
        data_root_dir: str = "/tmp/nvflare/psi/data",
        psi_writer_id: str = "psi_writer",
    ) -> None:
        super().__init__(psi_writer_id)
        self.data_root_dir = data_root_dir

    def load_items(self) -> list[str]:
        site = self.fl_ctx.get_identity_name()
        data_path = os.path.join(self.data_root_dir, site, "data.csv")

        if os.path.isfile(data_path):
            df = pd.read_csv(data_path)
        else:
            raise RuntimeError(f"invalid data path {data_path}")

        # important the PSI algorithms requires the items are unique
        # PSI requires unique, non-null string items
        items = df["email_address"].dropna().astype(str).drop_duplicates().tolist()
        return items
