# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Load each site's prepared CIFAR-10 sample IDs for private set intersection."""

from pathlib import Path

import numpy as np

from nvflare.app_common.psi.psi_spec import PSI


class Cifar10LocalPSI(PSI):
    def __init__(self, split_dir: str, psi_writer_id: str = "psi_writer"):
        super().__init__(psi_writer_id)
        self.split_dir = split_dir

    def load_items(self) -> list[str]:
        site_name = self.fl_ctx.get_identity_name()
        data_path = Path(self.split_dir) / f"{site_name}.npy"
        if not data_path.is_file():
            raise FileNotFoundError(f"Prepared site indices not found at {data_path}")

        # DH-PSI accepts unique string items and keeps the raw sample IDs private.
        return [str(index) for index in np.load(data_path)]
