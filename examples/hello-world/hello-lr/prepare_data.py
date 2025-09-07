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

from pathlib import Path

import numpy as np
from flamby.datasets.fed_heart_disease import FedHeartDisease
from torch.utils.data import DataLoader as dl

np.NaN = np.nan  # Patch for NumPy 2.0 compatibility


def prepare_data(data_dir: Path):
    """Convert the dataset to numpy files."""
    print(f"Converting data in {data_dir} to numpy files")

    for site in range(4):
        for flag in ("train", "test"):
            data = FedHeartDisease(center=site, train=(flag == "train"), data_path=data_dir)

            data_x, data_y = [], []
            for x, y in dl(data, batch_size=1, shuffle=False, num_workers=0):
                data_x.append(x.cpu().numpy().reshape(-1))
                data_y.append(y.cpu().numpy().reshape(-1))

            data_x = np.array(data_x).reshape(-1, 13)
            data_y = np.array(data_y).reshape(-1, 1)

            print(f"site {site} - {flag} - variables shape: {data_x.shape}")
            print(f"site {site} - {flag} - outcomes shape: {data_y.shape}")

            save_x_path = data_dir / f"site-{site + 1}.{flag}.x.npy"
            print(f"Saving data: {save_x_path}")
            np.save(str(save_x_path), data_x)

            save_y_path = data_dir / f"site-{site + 1}.{flag}.y.npy"
            print(f"Saving data: {save_y_path}")
            np.save(str(save_y_path), data_y)


def main():
    DATA_DIR = Path("/tmp/flare/dataset/heart_disease_data")
    prepare_data(DATA_DIR)


if __name__ == "__main__":
    main()
