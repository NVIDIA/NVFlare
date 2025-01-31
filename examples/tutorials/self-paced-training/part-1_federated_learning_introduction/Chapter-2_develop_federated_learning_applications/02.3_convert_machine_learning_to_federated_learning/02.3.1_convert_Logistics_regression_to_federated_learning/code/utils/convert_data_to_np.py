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


import argparse
import os

import numpy as np
from flamby.datasets.fed_heart_disease import FedHeartDisease
from torch.utils.data import DataLoader as dl

if __name__ == "__main__":

    parser = argparse.ArgumentParser("save UCI Heart Disease as numpy arrays.")
    parser.add_argument("save_dir", type=str, help="directory to save converted numpy arrays as .npy files.")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for site in range(4):

        for flag in ("train", "test"):

            # To load data a pytorch dataset
            data = FedHeartDisease(center=site, train=(flag == "train"))

            # Save training dataset
            data_x = []
            data_y = []
            for x, y in dl(data, batch_size=1, shuffle=False, num_workers=0):
                data_x.append(x.cpu().numpy().reshape(-1))
                data_y.append(y.cpu().numpy().reshape(-1))

            data_x = np.array(data_x).reshape(-1, 13)
            data_y = np.array(data_y).reshape(-1, 1)

            print("site {} - {} - variables shape: {}".format(site, flag, data_x.shape))
            print("site {} - {} - outcomes shape: {}".format(site, flag, data_y.shape))

            save_x_path = "{}/site-{}.{}.x.npy".format(args.save_dir, site + 1, flag)
            print("saving data: {}".format(save_x_path))
            np.save(save_x_path, data_x)

            save_y_path = "{}/site-{}.{}.y.npy".format(args.save_dir, site + 1, flag)
            print("saving data: {}".format(save_y_path))
            np.save(save_y_path, data_y)
