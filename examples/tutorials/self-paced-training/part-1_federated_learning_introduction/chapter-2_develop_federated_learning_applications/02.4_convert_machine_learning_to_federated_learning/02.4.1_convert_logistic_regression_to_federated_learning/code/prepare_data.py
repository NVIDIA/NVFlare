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

from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    output = Path("/tmp/flare/dataset/heart_disease_data")
    output.mkdir(parents=True, exist_ok=True)
    hospitals = ["cleveland", "hungarian", "switzerland", "va"]
    for site, hospital in enumerate(hospitals, 1):
        url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.{hospital}.data"
        with urlopen(url, timeout=60) as response:
            data = pd.read_csv(StringIO(response.read().decode()), header=None, na_values="?")
        x = data.iloc[:, :13].to_numpy()
        y = (data.iloc[:, 13].to_numpy() > 0).astype(float)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=7, stratify=y)
        imputer = SimpleImputer(keep_empty_features=True)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(train_x))
        test_x = scaler.transform(imputer.transform(test_x))
        for split, features, labels in [("train", train_x, train_y), ("test", test_x, test_y)]:
            np.save(output / f"site-{site}.{split}.x.npy", features)
            np.save(output / f"site-{site}.{split}.y.npy", labels.reshape(-1, 1))
        print(f"site-{site}: {hospital}, train={len(train_x)}, test={len(test_x)}")


if __name__ == "__main__":
    main()
