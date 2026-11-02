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

import numpy as np


def make_data(site_name):
    """Create reproducible, disjoint training and validation rows at this site."""
    site_index = int(site_name.split("-")[-1])
    rng = np.random.default_rng(100 + site_index)
    features = rng.normal(loc=0.2 * (site_index - 1.5), size=(96, 2))
    features = np.column_stack((features, np.ones(96)))  # last column is the bias
    targets = features @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.05, size=96)
    return features[:64], targets[:64], features[64:], targets[64:]
