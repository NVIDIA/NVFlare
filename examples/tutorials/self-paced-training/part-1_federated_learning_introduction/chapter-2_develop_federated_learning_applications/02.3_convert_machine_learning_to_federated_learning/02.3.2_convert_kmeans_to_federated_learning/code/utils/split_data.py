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

from enum import Enum
from typing import List

import numpy as np


class SplitMethod(Enum):
    UNIFORM = "uniform"
    LINEAR = "linear"
    SQUARE = "square"
    EXPONENTIAL = "exponential"


def get_split_ratios(site_num: int, split_method: SplitMethod):
    if split_method == SplitMethod.UNIFORM:
        ratio_vec = np.ones(site_num)
    elif split_method == SplitMethod.LINEAR:
        ratio_vec = np.linspace(1, site_num, num=site_num)
    elif split_method == SplitMethod.SQUARE:
        ratio_vec = np.square(np.linspace(1, site_num, num=site_num))
    elif split_method == SplitMethod.EXPONENTIAL:
        ratio_vec = np.exp(np.linspace(1, site_num, num=site_num))
    else:
        raise ValueError(f"Split method {split_method.name} not implemented!")

    return ratio_vec


def split_num_proportion(n, site_num, split_method: SplitMethod) -> List[int]:
    split = []
    ratio_vec = get_split_ratios(site_num, split_method)
    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def assign_data_index_to_sites(
    data_size: int,
    valid_fraction: float,
    num_sites: int,
    split_method: SplitMethod = SplitMethod.UNIFORM,
) -> dict:
    if valid_fraction > 1.0:
        raise ValueError("validation percent should be less than or equal to 100% of the total data")
    elif valid_fraction < 1.0:
        valid_size = int(round(data_size * valid_fraction, 0))
        train_size = data_size - valid_size
    else:
        valid_size = data_size
        train_size = data_size

    site_sizes = split_num_proportion(train_size, num_sites, split_method)
    split_data_indices = {
        "valid": {"start": 0, "end": valid_size},
    }
    for site in range(num_sites):
        site_id = site + 1
        if valid_fraction < 1.0:
            idx_start = valid_size + sum(site_sizes[:site])
            idx_end = valid_size + sum(site_sizes[: site + 1])
        else:
            idx_start = sum(site_sizes[:site])
            idx_end = sum(site_sizes[: site + 1])
        split_data_indices[site_id] = {"start": idx_start, "end": idx_end}

    return split_data_indices


def get_file_line_count(input_path: str) -> int:
    count = 0
    with open(input_path, "r") as fp:
        for i, _ in enumerate(fp):
            count += 1
    return count


def split_data(
    data_path: str,
    num_clients: int,
    valid_frac: float,
    split_method: SplitMethod = SplitMethod.UNIFORM,
):
    size_total_file = get_file_line_count(data_path)
    site_indices = assign_data_index_to_sites(size_total_file, valid_frac, num_clients, split_method)
    return site_indices
