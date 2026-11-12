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

# This Dirichlet sampling strategy for creating a heterogeneous partition is adopted
# from FedMA (https://github.com/IBM/FedMA).

# MIT License

# Copyright (c) 2020 International Business Machines

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
import os
import pickle
import time

import numpy as np
from filelock import FileLock, Timeout
from tensorflow.keras import datasets

from nvflare.utils.data.partition import dirichlet_label_partition

_PARTITIONER_ID = "nvflare.utils.data.partition.dirichlet_label_partition"


def cifar10_split(split_dir: str = None, num_sites: int = 8, alpha: float = 0.5, seed: int = 0):
    if split_dir is None:
        raise ValueError("You need to define a valid `split_dir` for splitting the data.")
    if not os.path.isabs(split_dir):
        raise ValueError("`split_dir` needs to be absolute path.")
    if alpha <= 0.0:
        raise ValueError(f"Alpha should be larger than 0.0 but was {alpha}!")

    train_idx_paths = []

    print(f"Partition CIFAR-10 dataset into {num_sites} sites with Dirichlet sampling under alpha {alpha}")
    site_idx, class_sum = _partition_data(num_sites, alpha, seed=seed)

    # write to files
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    sum_file_name = os.path.join(split_dir, "summary.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write(f"Number of clients: {num_sites}\n")
        sum_file.write(f"Dirichlet sampling parameter: {alpha}\n")
        sum_file.write(f"Seed: {seed}\n")
        sum_file.write(f"Partitioner: {_PARTITIONER_ID}\n")
        sum_file.write("Class counts for each client:\n")
        sum_file.write(json.dumps(class_sum))

    site_file_path = os.path.join(split_dir, "site-")
    for site in range(num_sites):
        site_file_name = site_file_path + str(site + 1) + ".npy"
        print(f"Save split index {site + 1} of {num_sites} to {site_file_name}")
        np.save(site_file_name, np.asarray(site_idx[site], dtype=np.int64))
        train_idx_paths.append(site_file_name)

    return train_idx_paths


def _partition_data(num_sites, alpha, seed=0):
    # only training label is needed for doing split
    (train_images, train_labels), (test_images, test_labels) = load_cifar10_with_retry()
    labels = np.asarray(train_labels).reshape(-1)
    result = dirichlet_label_partition(labels=labels, num_sites=num_sites, alpha=alpha, seed=seed)
    site_idx = {site: indices.tolist() for site, indices in result.site_indices.items()}
    return site_idx, result.class_counts


def load_cifar10_with_retry(max_retries=3, retry_delay=5):
    """
    Load CIFAR10 dataset with retry mechanism and proper error handling.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (train_images, train_labels), (test_images, test_labels)
    """
    lock_path = os.path.join("/tmp", "cifar10_download.lock")
    lock = FileLock(lock_path)

    for attempt in range(max_retries):
        try:
            with lock:
                # Clear any existing corrupted downloads
                if attempt > 0:
                    cache_dir = os.path.expanduser("~/.keras/datasets")
                    cifar10_path = os.path.join(cache_dir, "cifar-10-batches-py")
                    if os.path.exists(cifar10_path):
                        import shutil

                        shutil.rmtree(cifar10_path)

                # Load the dataset
                return datasets.cifar10.load_data()

        except (Timeout, pickle.UnpicklingError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to load CIFAR10 dataset after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    raise RuntimeError("Failed to load CIFAR10 dataset")
