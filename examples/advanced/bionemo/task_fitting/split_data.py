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

import json
import os
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.stats import dirichlet


def list_to_dataframe(data_list):
    data_dict = {}
    for p in data_list:
        for k, v in p.items():
            if k not in data_dict:
                data_dict[k] = []
            else:
                data_dict[k].append(v)

    return pd.DataFrame(data_dict)


def get_site_class_summary(train_labels, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_labels[data_idx], return_counts=True)
        tmp = {unq[i]: int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[f"site-{site + 1}"] = tmp
    return class_sum


def partition_data(train_labels, label_names, num_sites, alpha, seed):
    min_size = 0
    N = len(train_labels)
    site_idx = {}
    train_labels = np.asarray(train_labels)

    # split
    while min_size < 10:
        idx_batch = [[] for _ in range(num_sites)]
        # for each class in the dataset
        for k in label_names:
            idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = dirichlet.rvs(np.repeat(alpha, num_sites), random_state=seed)
            # Balance
            proportions = np.array([p * (len(idx_j) < N / num_sites) for p, idx_j in zip(proportions, idx_batch)])

            # Fix for "invalid value encountered in divide"
            proportions_sum = proportions.sum()
            if proportions_sum > 0:
                proportions = proportions / proportions_sum
            else:
                proportions = np.ones_like(proportions) / len(proportions)

            # Fix for "invalid value encountered in cast"
            cumsum = np.cumsum(proportions) * len(idx_k)
            proportions = np.where(np.isnan(cumsum), 0, cumsum.astype(int))[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # shuffle
    for j in range(num_sites):
        np.random.shuffle(idx_batch[j])
        site_idx[j] = idx_batch[j]

    # collect class summary
    class_sum = get_site_class_summary(train_labels, site_idx)

    return site_idx, class_sum


def split(proteins, num_sites, split_dir=".", alpha=1.0, seed=0, concat=False):
    train_proteins = []
    train_labels = []
    test_proteins = []
    for idx, entry in enumerate(proteins):
        if entry["SET"] == "train":
            train_proteins.append(entry)
            train_labels.append(entry["TARGET"])
        elif entry["SET"] == "test":
            test_proteins.append(entry)
    assert len(train_labels) > 0
    label_names = set(train_labels)
    print(
        f"Partition protein dataset with {len(label_names)} classes into {num_sites} sites with Dirichlet sampling under alpha {alpha}"
    )
    site_idx, class_sum = partition_data(train_labels, label_names, num_sites, alpha, seed)
    pprint(class_sum)

    # write summary info
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    sum_file_name = os.path.join(split_dir, "summary.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write(f"Number of clients: {num_sites} \n")
        sum_file.write(f"Dirichlet sampling parameter: {alpha} \n")
        sum_file.write("Class counts for each client: \n")
        sum_file.write(json.dumps(class_sum))

    # write split data
    train_proteins = np.asarray(train_proteins)
    for site in range(num_sites):
        client_name = f"site-{site + 1}"

        train_indices = site_idx[site]
        split_train_proteins = train_proteins[train_indices]

        df_split_train_proteins = list_to_dataframe(split_train_proteins)
        df_test_proteins = list_to_dataframe(test_proteins)

        if concat:
            split_df = pd.concat([df_split_train_proteins, df_test_proteins])
            split_df.to_csv(
                os.path.join(split_dir, f"data_{client_name}.csv"),
                index=False,
                columns=["id", "sequence", "TARGET", "SET"],
            )
        else:
            _split_dir = os.path.join(split_dir, "train")
            if not os.path.isdir(_split_dir):
                os.makedirs(_split_dir)
            df_split_train_proteins.to_csv(
                os.path.join(_split_dir, f"data_train_{client_name}.csv"),
                index=False,
                columns=["id", "sequence", "TARGET", "SET"],
            )
            _split_dir = os.path.join(split_dir, "val")
            if not os.path.isdir(_split_dir):
                os.makedirs(_split_dir)
            df_test_proteins.to_csv(
                os.path.join(_split_dir, f"data_val_{client_name}.csv"),
                index=False,
                columns=["id", "sequence", "TARGET", "SET"],
            )
            # validation & test are the same here!
            _split_dir = os.path.join(split_dir, "test")
            if not os.path.isdir(_split_dir):
                os.makedirs(_split_dir)
            df_test_proteins.to_csv(
                os.path.join(_split_dir, f"data_test_{client_name}.csv"),
                index=False,
                columns=["id", "sequence", "TARGET", "SET"],
            )

        print(
            f"Saved {len(df_split_train_proteins)} training and {len(test_proteins)} testing proteins for {client_name}, "
            f"({len(set(df_split_train_proteins['TARGET']))}/{len(set(df_test_proteins['TARGET']))}) unique train/test classes."
        )
