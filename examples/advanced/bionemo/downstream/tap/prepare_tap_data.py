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

import os

import numpy as np
import pandas as pd
from tdc.single_pred import Develop
from tdc.utils import retrieve_label_name_list

np.random.seed(1234)

split_dir = "/tmp/data/tap"
n_clients = 4
do_break_chains = False
do_clean_chains = True
do_normalize = False
alpha = 1.0


def clean_chains(df):
    a = df["Antibody"]
    b = []
    for chains in a:
        # split chains
        chains = chains.replace("['", "").replace("']", "").replace("'\\n '", " ")
        assert "'" not in chains
        assert "[" not in chains
        assert "]" not in chains
        assert "\\n" not in chains
        assert "'" not in chains
        b.append(chains)
    df["Antibody"] = b

    return df


def break_chains(df):
    out_df = {"Antibody": []}
    for idx, row in df.iterrows():
        # split chains
        chains = row["Antibody"]
        chains = chains.replace("['", "").replace("']", "").split("'\\n '")
        assert "'" not in chains
        assert "[" not in chains
        assert "]" not in chains
        assert "\\n" not in chains
        assert "'" not in chains

        for chain in chains:
            out_df["Antibody"].append(chain)
            for k in row.keys():
                if k == "Antibody":
                    continue
                if k not in out_df:
                    out_df[k] = [row[k]]
                else:
                    out_df[k].append(row[k])

    return pd.DataFrame(out_df)


def main():
    seed = 0
    label_list = retrieve_label_name_list("TAP")
    train_df = None
    test_df = None

    for label_name in label_list:
        data = Develop(name="TAP", label_name=label_name)
        split = data.get_split()

        train_split = pd.concat([split["train"], split["valid"]])
        if train_df is None:
            train_df = train_split
            train_df = train_df.rename(columns={"Y": label_name})
        else:
            assert (train_df["Antibody_ID"] == train_split["Antibody_ID"]).all()
            train_df[label_name] = train_split["Y"]

        if test_df is None:
            test_df = pd.concat([test_df, split["test"]])
            test_df = test_df.rename(columns={"Y": label_name})
        else:
            assert (test_df["Antibody_ID"] == split["test"]["Antibody_ID"]).all()
            test_df[label_name] = split["test"]["Y"]

    if do_normalize:
        total_df = pd.concat([train_df, test_df])
        stats = {}
        for label_name in label_list:
            _mean = np.mean(total_df[label_name])
            _std = np.std(total_df[label_name])
            stats[label_name] = {"mean": _mean, "std": _std}

            # normalize
            total_df[label_name] = (total_df[label_name] - _mean) / _std
            train_df[label_name] = (train_df[label_name] - _mean) / _std
            test_df[label_name] = (test_df[label_name] - _mean) / _std
            print(
                f"  ... normalize {label_name} from mean+-std {_mean:.3f}+-{_std:.3f} "
                f"to train: {np.mean(train_df[label_name]):.3f}+-{np.std(train_df[label_name]):.3f}"
                f"to test: {np.mean(test_df[label_name]):.3f}+-{np.std(test_df[label_name]):.3f}"
                f"to total: {np.mean(total_df[label_name]):.3f}+-{np.std(total_df[label_name]):.3f}"
            )

    # split client train
    client_train_dfs = []
    if alpha > 0:
        print(f"Sampling with alpha={alpha}")
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
    else:
        print("Uniform sampling")
        proportions = n_clients * [1 / n_clients]

    for client_id in range(n_clients):
        client_name = f"site-{client_id + 1}"
        client_train_df = train_df.sample(frac=proportions[client_id], replace=False, random_state=seed + client_id)

        if do_break_chains:
            client_train_df = break_chains(client_train_df)
        if do_clean_chains:
            client_train_df = clean_chains(client_train_df)
        client_train_dfs.append(client_train_df)

        _split_dir = os.path.join(split_dir, "train")
        if not os.path.isdir(_split_dir):
            os.makedirs(_split_dir)
        client_train_df.to_csv(os.path.join(_split_dir, f"tap_{client_name}_train.csv"), index=False)
        print(f"Save {len(client_train_df)} training proteins for {client_name} (frac={proportions[client_id]:0.3f})")

    # save full train, test, & valid
    if do_break_chains:
        train_df = break_chains(train_df)
        test_df = break_chains(test_df)
    if do_clean_chains:
        train_df = clean_chains(train_df)
        test_df = clean_chains(test_df)

    _split_dir = os.path.join(split_dir, "train")
    if not os.path.isdir(_split_dir):
        os.makedirs(_split_dir)
    train_df.to_csv(os.path.join(_split_dir, "tap_full_train.csv"), index=False)
    _split_dir = os.path.join(split_dir, "val")
    if not os.path.isdir(_split_dir):
        os.makedirs(_split_dir)
    test_df.to_csv(os.path.join(_split_dir, "tap_valid.csv"), index=False)
    _split_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(_split_dir):
        os.makedirs(_split_dir)
    test_df.to_csv(os.path.join(_split_dir, "tap_test.csv"), index=False)

    print(f"Saved {len(train_df)} training and {len(test_df)} testing proteins.")

    # measure overlap
    d = np.nan * np.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            if j <= i:
                continue
            a = np.asarray(client_train_dfs[i]["Antibody_ID"])
            b = np.asarray(client_train_dfs[j]["Antibody_ID"])
            assert len(np.unique(a)) == len(a)
            assert len(np.unique(b)) == len(b)
            d[i][j] = len(np.intersect1d(a, b)) / len(b)

    print(d)
    overlap = np.mean(d[~np.isnan(d)])
    print(f"Avg. overlap: {100 * overlap:0.2f}%")


if __name__ == "__main__":
    main()
