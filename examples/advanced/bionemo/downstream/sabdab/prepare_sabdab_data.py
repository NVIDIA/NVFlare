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

np.random.seed(1234)

out_name = "sabdab_chen"
split_dir = f"/tmp/data/{out_name}"
n_clients = 6
do_break_chains = False
do_clean_chains = True
do_normalize = False
alpha = 1.0


def clean_chains(df):
    a = df["Antibody"]
    b = []
    for chains in a:
        # split chains
        chains = chains.replace("['", "").replace("']", "").replace("', '", " ")
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

    data = Develop(name="SAbDab_Chen", path="/tmp/data")
    split = data.get_split()

    train_df = pd.concat([split["train"], split["valid"]])
    test_df = split["test"]

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
        client_train_df.to_csv(os.path.join(_split_dir, f"{out_name}_{client_name}_train.csv"), index=False)
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
    train_df.to_csv(os.path.join(_split_dir, f"{out_name}_full_train.csv"), index=False)
    _split_dir = os.path.join(split_dir, "val")
    if not os.path.isdir(_split_dir):
        os.makedirs(_split_dir)
    test_df.to_csv(os.path.join(_split_dir, f"{out_name}_valid.csv"), index=False)
    _split_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(_split_dir):
        os.makedirs(_split_dir)
    test_df.to_csv(os.path.join(_split_dir, f"{out_name}_test.csv"), index=False)

    print(f"Saved {len(train_df)} training and {len(test_df)} testing proteins.")

    for _set, _df in zip(["TRAIN", "TEST"], [train_df, test_df]):
        n_pos = np.sum(_df["Y"] == 0)
        n_neg = np.sum(_df["Y"] == 1)
        n = len(_df)
        print(f"  {_set} Pos/Neg ratio: neg={n_neg}, pos={n_pos}: {n_pos / n_neg:0.3f}")
        print(f"  {_set} Trivial accuracy: {n_pos / n:0.3f}")

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
