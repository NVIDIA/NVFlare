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

import multiprocessing
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
import xgboost.federated

PRINT_SAMPLE = False
DATASET_ROOT = "/tmp/nvflare/xgb_dataset/horizontal_xgb_data"
TEST_DATA_PATH = "/tmp/nvflare/xgb_dataset/test.csv"
OUTPUT_ROOT = "/tmp/nvflare/xgb_exp/hori_base_gpu"
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)


def load_test_data(data_path: str):
    df = pd.read_csv(data_path)
    # Split to feature and label
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return X, y


def run_server(port: int, world_size: int) -> None:
    xgboost.federated.run_federated_server(port, world_size)


def run_worker(port: int, world_size: int, rank: int) -> None:
    communicator_env = {
        "dmlc_communicator": "federated",
        "federated_server_address": f"localhost:{port}",
        "federated_world_size": world_size,
        "federated_rank": rank,
    }

    # Always call this before using distributed module
    with xgb.collective.CommunicatorContext(**communicator_env):
        # Specify file path, rank 0 as the label owner, others as the feature owner
        train_path = f"{DATASET_ROOT}/site-{rank + 1}/train.csv"
        valid_path = f"{DATASET_ROOT}/site-{rank + 1}/valid.csv"

        # Load file directly to tell the match from loading with DMatrix
        df_train = pd.read_csv(train_path, header=None)
        if PRINT_SAMPLE:
            # print number of rows and columns for each worker
            print(f"Direct load: rank={rank}, nrow={df_train.shape[0]}, ncol={df_train.shape[1]}")
            # print one sample row of the data
            print(f"Direct load: rank={rank}, one sample row of the data: \n {df_train.iloc[0]}")

        # Load file, file will not be sharded in federated mode.
        label = "&label_column=0"
        # for Vertical XGBoost, read from csv with label_column and set data_split_mode to 1 for column mode
        dtrain = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=0)
        dvalid = xgb.DMatrix(valid_path + f"?format=csv{label}", data_split_mode=0)

        if PRINT_SAMPLE:
            # print number of rows and columns for each worker
            print(f"DMatrix: rank={rank}, nrow={dtrain.num_row()}, ncol={dtrain.num_col()}")
            # print one sample row of the data
            data_sample = dtrain.get_data()[0]
            print(f"DMatrix: rank={rank}, one sample row of the data: \n {data_sample}")

        # Specify parameters via map, definition are same as c++ version
        param = {
            "max_depth": 3,
            "eta": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda:0",
            "nthread": 1,
        }

        # Specify validations set to watch performance
        watchlist = [(dvalid, "eval"), (dtrain, "train")]
        num_round = 3

        # Run training, all the features in training API is available.
        bst = xgb.train(param, dtrain, num_round, evals=watchlist)

        # Save the model
        rank = xgb.collective.get_rank()
        bst.save_model(f"{OUTPUT_ROOT}/model.hori.base.{rank}.json")
        xgb.collective.communicator_print("Finished training\n")

        # save feature importance score to file
        score = bst.get_score(importance_type="gain")
        with open(f"{OUTPUT_ROOT}/feat_importance.hori.base.{rank}.txt", "w") as f:
            for key in score:
                f.write(f"{key}: {score[key]}\n")

        # Load test data
        X_test, y_test = load_test_data(TEST_DATA_PATH)
        # construct xgboost DMatrix
        dmat_test = xgb.DMatrix(X_test, label=y_test)

        # Explain the model
        explainer = shap.TreeExplainer(bst)
        explanation = explainer(dmat_test)

        # save the beeswarm plot to png file
        shap.plots.beeswarm(explanation, show=False)
        img = plt.gcf()
        img.savefig(f"{OUTPUT_ROOT}/shap.hori.base.{rank}.png")

        # dump tree and save to text file
        dump = bst.get_dump()
        with open(f"{OUTPUT_ROOT}/tree_dump.hori.base.{rank}.txt", "w") as f:
            for tree in dump:
                f.write(tree)

        # plot tree and save to png file
        xgb.plot_tree(bst, num_trees=0, rankdir="LR")
        fig = plt.gcf()
        fig.set_size_inches(18, 5)
        plt.savefig(f"{OUTPUT_ROOT}/tree.hori.base.{rank}.png", dpi=100)

        # export tree to dataframe
        tree_df = bst.trees_to_dataframe()
        tree_df.to_csv(f"{OUTPUT_ROOT}/tree_df.hori.base.{rank}.csv")


def run_federated() -> None:
    port = 11111
    world_size = int(sys.argv[1])

    server = multiprocessing.Process(target=run_server, args=(world_size, port))
    server.start()
    time.sleep(1)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for rank in range(world_size):
        worker = multiprocessing.Process(target=run_worker, args=(port, world_size, rank))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    server.terminate()


if __name__ == "__main__":
    run_federated()
