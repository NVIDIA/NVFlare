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
import multiprocessing
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
import xgboost.federated

PRINT_SAMPLE = False


def train_federated_args_parser():
    parser = argparse.ArgumentParser(description="Train federated XGBoost model")
    parser.add_argument("--world_size", type=int, default=3, help="Total number of clients")
    parser.add_argument("--gpu", type=int, default=0, help="Whether to use gpu for training, 0 for cpu, 1 for gpu")
    parser.add_argument(
        "--vert", type=int, default=0, help="Horizontal or vertical training, 0 for horizontal, 1 for vertical"
    )
    parser.add_argument(
        "--enc", type=int, default=0, help="Whether to use encryption plugin, 0 for non-encrypted, 1 for encrypted"
    )
    parser.add_argument(
        "--data_train_root",
        type=str,
        default="/tmp/nvflare/dataset/xgb_dataset/base_xgb_data",
        help="Path to training data folder",
    )
    parser.add_argument(
        "--data_test_file",
        type=str,
        default="/tmp/nvflare/dataset/xgb_dataset/test.csv",
        help="Path to testing data file",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/tmp/nvflare/workspace/fedxgb_secure/train_standalone/federated",
        help="Output path for the data split file",
    )
    return parser


def load_test_data(data_path: str):
    df = pd.read_csv(data_path)
    # Split to feature and label
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return X, y


def run_server(port: int, world_size: int) -> None:
    xgboost.federated.run_federated_server(port=port, n_workers=world_size)


def run_worker(port: int, world_size: int, rank: int, args) -> None:
    if args.enc:
        plugin = {"name": "mock"}
    else:
        plugin = {}
    communicator_env = {
        "dmlc_communicator": "federated",
        "federated_server_address": f"localhost:{port}",
        "federated_world_size": world_size,
        "federated_rank": rank,
        "federated_plugin": plugin,
    }

    # Always call this before using distributed module
    with xgb.collective.CommunicatorContext(**communicator_env):
        # Specify file path, rank 0 as the label owner, others as the feature owner
        train_path = f"{args.data_train_root}/site-{rank + 1}/train.csv"
        valid_path = f"{args.data_train_root}/site-{rank + 1}/valid.csv"

        # Load file directly to tell the match from loading with DMatrix
        df_train = pd.read_csv(train_path, header=None)
        if PRINT_SAMPLE:
            # print number of rows and columns for each worker
            print(f"Direct load: rank={rank}, nrow={df_train.shape[0]}, ncol={df_train.shape[1]}")
            # print one sample row of the data
            print(f"Direct load: rank={rank}, one sample row of the data: \n {df_train.iloc[0]}")

        # Load file, file will not be sharded in federated mode.
        if args.vert:
            data_split_mode = 1
            if rank == 0:
                label = "&label_column=0"
            else:
                label = ""
        else:
            data_split_mode = 0
            label = "&label_column=0"
        dtrain = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=data_split_mode)
        dvalid = xgb.DMatrix(valid_path + f"?format=csv{label}", data_split_mode=data_split_mode)

        if PRINT_SAMPLE:
            # print number of rows and columns for each worker
            print(f"DMatrix: rank={rank}, nrow={dtrain.num_row()}, ncol={dtrain.num_col()}")
            # print one sample row of the data
            data_sample = dtrain.get_data()[0]
            print(f"DMatrix: rank={rank}, one sample row of the data: \n {data_sample}")

        # Specify parameters via map, definition are same as c++ version
        if args.gpu:
            device = "cuda:0"
        else:
            device = "cpu"
        param = {
            "max_depth": 3,
            "eta": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": device,
            "nthread": 1,
        }

        # Specify validations set to watch performance
        watchlist = [(dvalid, "eval"), (dtrain, "train")]
        num_round = 3

        # Run training, all the features in training API is available.
        bst = xgb.train(param, dtrain, num_round, evals=watchlist)

        # Save the model
        rank = xgb.collective.get_rank()
        bst.save_model(f"{args.out_path}/model.{rank}.json")
        xgb.collective.communicator_print("Finished training\n")

        # save feature importance score to file
        score = bst.get_score(importance_type="gain")
        with open(f"{args.out_path}/feat_importance.{rank}.txt", "w") as f:
            for key in score:
                f.write(f"{key}: {score[key]}\n")

        # Load test data
        X_test, y_test = load_test_data(args.data_test_file)
        # construct xgboost DMatrix
        dmat_test = xgb.DMatrix(X_test, label=y_test)

        # Explain the model
        explainer = shap.TreeExplainer(bst)
        explanation = explainer(dmat_test)

        # save the beeswarm plot to png file
        shap.plots.beeswarm(explanation, show=False)
        img = plt.gcf()
        img.savefig(f"{args.out_path}/shap.{rank}.png")

        # dump tree and save to text file
        dump = bst.get_dump()
        with open(f"{args.out_path}/tree_dump.{rank}.txt", "w") as f:
            for tree in dump:
                f.write(tree)

        # plot tree and save to png file
        xgb.plot_tree(bst, num_trees=0, rankdir="LR")
        fig = plt.gcf()
        fig.set_size_inches(18, 5)
        plt.savefig(f"{args.out_path}/tree.{rank}.png", dpi=100)

        # export tree to dataframe
        tree_df = bst.trees_to_dataframe()
        tree_df.to_csv(f"{args.out_path}/tree_df.{rank}.csv")


def main():
    parser = train_federated_args_parser()
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    port = 1111
    world_size = args.world_size

    server = multiprocessing.Process(target=run_server, args=(port, world_size))
    server.start()
    time.sleep(1)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for rank in range(world_size):
        worker = multiprocessing.Process(target=run_worker, args=(port, world_size, rank, args))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    server.terminate()


if __name__ == "__main__":
    main()
