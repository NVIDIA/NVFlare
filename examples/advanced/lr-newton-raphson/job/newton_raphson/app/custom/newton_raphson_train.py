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
from sklearn.metrics import accuracy_score, precision_score

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.np.constants import NPConstants
from nvflare.client.tracking import SummaryWriter


def parse_arguments():
    """
    Parse command line args for client side training.
    """
    parser = argparse.ArgumentParser(description="Federated Second-Order Newton Raphson")

    parser.add_argument("--data_root", type=str, help="Path to load client side data.")

    return parser.parse_args()


def load_data(data_root, site_name):
    """
    Load the data for each client.

    Args:
        data_root: root directory storing client site data.
        site_name: client site name
    Returns:
        A dict with client site training and validation data.
    """
    print("loading data for client {} from: {}".format(site_name, data_root))
    train_x_path = os.path.join(data_root, "{}.train.x.npy".format(site_name))
    train_y_path = os.path.join(data_root, "{}.train.y.npy".format(site_name))
    test_x_path = os.path.join(data_root, "{}.test.x.npy".format(site_name))
    test_y_path = os.path.join(data_root, "{}.test.y.npy".format(site_name))

    train_X = np.load(train_x_path)
    train_y = np.load(train_y_path)
    valid_X = np.load(test_x_path)
    valid_y = np.load(test_y_path)

    return {"train_X": train_X, "train_y": train_y, "valid_X": valid_X, "valid_y": valid_y}


def sigmoid(inp):
    return 1.0 / (1.0 + np.exp(-inp))


def train_newton_raphson(data, theta):
    """
    Compute gradient and hessian on local data
    based on paramters received from server.

    """
    train_X = data["train_X"]
    train_y = data["train_y"]

    # Add intercept, pre-pend 1s to as first
    # column of train_X
    train_X = np.concatenate((np.ones((train_X.shape[0], 1)), train_X), axis=1)

    # Compute probabilities from current weights
    proba = sigmoid(np.dot(train_X, theta))

    # The gradient is X^T . (y - proba)
    gradient = np.dot(train_X.T, (train_y - proba))

    # The hessian is X^T . D . X, where D is the
    # diagnoal matrix with values proba * (1 - proba)
    D = np.diag((proba * (1 - proba))[:, 0])
    hessian = train_X.T.dot(D).dot(train_X)

    return {"gradient": gradient, "hessian": hessian}


def validate(data, theta):
    """
    Performs local validation.
    Computes accuracy and precision scores.

    """
    valid_X = data["valid_X"]
    valid_y = data["valid_y"]

    # Add intercept, pre-pend 1s to as first
    # column of valid_X
    valid_X = np.concatenate((np.ones((valid_X.shape[0], 1)), valid_X), axis=1)

    # Compute probabilities from current weights
    proba = sigmoid(np.dot(valid_X, theta))

    return {"accuracy": accuracy_score(valid_y, proba.round()), "precision": precision_score(valid_y, proba.round())}


def main():
    """
    This is a typical ML training loop,
    augmented with Flare Client API to
    perform local training on each client
    side and send result to server.

    """
    args = parse_arguments()

    flare.init()

    site_name = flare.get_site_name()
    print("training on client site: {}".format(site_name))

    # Load client site data.
    data = load_data(args.data_root, site_name)

    # Get metric summary writer
    writer = SummaryWriter()

    while flare.is_running():

        # Receive global model (FLModel) from server.
        global_model = flare.receive()

        curr_round = global_model.current_round
        print("current_round={}".format(curr_round))

        print(
            ("[ROUND {}] - client site: {}, received " "global model: {}").format(curr_round, site_name, global_model)
        )

        # Get the weights, aka parameter theta for
        # logistic regression.
        global_weights = global_model.params[NPConstants.NUMPY_KEY]
        print("[ROUND {}] - global model weights: {}".format(curr_round, global_weights))

        # Local validation before training
        print(("[ROUND {}] - start validation of global " "model on client: {}").format(curr_round, site_name))
        validation_scores = validate(data, global_weights)
        print(
            ("[ROUND {}] - validation metric scores on " "client: {} = {}").format(
                curr_round, site_name, validation_scores
            )
        )

        # Write validation metric summary
        writer.add_scalar("{}/accuracy".format(site_name), validation_scores["accuracy"], curr_round)

        writer.add_scalar("{}/precision".format(site_name), validation_scores["precision"], curr_round)

        # Local training
        print(("[ROUND {}] - start local training on client " "site: {}").format(curr_round, site_name))
        result_dict = train_newton_raphson(data, theta=global_weights)

        # Send result to server for aggregation.
        result_model = FLModel(params=result_dict, params_type=ParamsType.FULL)
        result_model.meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] = data["train_X"].shape[0]

        print(
            (
                "[ROUND {}] - local newton raphson training from " "client: {} complete, sending results to server: {}"
            ).format(curr_round, site_name, result_model)
        )

        flare.send(result_model)


if __name__ == "__main__":
    main()
