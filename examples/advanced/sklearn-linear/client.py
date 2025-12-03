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

"""
Client training script for federated linear model with scikit-learn.
Uses the NVFlare Client API for federated learning.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

import nvflare.client as flare


def load_data(data_path, start, end):
    """Load data for a specific range.

    Args:
        data_path: Path to CSV file
        start: Start row index
        end: End row index

    Returns:
        Tuple of (X, y, size) where X is features, y is labels, size is number of samples
    """
    df = pd.read_csv(data_path, header=None, skiprows=start, nrows=end - start)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, y, len(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--train_start", type=int, default=1100000, help="Training data start index")
    parser.add_argument("--train_end", type=int, default=3000000, help="Training data end index")
    parser.add_argument("--valid_start", type=int, default=0, help="Validation data start index")
    parser.add_argument("--valid_end", type=int, default=1100000, help="Validation data end index")
    args = parser.parse_args()

    print(f"Loading training data from {args.data_path}")
    print(f"  Train range: [{args.train_start}, {args.train_end})")
    print(f"  Valid range: [{args.valid_start}, {args.valid_end})")

    # Load data
    X_train, y_train, train_size = load_data(args.data_path, args.train_start, args.train_end)
    X_valid, y_valid, valid_size = load_data(args.data_path, args.valid_start, args.valid_end)

    print(f"Loaded {train_size} training samples, {valid_size} validation samples")
    print(f"Feature dimension: {X_train.shape[1]}")

    # Initialize Federated Learning
    flare.init()

    local_model = None
    n_features = X_train.shape[1]

    print("Starting federated learning loop...")

    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        curr_round = input_model.current_round
        print(f"\n=== Round {curr_round} ===")

        if curr_round == 0:
            # Initialize model with global parameters
            print("Initializing model with global parameters")
            global_params = input_model.params

            fit_intercept = bool(global_params["fit_intercept"])
            local_model = SGDClassifier(
                loss=global_params["loss"],
                penalty=global_params["penalty"],
                fit_intercept=fit_intercept,
                learning_rate=global_params["learning_rate"],
                eta0=global_params["eta0"],
                max_iter=1,  # One iteration per round
                warm_start=True,  # Continue from previous state
                random_state=0,
            )

            # Initialize weights to zero
            n_classes = global_params["n_classes"]
            local_model.classes_ = np.array(list(range(n_classes)))
            local_model.coef_ = np.zeros((1, n_features))
            if fit_intercept:
                local_model.intercept_ = np.zeros((1,))
        else:
            # Update model with global parameters from aggregation
            print("Updating model with global parameters")
            local_model.coef_ = input_model.params["coef"]
            if local_model.fit_intercept:
                local_model.intercept_ = input_model.params["intercept"]

        # Train locally for one iteration
        print(f"Training on {train_size} samples...")
        local_model.fit(X_train, y_train)

        # Validate - use decision_function for AUC (returns signed distance to hyperplane)
        y_scores = local_model.decision_function(X_valid)
        auc = roc_auc_score(y_valid, y_scores)
        print(f"Validation AUC: {auc:.4f}")

        # Prepare parameters to send back
        if local_model.fit_intercept:
            params = {
                "coef": local_model.coef_,
                "intercept": local_model.intercept_,
            }
        else:
            params = {"coef": local_model.coef_}

        # Send model back to server
        print(f"Sending model back to server (train_size={train_size})")
        flare.send(
            flare.FLModel(
                params=params,
                metrics={"AUC": auc},
                meta={"NUM_STEPS_CURRENT_ROUND": train_size},
            )
        )

    print("\nFederated learning completed!")


if __name__ == "__main__":
    main()
