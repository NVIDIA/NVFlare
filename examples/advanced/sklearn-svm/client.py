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
Client training script for federated SVM with scikit-learn.
Uses the NVFlare Client API for federated learning.

SVM training requires two rounds:
- Round 0: Train local SVM, extract and send support vectors
- Round 1: Validate using global support vectors
"""

import argparse

from sklearn.metrics import roc_auc_score

import nvflare.client as flare
from nvflare.app_opt.sklearn.data_loader import load_data_for_range
from nvflare.fuel.utils.import_utils import optional_import


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument(
        "--backend",
        type=str,
        default="sklearn",
        choices=["sklearn", "cuml"],
        help="Backend library to use",
    )
    parser.add_argument("--train_start", type=int, default=0, help="Training data start index")
    parser.add_argument("--train_end", type=int, default=100, help="Training data end index")
    parser.add_argument("--valid_start", type=int, default=100, help="Validation data start index")
    parser.add_argument("--valid_end", type=int, default=569, help="Validation data end index")
    args = parser.parse_args()

    print(f"Loading training data from {args.data_path}")
    print(f"  Train range: [{args.train_start}, {args.train_end})")
    print(f"  Valid range: [{args.valid_start}, {args.valid_end})")
    print(f"  Backend: {args.backend}")

    # Import the appropriate SVM library
    if args.backend == "sklearn":
        svm_lib, flag = optional_import(module="sklearn.svm")
        if not flag:
            raise ImportError("Failed to import sklearn.svm")
    elif args.backend == "cuml":
        svm_lib, flag = optional_import(module="cuml.svm")
        if not flag:
            raise ImportError("Failed to import cuml.svm. Make sure cuML is installed.")
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Load data
    train_data = load_data_for_range(args.data_path, args.train_start, args.train_end)
    X_train, y_train, train_size = train_data

    valid_data = load_data_for_range(args.data_path, args.valid_start, args.valid_end)
    X_valid, y_valid, valid_size = valid_data

    print(f"Loaded {train_size} training samples, {valid_size} validation samples")
    print(f"Feature dimension: {X_train.shape[1]}")

    # Initialize Federated Learning
    flare.init()

    print("Starting federated learning loop...")
    print("Note: SVM requires 2 rounds - round 0 for training, round 1 for validation")

    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        curr_round = input_model.current_round
        print(f"\n=== Round {curr_round} ===")

        # Extract kernel from global params (used in both rounds)
        global_params = input_model.params
        kernel = global_params.get("kernel", "rbf")
        print(f"Using kernel: {kernel}")

        if curr_round == 0:
            # Round 0: Train local SVM and extract support vectors
            print("Training local SVM on client data")

            # Train local SVM (kept in memory for potential debugging, not used afterwards)
            local_svm = svm_lib.SVC(kernel=kernel)
            local_svm.fit(X_train, y_train)

            # Extract support vectors
            support_indices = local_svm.support_
            local_support_x = X_train[support_indices]
            local_support_y = y_train[support_indices]

            print("Local SVM training complete")
            print(f"Number of support vectors: {len(support_indices)}")

            # Send support vectors to server
            params = {
                "support_x": local_support_x,
                "support_y": local_support_y,
            }

            # No validation in round 0
            auc = 0.0

        elif curr_round == 1:
            # Round 1: Validate using global support vectors
            print("Validating with global support vectors")
            global_support_x = global_params["support_x"]
            global_support_y = global_params["support_y"]

            print(f"Number of global support vectors: {len(global_support_x)}")

            # Train a new SVM with global support vectors
            global_svm = svm_lib.SVC(kernel=kernel)
            global_svm.fit(global_support_x, global_support_y)

            # Validate on local validation data
            y_pred = global_svm.predict(X_valid)
            auc = roc_auc_score(y_valid, y_pred)
            print(f"Validation AUC: {auc:.4f}")

            # No parameters to send in round 1 (just validation)
            params = {
                "support_x": global_support_x,
                "support_y": global_support_y,
            }

        else:
            # SVM should only run for 2 rounds
            print(f"Warning: SVM should only run for 2 rounds, but got round {curr_round}")
            break

        # Send model back to server
        print(f"Sending results back to server (train_size={train_size})")
        flare.send(
            flare.FLModel(
                params=params,
                metrics={"AUC": auc},
                meta={"NUM_STEPS_CURRENT_ROUND": train_size},
            )
        )

    print("\nFederated SVM training completed!")


if __name__ == "__main__":
    main()
