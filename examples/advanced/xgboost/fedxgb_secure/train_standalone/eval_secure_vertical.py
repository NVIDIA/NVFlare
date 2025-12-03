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

import argparse
import multiprocessing
import time

import xgboost as xgb
import xgboost.federated
from sklearn.metrics import roc_auc_score


def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate federated XGBoost model")
    parser.add_argument("--world_size", type=int, default=3, help="Total number of clients")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/tmp/nvflare/dataset/xgb_dataset/vertical_xgb_data",
        help="Path to validation data folder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/nvflare/workspace/fedxgb_secure/train_standalone/vert_cpu_enc",
        help="Path to model",
    )
    return parser


def run_server(port: int, world_size: int) -> None:
    xgboost.federated.run_federated_server(port=port, n_workers=world_size)


def run_worker(port: int, world_size: int, rank: int, data_root: str, model_path: str) -> None:
    """Run a federated worker for model evaluation."""
    communicator_env = {
        "dmlc_communicator": "federated",
        "federated_server_address": f"localhost:{port}",
        "federated_world_size": world_size,
        "federated_rank": rank,
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        # Load validation data - rank 0 as label owner, others as feature owners
        valid_path = f"{data_root}/site-{rank + 1}/valid.csv"
        label_param = "&label_column=0" if rank == 0 else ""
        dvalid = xgb.DMatrix(f"{valid_path}?format=csv{label_param}", data_split_mode=1)

        # Load the trained model
        bst = xgb.Booster({"nthread": 1})
        current_rank = xgb.collective.get_rank()
        bst.load_model(f"{model_path}/model.{current_rank}.json")
        xgb.collective.communicator_print("Finished loading local models\n")

        # Make predictions
        preds = bst.predict(dvalid)

        # Only label owner calculates and reports metrics
        if current_rank == 0:
            y_valid = dvalid.get_label()
            auc_score = roc_auc_score(y_valid, preds)
            print(f"Validation AUC: {auc_score:.4f}")


def main():
    """Main function to run federated evaluation."""
    parser = create_parser()
    args = parser.parse_args()

    port = 1111
    world_size = args.world_size

    # Start federated server
    server = multiprocessing.Process(target=run_server, args=(port, world_size))
    server.start()
    time.sleep(1)

    if not server.is_alive():
        raise RuntimeError("Failed to start federated learning server")

    # Start workers
    workers = []
    for rank in range(world_size):
        worker = multiprocessing.Process(
            target=run_worker, args=(port, world_size, rank, args.data_root, args.model_path)
        )
        workers.append(worker)
        worker.start()

    # Wait for all workers to complete
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {worker.exitcode}")

    # Clean up
    server.terminate()


if __name__ == "__main__":
    main()
