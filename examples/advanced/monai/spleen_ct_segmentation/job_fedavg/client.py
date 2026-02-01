# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from monai.fl.client import MonaiAlgo
from monai.fl.utils.constants import FlStatistics
from monai.fl.utils.exchange_object import ExchangeObject

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", type=str, required=True, help="Path to MONAI bundle")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local training epochs")
    parser.add_argument(
        "--send_weight_diff", action="store_true", help="Send weight differences instead of full weights"
    )
    args = parser.parse_args()

    # (2) initializes NVFlare client API
    flare.init()

    # (3) Create MonaiAlgo instance with bundle configuration
    algo = MonaiAlgo(
        bundle_root=args.bundle_root,
        local_epochs=args.local_epochs,
        send_weight_diff=args.send_weight_diff,
        config_train_filename="configs/train.json",
        disable_ckpt_loading=True,
    )

    # (4) Initialize the MonaiAlgo
    # Note: app_root can be set if bundle_root is relative
    algo.initialize(extra={"CLIENT_NAME": flare.get_site_name()})

    # (optional) Setup TensorBoard logging
    summary_writer = SummaryWriter()

    # (5) Federated Learning loop
    while flare.is_running():
        # Receive FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # Convert NVFlare FLModel to MONAI ExchangeObject
        global_weights = ExchangeObject(weights=input_model.params)

        # Evaluate the global weights
        test_report = algo.evaluate(data=global_weights)
        test_key_metric = test_report.metrics.get("val_accuracy")
        print("Test report:")
        for key, value in test_report.metrics.items():
            print(f"{key}: {value}")
            # Log each metric to SummaryWriter
            summary_writer.add_scalar(tag=f"validation/{key}", scalar=value, global_step=input_model.current_round)
        print(f"Test key metric: {test_key_metric}")

        # Train using MonaiAlgo
        algo.train(data=global_weights)

        # Get updated weights from MonaiAlgo
        updated_weights = algo.get_weights()

        # get the number of executed steps
        statistics = updated_weights.statistics
        executed_steps = statistics.get(FlStatistics.NUM_EXECUTED_ITERATIONS)
        print(f"Completed {executed_steps} training steps for current round")

        # Log training statistics to SummaryWriter
        if statistics:
            for key, value in statistics.items():
                if isinstance(value, (int, float)):
                    summary_writer.add_scalar(
                        tag=f"training/{key}", scalar=value, global_step=input_model.current_round
                    )

        # Convert MONAI ExchangeObject back to NVFlare FLModel
        output_model = flare.FLModel(
            params=updated_weights.weights,
            metrics=updated_weights.statistics if updated_weights.statistics else {},
            meta={
                "weight_type": updated_weights.weight_type.value if updated_weights.weight_type else "WEIGHTS",
                "NUM_STEPS_CURRENT_ROUND": executed_steps,
                "accuracy": test_key_metric,  # NVFlare expects the key metric to be "accuracy"
            },
        )

        # Send model back to NVFlare
        flare.send(output_model)

    # (6) Finalize the MonaiAlgo
    algo.finalize()


if __name__ == "__main__":
    main()
