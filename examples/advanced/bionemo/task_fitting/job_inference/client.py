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

"""Client script for ESM2 embedding inference."""

import argparse
import os
import subprocess

import torch

import nvflare.client as flare


def main():
    parser = argparse.ArgumentParser(description="Run ESM2 inference for embeddings")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to ESM2 checkpoint")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--results-path", type=str, required=True, help="Path to save results")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision for inference")
    parser.add_argument("--micro-batch-size", type=int, default=64, help="Micro batch size")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")

    args = parser.parse_args()

    # Initialize NVFlare client
    flare.init()

    # Get the site name for this client
    site_name = flare.get_site_name()
    print(f"\n[Site={site_name}] Running ESM2 inference\n")

    # Construct data path for this client
    data_path = os.path.join(args.data_root, f"data_{site_name}.csv")
    results_path_full = os.path.join(args.results_path, f"inference_results_{site_name}")

    # Inference loop
    while flare.is_running():
        # Receive the model task from the server
        input_model = flare.receive()
        print(f"\n[Site={site_name}] Received task at round {input_model.current_round}\n")

        # Validate that this is a single-round inference job
        assert (
            input_model.total_rounds == 1
        ), f"Inference should be a single round but server requested {input_model.total_rounds} rounds!"

        # Build the inference command
        command = [
            "infer_esm2",
            "--checkpoint-path",
            args.checkpoint_path,
            "--data-path",
            data_path,
            "--results-path",
            results_path_full,
            "--precision",
            args.precision,
            "--include-embeddings",
            "--include-logits",
            "--include-input-ids",
            "--micro-batch-size",
            str(args.micro_batch_size),
            "--num-gpus",
            str(args.num_gpus),
        ]

        print(f"[Site={site_name}] Running command: {' '.join(command)}")

        # Run the inference
        subprocess_result = subprocess.run(command, capture_output=True, text=True)

        if subprocess_result.returncode == 0:
            print(f"[Site={site_name}] Inference completed successfully")
            print(f"[Site={site_name}] Results saved to: {results_path_full}")

            # Load results and extract metadata
            results_file = os.path.join(results_path_full, "predictions__rank_0.pt")
            if os.path.isfile(results_file):
                print(f"[Site={site_name}] Loading result info from: {results_file}")
                results = torch.load(results_file, weights_only=False)

                result_shapes = {}
                for k, v in results.items():
                    if v is not None:
                        result_shapes[k] = list(v.shape)  # Convert torch.Size to list

                n_sequences = len(results["embeddings"])
                print(f"[Site={site_name}] Extracted {n_sequences} embeddings")

                # For FedAvgRecipe, send back the same model we received (no training occurred)
                # Attach inference metadata to the meta field
                result = flare.FLModel(
                    params={"success": True},
                    params_type=input_model.params_type,
                    meta={
                        "num_sequences": n_sequences,
                        "result_shapes": result_shapes,
                    },
                )
            else:
                print(f"[Site={site_name}] Warning: {results_file} doesn't exist!")
                result = flare.FLModel(
                    params={"success": False},
                    params_type=input_model.params_type,
                    meta={"num_sequences": "n/a", "result_shapes": "n/a", "error": "Results file not found"},
                )
        else:
            print(f"[Site={site_name}] Inference failed with return code {subprocess_result.returncode}")
            print(f"STDOUT: {subprocess_result.stdout}")
            print(f"STDERR: {subprocess_result.stderr}")

            # Send failure status but still return the model params
            result = flare.FLModel(
                params={"success": False},
                params_type=input_model.params_type,
                meta={
                    "error": f"Inference failed with return code {subprocess_result.returncode}",
                },
            )

        print(f"[Site={site_name}] Sending meta information back to server")
        flare.send(result)


if __name__ == "__main__":
    main()
    flare.shutdown()
