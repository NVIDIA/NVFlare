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

"""Standard NVFlare Client API adapter for the matched SFT benchmark."""

import argparse

from collab.pt_llm_sft.pt_llm_sft import LLMSFTClient

import nvflare.client as flare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--syncs-per-epoch", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--evaluate-global-model", action="store_true")
    args = parser.parse_args()

    flare.init()
    client = LLMSFTClient(
        model_name_or_path=args.model_name_or_path,
        data_root=args.data_root,
        output_root=args.output_root,
        syncs_per_epoch=args.syncs_per_epoch,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        evaluate_global_model=args.evaluate_global_model,
        model_revision=args.model_revision,
        precision=args.precision,
        site_name=flare.get_site_name(),
    )
    client.initialize()

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        sync_number = input_model.current_round + 1
        global_weights = {name.removeprefix("model."): value for name, value in input_model.params.items()}
        result = client.train(sync_number, global_weights)
        output_weights = {f"model.{name}": value for name, value in result["weights"].items()}
        metrics = {
            "train_loss": result["train_loss"],
            "model_selection_score": -result["train_loss"],
        }
        if result["eval_loss"] is not None:
            metrics["eval_loss"] = result["eval_loss"]
            metrics["neg_eval_loss"] = -result["eval_loss"]

        flare.send(
            flare.FLModel(
                params=output_weights,
                metrics=metrics,
                meta={"NUM_STEPS_CURRENT_ROUND": result["num_examples"]},
            )
        )


if __name__ == "__main__":
    main()
