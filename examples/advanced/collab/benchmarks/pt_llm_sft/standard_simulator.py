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

"""Run the matched SFT workload with the standard NVFlare simulator."""

import argparse
import json
import platform
import shlex
import time
from pathlib import Path

import torch

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv, set_per_site_config

STANDARD_CLIENT_SOURCE = Path("collab/benchmarks/pt_llm_sft/standard_client.py")
STANDARD_MODEL_SOURCE = Path("collab/benchmarks/pt_llm_sft/standard_model.py")
COLLAB_SFT_SOURCE = Path("collab/pt_llm_sft/pt_llm_sft.py")


def load_config(path: Path, metrics_file: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["metrics_file"] = str(metrics_file.expanduser().resolve())
    return config


def make_recipe(config: dict) -> FedAvgRecipe:
    sites = [f"site-{number}" for number in range(1, config["num_clients"] + 1)]
    client_args = shlex.join(
        [
            "--model-name-or-path",
            config["model_name_or_path"],
            "--data-root",
            config["data_root"],
            "--output-root",
            config["trainer_output_root"],
            "--syncs-per-epoch",
            str(config["syncs_per_epoch"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--max-length",
            str(config["max_length"]),
            "--precision",
            config["precision"],
        ]
        + (["--evaluate-global-model"] if config["evaluate_global_model"] else [])
    )
    recipe = FedAvgRecipe(
        name="benchmark_pt_llm_sft_standard",
        model={
            "class_path": "collab.benchmarks.pt_llm_sft.standard_model.CausalLMModel",
            "args": {
                "model_name_or_path": config["model_name_or_path"],
                "precision": config["precision"],
            },
        },
        min_clients=config["num_clients"],
        num_rounds=config["num_epochs"] * config["syncs_per_epoch"],
        train_script=str(STANDARD_CLIENT_SOURCE),
        train_args=client_args,
        launch_external_process=True,
        server_expected_format=ExchangeFormat.PYTORCH,
        key_metric="model_selection_score",
    )
    set_per_site_config(recipe, {site: {} for site in sites})
    recipe.add_client_file(str(COLLAB_SFT_SOURCE))
    recipe.add_server_file(str(STANDARD_MODEL_SOURCE))
    recipe.add_client_config(
        {
            "get_task_timeout": config["call_timeout"],
            "submit_task_result_timeout": config["call_timeout"],
        }
    )
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--workspace-root", required=True)
    args = parser.parse_args()

    config = load_config(args.config.resolve(), args.metrics_file)
    sites = [f"site-{number}" for number in range(1, config["num_clients"] + 1)]
    started = time.perf_counter()
    run = make_recipe(config).execute(
        SimEnv(
            clients=sites,
            num_threads=config["num_clients"],
            gpu_config=config.get("gpu_config"),
            workspace_root=args.workspace_root,
        )
    )
    execution_seconds = time.perf_counter() - started
    metrics = {
        "workload": "pt_llm_sft",
        "scheme": "standard",
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "execution_seconds": execution_seconds,
        "result": str(run.get_result()),
    }
    metrics_file = Path(config["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    print(f"Wrote benchmark metrics to {metrics_file}")
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
