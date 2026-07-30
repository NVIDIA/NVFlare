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

"""Benchmark full-model SFT through Collab direct function calls."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from collab.pt_llm_sft.pt_llm_sft import LLMSFTClient, average_model_states, validate_prepared_data

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.recipe import SimEnv

EXCHANGE_FORMATS = ("native", "numpy")
_client = None
_exchange_format = None


def encode_state(state: dict[str, torch.Tensor], exchange_format: str):
    if exchange_format == "native":
        return state
    encoded = {}
    for name, tensor in state.items():
        if tensor.device.type != "cpu":
            raise ValueError("the NumPy transition control requires CPU tensors")
        try:
            encoded[name] = tensor.detach().numpy()
        except TypeError as error:
            raise TypeError(f"cannot represent {name} with dtype {tensor.dtype} as NumPy") from error
    return encoded


def decode_state(state, exchange_format: str) -> dict[str, torch.Tensor]:
    if exchange_format == "native":
        return state
    return {name: torch.from_numpy(array) for name, array in state.items()}


def mean(values) -> float:
    return float(statistics.mean(values)) if values else 0.0


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }


def summarize(samples: list[dict]) -> dict:
    totals = [sample["total_seconds"] for sample in samples]
    return {
        "num_samples": len(samples),
        "mean_total_seconds": mean(totals),
        "median_total_seconds": float(statistics.median(totals)),
        "mean_call_seconds": mean([sample["call_seconds"] for sample in samples]),
        "mean_server_transition_seconds": mean(
            [
                sample["server_outgoing_transition_seconds"] + sample["server_incoming_transition_seconds"]
                for sample in samples
            ]
        ),
        "mean_aggregation_seconds": mean([sample["aggregation_seconds"] for sample in samples]),
        "mean_client_training_seconds": mean([sample["client_training_seconds_mean"] for sample in samples]),
        "mean_client_transition_seconds": mean(
            [
                sample["client_incoming_transition_seconds_mean"] + sample["client_outgoing_transition_seconds_mean"]
                for sample in samples
            ]
        ),
    }


@collab.init
def initialize_client():
    global _client, _exchange_format
    if collab.site_name == "server":
        return

    config = collab.get_app_prop("benchmark_config")
    _exchange_format = config["exchange_format"]
    _client = LLMSFTClient(
        model_name_or_path=config["model_name_or_path"],
        data_root=config["data_root"],
        output_root=config["trainer_output_root"],
        syncs_per_epoch=config["syncs_per_epoch"],
        learning_rate=config["learning_rate"],
        max_length=config["max_length"],
        evaluate_global_model=config["evaluate_global_model"],
        model_revision=config.get("model_revision"),
        precision=config["precision"],
    )
    _client.initialize()


@collab.publish
def get_model_state():
    if _client is None:
        raise RuntimeError("SFT benchmark client is not initialized")
    state = _client.get_model_state()
    transition_start = time.perf_counter()
    payload = encode_state(state, _exchange_format)
    return {
        "weights": payload,
        "transition_seconds": time.perf_counter() - transition_start,
    }


@collab.publish
def train(sync_number, global_weights):
    if _client is None:
        raise RuntimeError("SFT benchmark client is not initialized")

    transition_start = time.perf_counter()
    decoded_weights = decode_state(global_weights, _exchange_format)
    incoming_transition_seconds = time.perf_counter() - transition_start

    training_start = time.perf_counter()
    result = _client.train(sync_number, decoded_weights)
    training_seconds = time.perf_counter() - training_start

    transition_start = time.perf_counter()
    result["weights"] = encode_state(result["weights"], _exchange_format)
    outgoing_transition_seconds = time.perf_counter() - transition_start
    result["benchmark_timing"] = {
        "training_seconds": training_seconds,
        "incoming_transition_seconds": incoming_transition_seconds,
        "outgoing_transition_seconds": outgoing_transition_seconds,
    }
    return result


@collab.main
def run_benchmark():
    config = collab.get_app_prop("benchmark_config")
    exchange_format = config["exchange_format"]
    metrics_file = Path(config["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    client_group = collab.clients(timeout=config["call_timeout"])

    initial = collab.clients[0](timeout=config["call_timeout"]).get_model_state()
    global_weights = decode_state(initial["weights"], exchange_format)
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in global_weights.values())
    samples = []
    total_syncs = config.get("max_syncs", config["num_epochs"] * config["syncs_per_epoch"])

    for sync_number in range(1, total_syncs + 1):
        total_start = time.perf_counter()
        transition_start = time.perf_counter()
        outgoing_weights = encode_state(global_weights, exchange_format)
        server_outgoing_transition_seconds = time.perf_counter() - transition_start

        call_start = time.perf_counter()
        call_results = client_group.train(sync_number, outgoing_weights)
        call_seconds = time.perf_counter() - call_start
        updates = dict(call_results)
        if call_results.failures:
            failures = ", ".join(f"{site}: {error}" for site, error in call_results.failures.items())
            raise RuntimeError(f"SFT benchmark client failures: {failures}")

        transition_start = time.perf_counter()
        for update in updates.values():
            update["weights"] = decode_state(update["weights"], exchange_format)
        server_incoming_transition_seconds = time.perf_counter() - transition_start

        aggregation_start = time.perf_counter()
        global_weights, average_loss = average_model_states(updates, config["num_clients"])
        aggregation_seconds = time.perf_counter() - aggregation_start
        timings = [update["benchmark_timing"] for update in updates.values()]
        sample = {
            "sync_number": sync_number,
            "average_loss": average_loss,
            "call_seconds": call_seconds,
            "server_outgoing_transition_seconds": server_outgoing_transition_seconds,
            "server_incoming_transition_seconds": server_incoming_transition_seconds,
            "aggregation_seconds": aggregation_seconds,
            "client_training_seconds_mean": mean([timing["training_seconds"] for timing in timings]),
            "client_incoming_transition_seconds_mean": mean(
                [timing["incoming_transition_seconds"] for timing in timings]
            ),
            "client_outgoing_transition_seconds_mean": mean(
                [timing["outgoing_transition_seconds"] for timing in timings]
            ),
            "total_seconds": time.perf_counter() - total_start,
        }
        samples.append(sample)
        print(
            f"[{exchange_format}] sync {sync_number}/{total_syncs}: "
            f"total={sample['total_seconds']:.4f}s call={call_seconds:.4f}s "
            f"aggregation={aggregation_seconds:.4f}s"
        )

    metrics = {
        "workload": "pt_llm_sft",
        "scheme": "collab",
        "exchange_format": exchange_format,
        "config": config,
        "environment": environment_info(),
        "payload_bytes": payload_bytes,
        "initial_client_transition_seconds": initial["transition_seconds"],
        "samples": samples,
        "summary": summarize(samples),
    }
    with metrics_file.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    print(f"Wrote benchmark metrics to {metrics_file}")
    return metrics["summary"]


def load_config(path: Path, exchange_format: str, metrics_file: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if exchange_format not in EXCHANGE_FORMATS:
        raise ValueError(f"unsupported exchange format: {exchange_format}")
    config["exchange_format"] = exchange_format
    config["metrics_file"] = str(metrics_file.expanduser().resolve())
    return config


def make_recipe(config: dict) -> CollabRecipe:
    recipe = CollabRecipe(
        job_name=f"benchmark_pt_llm_sft_{config['exchange_format']}",
        min_clients=config["num_clients"],
        sync_task_timeout=config["call_timeout"],
    )
    recipe.set_server_prop("benchmark_config", config)
    recipe.set_client_prop("benchmark_config", config)
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exchange-format", choices=EXCHANGE_FORMATS, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--workspace-root", required=True)
    args = parser.parse_args()

    config = load_config(args.config.resolve(), args.exchange_format, args.metrics_file)
    validate_prepared_data(Path(config["data_root"]), config["num_clients"])
    simple_logging()
    run = make_recipe(config).execute(
        SimEnv(
            num_clients=config["num_clients"],
            gpu_config=config.get("gpu_config"),
            workspace_root=args.workspace_root,
        )
    )
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
