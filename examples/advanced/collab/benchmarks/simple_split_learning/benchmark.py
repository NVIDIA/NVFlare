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

"""Benchmark split learning with native tensors or explicit NumPy transitions."""

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import default_collate
from torchvision import datasets, transforms

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.recipe import SimEnv

EXCHANGE_FORMATS = ("native", "numpy")
_activations = None
_bottom_model = None
_bottom_optimizer = None
_client_data = None
_config = None


def load_mnist(data_root: str):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return datasets.MNIST(root=data_root, train=True, download=False, transform=transform)


def get_batch(dataset, step: int, batch_size: int):
    if len(dataset) == 0:
        raise ValueError("dataset must not be empty")
    num_batches = math.ceil(len(dataset) / batch_size)
    batch_index = step % num_batches
    start = batch_index * batch_size
    stop = min(start + batch_size, len(dataset))
    return default_collate([dataset[index] for index in range(start, stop)])


def encode_tensor(tensor: torch.Tensor, exchange_format: str):
    if exchange_format == "native":
        return tensor
    return tensor.detach().numpy()


def decode_tensor(value, exchange_format: str) -> torch.Tensor:
    if exchange_format == "native":
        return value
    return torch.from_numpy(value)


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
        "steps_per_second": len(samples) / sum(totals),
        "mean_forward_call_seconds": mean([sample["forward_call_seconds"] for sample in samples]),
        "mean_backward_call_seconds": mean([sample["backward_call_seconds"] for sample in samples]),
        "mean_server_transition_seconds": mean(
            [
                sample["server_activation_transition_seconds"] + sample["server_gradient_transition_seconds"]
                for sample in samples
            ]
        ),
        "mean_client_transition_seconds": mean(
            [
                sample["client_activation_transition_seconds"] + sample["client_gradient_transition_seconds"]
                for sample in samples
            ]
        ),
        "mean_compute_seconds": mean([sample["server_compute_seconds"] for sample in samples]),
    }


@collab.init
def initialize_client():
    global _bottom_model, _bottom_optimizer, _client_data, _config
    if collab.site_name == "server":
        return

    torch.manual_seed(0)
    _config = collab.get_app_prop("benchmark_config")
    _client_data = load_mnist(_config["data_root"])
    _bottom_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, _config["hidden_dim"]),
        nn.ReLU(),
    )
    _bottom_optimizer = optim.SGD(_bottom_model.parameters(), lr=_config["learning_rate"])


@collab.publish
def forward(step):
    global _activations
    if _bottom_model is None:
        raise RuntimeError("split-learning benchmark client is not initialized")

    compute_start = time.perf_counter()
    images, _ = get_batch(_client_data, step, _config["batch_size"])
    _activations = _bottom_model(images)
    compute_seconds = time.perf_counter() - compute_start

    transition_start = time.perf_counter()
    payload = encode_tensor(_activations.detach(), _config["exchange_format"])
    transition_seconds = time.perf_counter() - transition_start
    return {
        "activations": payload,
        "compute_seconds": compute_seconds,
        "transition_seconds": transition_seconds,
    }


@collab.publish
def backward(gradients):
    if _bottom_optimizer is None or _activations is None:
        raise RuntimeError("backward() called before a successful forward()")

    transition_start = time.perf_counter()
    decoded_gradients = decode_tensor(gradients, _config["exchange_format"])
    transition_seconds = time.perf_counter() - transition_start

    compute_start = time.perf_counter()
    _bottom_optimizer.zero_grad(set_to_none=True)
    _activations.backward(decoded_gradients)
    _bottom_optimizer.step()
    compute_seconds = time.perf_counter() - compute_start
    return {
        "compute_seconds": compute_seconds,
        "transition_seconds": transition_seconds,
    }


@collab.main
def run_benchmark():
    config = collab.get_app_prop("benchmark_config")
    exchange_format = config["exchange_format"]
    metrics_file = Path(config["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    server_data = load_mnist(config["data_root"])
    top_model = nn.Linear(config["hidden_dim"], 10)
    top_optimizer = optim.SGD(top_model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    client = collab.clients[0](timeout=config["call_timeout"])
    samples = []
    total_steps = config["warmup_steps"] + config["num_steps"]

    for step in range(total_steps):
        total_start = time.perf_counter()
        _, labels = get_batch(server_data, step, config["batch_size"])

        call_start = time.perf_counter()
        forward_result = client.forward(step)
        forward_call_seconds = time.perf_counter() - call_start

        transition_start = time.perf_counter()
        activations = decode_tensor(forward_result["activations"], exchange_format).requires_grad_(True)
        server_activation_transition_seconds = time.perf_counter() - transition_start

        compute_start = time.perf_counter()
        top_optimizer.zero_grad(set_to_none=True)
        logits = top_model(activations)
        loss = criterion(logits, labels)
        loss.backward()
        server_compute_seconds = time.perf_counter() - compute_start

        transition_start = time.perf_counter()
        gradients = encode_tensor(activations.grad, exchange_format)
        server_gradient_transition_seconds = time.perf_counter() - transition_start

        call_start = time.perf_counter()
        backward_result = client.backward(gradients)
        backward_call_seconds = time.perf_counter() - call_start
        top_optimizer.step()

        if step >= config["warmup_steps"]:
            sample = {
                "step": step - config["warmup_steps"] + 1,
                "loss": float(loss.detach()),
                "activation_bytes": activations.numel() * activations.element_size(),
                "forward_call_seconds": forward_call_seconds,
                "backward_call_seconds": backward_call_seconds,
                "server_activation_transition_seconds": server_activation_transition_seconds,
                "server_gradient_transition_seconds": server_gradient_transition_seconds,
                "server_compute_seconds": server_compute_seconds,
                "client_forward_compute_seconds": forward_result["compute_seconds"],
                "client_backward_compute_seconds": backward_result["compute_seconds"],
                "client_activation_transition_seconds": forward_result["transition_seconds"],
                "client_gradient_transition_seconds": backward_result["transition_seconds"],
                "total_seconds": time.perf_counter() - total_start,
            }
            samples.append(sample)

        completed = step + 1
        if completed == total_steps or completed % config["log_every"] == 0:
            print(f"[{exchange_format}] step {completed}/{total_steps}: loss={float(loss.detach()):.4f}")

    metrics = {
        "workload": "simple_split_learning",
        "exchange_format": exchange_format,
        "config": config,
        "environment": environment_info(),
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
        job_name=f"benchmark_split_learning_{config['exchange_format']}",
        min_clients=1,
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
    mnist_dir = Path(config["data_root"]) / "MNIST" / "raw"
    if not mnist_dir.is_dir():
        raise FileNotFoundError(f"MNIST is not prepared under {config['data_root']}; run prepare_data.py first")
    simple_logging()
    run = make_recipe(config).execute(SimEnv(num_clients=1, workspace_root=args.workspace_root))
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
