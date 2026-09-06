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

"""Real-data robustness benchmark on scikit-learn's handwritten digits."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adaptive_hetero.policy import AdaptiveHeterogeneityPolicy, AdaptiveWeightingConfig

SETTINGS = {
    "mild": {"dirichlet_alpha": 0.50, "shift_scale": 0.10},
    "severe": {"dirichlet_alpha": 0.08, "shift_scale": 0.40},
    "extreme": {"dirichlet_alpha": 0.025, "shift_scale": 0.70},
}

NUM_FEATURES = 64
NUM_CLASSES = 10


def _make_model(model_name: str):
    if model_name == "linear":
        return torch.nn.Linear(NUM_FEATURES, NUM_CLASSES)
    if model_name == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(NUM_FEATURES, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, NUM_CLASSES),
        )
    raise ValueError(f"unknown model: {model_name}")


def _state_vector(model):
    return torch.cat([parameter.data.flatten() for parameter in model.parameters()])


def _load_state_vector(model, vector):
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.copy_(vector[offset : offset + size].view_as(parameter))
            offset += size


def _evaluate(model, data):
    features, labels = data
    with torch.no_grad():
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return accuracy, loss


def make_clients(seed, num_clients, dirichlet_alpha, shift_scale):
    dataset = load_digits()
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int64)
    features = StandardScaler().fit_transform(features).astype(np.float32)
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        proportions = 0.90 * proportions + 0.10 / num_clients
        proportions /= proportions.sum()
        cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(class_indices, cuts)):
            client_indices[client_id].extend(chunk.tolist())

    shifts = rng.normal(0.0, shift_scale, size=(num_clients, NUM_FEATURES)).astype(np.float32)
    shift_mask = np.zeros(NUM_FEATURES, dtype=np.float32)
    shift_mask[:16] = 1.0

    clients = []
    for client_id, indices in enumerate(client_indices):
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        client_features = features[indices].copy() + shifts[client_id] * shift_mask
        client_labels = labels[indices].copy()
        split = max(1, int(0.75 * len(indices)))
        if split >= len(indices):
            split = len(indices) - 1
        train = (
            torch.tensor(client_features[:split], dtype=torch.float32),
            torch.tensor(client_labels[:split], dtype=torch.long),
        )
        validation = (
            torch.tensor(client_features[split:], dtype=torch.float32),
            torch.tensor(client_labels[split:], dtype=torch.long),
        )
        counts = np.bincount(client_labels[:split], minlength=NUM_CLASSES).astype(np.float64)
        clients.append(
            {
                "train": train,
                "validation": validation,
                "descriptor": counts / counts.sum(),
            }
        )
    return clients


def _train_local(global_vector, model_name, client, learning_rate=0.12, local_steps=2):
    model = _make_model(model_name)
    _load_state_vector(model, global_vector)
    _, baseline_loss = _evaluate(model, client["validation"])
    features, labels = client["train"]
    for _ in range(local_steps):
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(features), labels)
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
    accuracy, final_loss = _evaluate(model, client["validation"])
    return _state_vector(model) - global_vector, accuracy, baseline_loss - final_loss


def run_method(method, model_name, clients, seed, rounds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = _make_model(model_name)
    global_vector = _state_vector(model).detach().clone()
    first_moment = torch.zeros_like(global_vector)
    second_moment = torch.zeros_like(global_vector)
    sample_counts = np.asarray([len(client["train"][1]) for client in clients], dtype=np.float64)
    descriptors = [client["descriptor"] for client in clients]
    policy = AdaptiveHeterogeneityPolicy(
        AdaptiveWeightingConfig(
            sample_exponent=0.65,
            representation_exponent=0.70,
            quality_exponent=0.40,
            fairness_strength=1.00,
            heterogeneity_threshold=0.26,
            heterogeneity_temperature=0.04,
            heterogeneity_deadband=0.15,
            max_blend_factor=0.40,
            min_weight=0.02,
            max_weight=0.30,
        )
    )

    mean_heterogeneity = 0.0
    blend_factor = 0.0
    final_weights = None
    beta1, beta2, epsilon, server_lr = 0.9, 0.99, 1e-8, 0.45
    for round_number in range(1, rounds + 1):
        updates = []
        local_metrics = []
        quality_improvements = []
        for client in clients:
            update, metric, improvement = _train_local(global_vector, model_name, client)
            updates.append(update)
            local_metrics.append(metric)
            quality_improvements.append(improvement)

        if method == "fedopt":
            weights = sample_counts / sample_counts.sum()
        elif method == "adaptive":
            result = policy.compute(
                sample_counts=sample_counts,
                descriptors=descriptors,
                client_metrics=local_metrics,
                quality_improvements=quality_improvements,
            )
            weights = result.weights
            mean_heterogeneity = result.mean_heterogeneity
            blend_factor = result.blend_factor
            final_weights = [float(value) for value in weights]
        else:
            raise ValueError(f"unknown method: {method}")

        mean_update = sum(float(weight) * update for weight, update in zip(weights, updates))
        first_moment = beta1 * first_moment + (1.0 - beta1) * mean_update
        second_moment = beta2 * second_moment + (1.0 - beta2) * mean_update.square()
        corrected_moment = first_moment / (1.0 - beta1**round_number)
        corrected_variance = second_moment / (1.0 - beta2**round_number)
        global_vector = global_vector + server_lr * corrected_moment / (torch.sqrt(corrected_variance) + epsilon)

    _load_state_vector(model, global_vector)
    client_accuracies = [_evaluate(model, client["validation"])[0] for client in clients]
    validation_sizes = [len(client["validation"][1]) for client in clients]
    return {
        "global_accuracy": float(np.average(client_accuracies, weights=validation_sizes)),
        "worst_client_accuracy": float(min(client_accuracies)),
        "client_accuracies": [float(value) for value in client_accuracies],
        "mean_heterogeneity": float(mean_heterogeneity),
        "blend_factor": float(blend_factor),
        "final_weights": final_weights,
    }


def summarize(runs):
    global_values = np.asarray([run["global_accuracy"] for run in runs], dtype=np.float64)
    worst_values = np.asarray([run["worst_client_accuracy"] for run in runs], dtype=np.float64)
    return {
        "global_accuracy_mean": float(global_values.mean()),
        "global_accuracy_std": float(global_values.std(ddof=1)) if len(global_values) > 1 else 0.0,
        "worst_client_accuracy_mean": float(worst_values.mean()),
        "worst_client_accuracy_std": float(worst_values.std(ddof=1)) if len(worst_values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31, 43, 57])
    parser.add_argument("--settings", choices=sorted(SETTINGS), nargs="+", default=list(SETTINGS))
    parser.add_argument("--models", choices=["linear", "mlp"], nargs="+", default=["linear", "mlp"])
    parser.add_argument("--methods", choices=["fedopt", "adaptive"], nargs="+", default=["fedopt", "adaptive"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    report = {
        "benchmark": "scikit-learn handwritten digits; external CPU benchmark, not an official NVFlare result",
        "config": {
            "clients": args.clients,
            "rounds": args.rounds,
            "seeds": args.seeds,
            "settings": args.settings,
            "models": args.models,
            "methods": args.methods,
        },
        "models": {},
    }

    for model_name in args.models:
        model_report = {}
        for setting_name in args.settings:
            setting = SETTINGS[setting_name]
            setting_report = {"parameters": setting, "runs": {}, "summary": {}}
            for method in args.methods:
                setting_report["runs"][method] = []
            for seed in args.seeds:
                clients = make_clients(seed=seed, num_clients=args.clients, **setting)
                for method in args.methods:
                    result = run_method(method, model_name, clients, seed, args.rounds)
                    result["seed"] = seed
                    setting_report["runs"][method].append(result)
            for method in args.methods:
                setting_report["summary"][method] = summarize(setting_report["runs"][method])
            model_report[setting_name] = setting_report
        report["models"][model_name] = model_report

    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
