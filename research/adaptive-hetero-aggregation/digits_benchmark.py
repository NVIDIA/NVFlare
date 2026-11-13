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

"""Real-data robustness benchmark on scikit-learn handwritten digits."""

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


def _make_model(model_name):
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
        return (logits.argmax(dim=1) == labels).float().mean().item()


def make_clients(seed, num_clients, dirichlet_alpha, shift_scale):
    dataset = load_digits()
    features = StandardScaler().fit_transform(dataset.data.astype(np.float32)).astype(np.float32)
    labels = dataset.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for class_id in range(NUM_CLASSES):
        indices = np.where(labels == class_id)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        proportions = 0.90 * proportions + 0.10 / num_clients
        cuts = (np.cumsum(proportions / proportions.sum()) * len(indices)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(indices, cuts)):
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
        split = max(1, min(len(indices) - 1, int(0.75 * len(indices))))
        train_labels = client_labels[:split]
        counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
        clients.append(
            {
                "train": (
                    torch.tensor(client_features[:split], dtype=torch.float32),
                    torch.tensor(train_labels, dtype=torch.long),
                ),
                "validation": (
                    torch.tensor(client_features[split:], dtype=torch.float32),
                    torch.tensor(client_labels[split:], dtype=torch.long),
                ),
                "descriptor": counts / counts.sum(),
            }
        )
    return clients


def _train_local(global_vector, model_name, client, learning_rate=0.12, local_steps=2):
    model = _make_model(model_name)
    _load_state_vector(model, global_vector)
    # Match the real NVFlare client contract: fairness uses the received global
    # model's validation accuracy, not the locally trained model's accuracy.
    global_model_accuracy = _evaluate(model, client["validation"])
    features, labels = client["train"]
    for _ in range(local_steps):
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(features), labels)
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
    return _state_vector(model) - global_vector, global_model_accuracy


def run_method(method, model_name, clients, seed, rounds, participation_rate, cohort_hold_rounds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = _make_model(model_name)
    global_vector = _state_vector(model).detach().clone()
    first_moment = torch.zeros_like(global_vector)
    second_moment = torch.zeros_like(global_vector)
    policy = AdaptiveHeterogeneityPolicy(AdaptiveWeightingConfig(max_weight=0.30))
    rng = np.random.default_rng(seed + 2003)
    beta1, beta2, epsilon, server_lr = 0.9, 0.99, 1e-8, 0.45
    last_diag = None
    active_ids = None
    adaptive_rounds = 0
    max_observed_blend_factor = 0.0

    for round_number in range(1, rounds + 1):
        active_count = max(2, int(np.ceil(len(clients) * participation_rate)))
        if active_ids is None or (round_number - 1) % cohort_hold_rounds == 0:
            active_ids = sorted(rng.choice(len(clients), active_count, replace=False).tolist())
        updates, metrics, counts, descriptors = [], [], [], []
        for client_id in active_ids:
            client = clients[client_id]
            update, metric = _train_local(global_vector, model_name, client)
            updates.append(update)
            metrics.append(metric)
            counts.append(len(client["train"][1]))
            descriptors.append(client["descriptor"])
        counts = np.asarray(counts, dtype=np.float64)

        if method == "fedopt":
            weights = counts / counts.sum()
        elif method == "adaptive":
            last_diag = policy.compute(counts, descriptors, metrics, cohort_key=tuple(active_ids))
            weights = last_diag.weights
            if last_diag.blend_factor > 0.0:
                adaptive_rounds += 1
                max_observed_blend_factor = max(max_observed_blend_factor, float(last_diag.blend_factor))
        else:
            raise ValueError(f"unknown method: {method}")

        mean_update = sum(float(weight) * update for weight, update in zip(weights, updates))
        first_moment = beta1 * first_moment + (1.0 - beta1) * mean_update
        second_moment = beta2 * second_moment + (1.0 - beta2) * mean_update.square()
        corrected_moment = first_moment / (1.0 - beta1**round_number)
        corrected_variance = second_moment / (1.0 - beta2**round_number)
        global_vector += server_lr * corrected_moment / (torch.sqrt(corrected_variance) + epsilon)

    _load_state_vector(model, global_vector)
    accuracies = [_evaluate(model, client["validation"]) for client in clients]
    sizes = [len(client["validation"][1]) for client in clients]
    return {
        "global_accuracy": float(np.average(accuracies, weights=sizes)),
        "worst_client_accuracy": float(min(accuracies)),
        "client_accuracies": [float(value) for value in accuracies],
        "mean_heterogeneity": 0.0 if last_diag is None else last_diag.mean_heterogeneity,
        "raw_metric_gap": 0.0 if last_diag is None else last_diag.raw_metric_gap,
        "metric_gap": 0.0 if last_diag is None else last_diag.metric_gap,
        "candidate_blend_factor": (0.0 if last_diag is None else last_diag.candidate_blend_factor),
        "blend_factor": 0.0 if last_diag is None else last_diag.blend_factor,
        "activation_streak": 0 if last_diag is None else last_diag.activation_streak,
        "adaptive_rounds": adaptive_rounds,
        "max_observed_blend_factor": max_observed_blend_factor,
    }


def summarize(runs):
    global_values = np.asarray([run["global_accuracy"] for run in runs])
    worst_values = np.asarray([run["worst_client_accuracy"] for run in runs])
    return {
        "global_accuracy_mean": float(global_values.mean()),
        "global_accuracy_std": (float(global_values.std(ddof=1)) if len(runs) > 1 else 0.0),
        "worst_client_accuracy_mean": float(worst_values.mean()),
        "worst_client_accuracy_std": (float(worst_values.std(ddof=1)) if len(runs) > 1 else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument(
        "--cohort-hold-rounds",
        type=int,
        default=1,
        help="Keep a sampled participant cohort for this many consecutive rounds before resampling.",
    )
    parser.add_argument(
        "--require-adaptive-activation",
        action="store_true",
        help="Fail unless at least one adaptive benchmark run uses a non-zero blend factor.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31, 43, 57])
    parser.add_argument("--settings", choices=sorted(SETTINGS), nargs="+", default=list(SETTINGS))
    parser.add_argument("--models", choices=["linear", "mlp"], nargs="+", default=["linear", "mlp"])
    parser.add_argument(
        "--methods",
        choices=["fedopt", "adaptive"],
        nargs="+",
        default=["fedopt", "adaptive"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.participation_rate <= 1.0:
        raise ValueError("participation-rate must be in (0, 1]")
    if args.cohort_hold_rounds < 1:
        raise ValueError("cohort-hold-rounds must be at least 1")
    if args.require_adaptive_activation and "adaptive" not in args.methods:
        raise ValueError("require-adaptive-activation requires the adaptive method")

    torch.set_num_threads(2)
    report = {"config": vars(args).copy(), "models": {}}
    if args.output:
        report["config"]["output"] = str(args.output)
    for model_name in args.models:
        report["models"][model_name] = {}
        for setting_name in args.settings:
            setting_report = {
                "parameters": SETTINGS[setting_name],
                "runs": {},
                "summary": {},
            }
            for method in args.methods:
                setting_report["runs"][method] = []
            for seed in args.seeds:
                clients = make_clients(seed, args.clients, **SETTINGS[setting_name])
                for method in args.methods:
                    result = run_method(
                        method,
                        model_name,
                        clients,
                        seed,
                        args.rounds,
                        args.participation_rate,
                        args.cohort_hold_rounds,
                    )
                    result["seed"] = seed
                    setting_report["runs"][method].append(result)
            for method in args.methods:
                setting_report["summary"][method] = summarize(setting_report["runs"][method])
            report["models"][model_name][setting_name] = setting_report

    if args.require_adaptive_activation:
        adaptive_runs = [
            run
            for model_report in report["models"].values()
            for setting_report in model_report.values()
            for run in setting_report["runs"].get("adaptive", [])
        ]
        if not any(run["adaptive_rounds"] > 0 for run in adaptive_runs):
            raise RuntimeError("adaptive weighting never activated in the requested benchmark")

    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
