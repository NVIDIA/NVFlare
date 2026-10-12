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

"""Class-free asynchronous MNIST training with nonblocking Collab calls.

Run from this directory with: python job.py
"""

import argparse
import json
import queue
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nvflare.collab import CollabRecipe, collab
from nvflare.recipe import SimEnv

_train_loader = None


def make_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))


def load_data(train):
    data = datasets.MNIST(collab.get_app_prop("data_root"), train=train, transform=transforms.ToTensor())
    if train:
        # Two disjoint, equally sized shards; only this site's shard is trained on.
        data = Subset(data, range(collab.get_app_prop("index"), len(data), 2))
    return DataLoader(data, batch_size=128 if train else 512, shuffle=train)


@collab.publish
def train(weights, version):
    global _train_loader
    if _train_loader is None:
        torch.set_num_threads(1)
        torch.manual_seed(collab.get_app_prop("index"))
        _train_loader = load_data(train=True)
    model = make_model()
    model.load_state_dict(weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for images, labels in _train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    time.sleep(collab.get_app_prop("delay"))  # simulate a slower site
    return model.state_dict(), version


def evaluate(model, loader):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return {"loss": loss / len(loader.dataset), "accuracy": correct / len(loader.dataset)}


@collab.main
def run():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    num_updates = collab.get_app_prop("num_updates")
    if num_updates < len(collab.clients):
        raise ValueError("num_updates must be at least the number of clients")
    model, test_loader = make_model(), load_data(train=False)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    history = [{"update": 0, "seconds": 0.0, **evaluate(model, test_loader)}]
    print(f"Initial test metrics: {history[0]}", flush=True)
    updates = queue.Queue()

    def accept_result(group_call_context, result):
        # Callbacks can run concurrently; only the main loop updates weights.
        updates.put((collab.caller, result))

    # Call options go BEFORE .train(). This returns while clients train.
    started = time.perf_counter()
    calls = [collab.clients(blocking=False, timeout=120, process_resp_cb=accept_result).train(weights, 0)]
    submitted = len(collab.clients)
    version = 0
    while version < num_updates:
        if collab.is_aborted:
            raise RuntimeError("Training aborted")
        # Failed calls do not invoke the success callback. Do not wait forever.
        for results in calls:
            if results.failures:
                raise RuntimeError(f"Client training failed: {results.failures}")
        try:
            site, (local_weights, base_version) = updates.get(timeout=0.1)
        except queue.Empty:
            continue

        staleness = version - base_version
        weights = {name: 0.5 * value + 0.5 * local_weights[name] for name, value in weights.items()}
        version += 1
        if submitted < num_updates:
            # Refill only the finished client. Others keep their original snapshots.
            calls.append(
                collab.get_clients([site])(blocking=False, timeout=120, process_resp_cb=accept_result).train(
                    weights, version
                )
            )
            submitted += 1

        # Evaluate after refill so the finished client can already train again.
        model.load_state_dict(weights)
        metrics = evaluate(model, test_loader)
        history.append({"update": version, "seconds": time.perf_counter() - started, **metrics})
        print(
            f"Update {version}/{num_updates} from {site}: base_version={base_version}, "
            f"staleness={staleness}, loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.2%}",
            flush=True,
        )

    # All submitted updates were accepted. Drain streams to finish the callbacks.
    for results in calls:
        for _ in results:
            pass
        if results.failures:
            raise RuntimeError(f"Client training failed: {results.failures}")
    elapsed = time.perf_counter() - started
    output = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
    torch.save(weights, output / "final_model.pt")
    (output / "metrics.json").write_text(json.dumps({"training_seconds": elapsed, "history": history}, indent=2))
    print(f"Training and evaluation finished in {elapsed:.2f}s. Results: {output}", flush=True)
    return weights


def make_recipe(args):
    if args.num_updates < 2:
        raise ValueError("num_updates must be >= 2")
    # Collab uses this module's decorated functions; no server/client classes.
    recipe = CollabRecipe(job_name="collab_simple_async", min_clients=2)
    recipe.set_server_prop("num_updates", args.num_updates)
    recipe.set_server_prop("data_root", args.data_root)
    recipe.set_client_prop("data_root", args.data_root)
    recipe.set_per_site_config({"site-1": {"index": 0, "delay": 0.0}, "site-2": {"index": 1, "delay": 1.0}})
    return recipe


def main():
    parser = argparse.ArgumentParser(description="Simple asynchronous MNIST training with Collab")
    parser.add_argument("--num-updates", type=int, default=20, help="Total accepted updates across both clients (>= 2)")
    parser.add_argument("--data-root", default="/tmp/nvflare/data")
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab/simple_async")
    args = parser.parse_args()
    if args.num_updates < 2:
        parser.error("--num-updates must be >= 2")
    # Download once before starting clients, avoiding concurrent downloads.
    for train_split in (True, False):
        datasets.MNIST(args.data_root, train=train_split, download=True)
    recipe = make_recipe(args)
    run = recipe.execute(SimEnv(clients=recipe.configured_sites(), workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
