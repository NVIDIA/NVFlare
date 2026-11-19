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

"""Reserve server proxy images, then partition the remaining CIFAR-10 training data."""

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision import datasets

from nvflare.utils.data.partition import dirichlet_label_partition

from ..data.cifar10_data_utils import CIFAR10_ROOT

DEFAULT_SPLIT_ROOT = "/tmp/cifar10_splits/pt_cifar10_feddf"


def prepare_splits(labels, split_root, num_clients=8, alpha=0.1, proxy_size=5000, seed=0):
    """Write disjoint proxy/client indices into a new directory, never the test set.

    Both aggregation modes consume these exact client shards. Proxy labels are
    used only for preparation summaries; distillation does not receive labels.
    """
    labels = np.asarray(labels)
    if not 0 < proxy_size <= len(labels) - 10 * num_clients:
        raise ValueError("Reserve a positive proxy set and at least ten private samples per client")
    permutation = np.random.RandomState(seed).permutation(len(labels))
    proxy_indices, private_indices = permutation[:proxy_size], permutation[proxy_size:]
    partition = dirichlet_label_partition(labels[private_indices], num_sites=num_clients, alpha=alpha, seed=seed)
    root = Path(split_root)
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / "proxy.npy", proxy_indices)
    for site, relative_indices in partition.site_indices.items():
        np.save(root / f"site-{site + 1}.npy", private_indices[relative_indices])
    manifest = {
        "num_clients": num_clients,
        "alpha": alpha,
        "split_seed": seed,
        "proxy_size": proxy_size,
        "training_size": len(labels),
        "client_class_counts": partition.class_counts,
        "proxy_class_counts": np.bincount(labels[proxy_indices], minlength=10).tolist(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--proxy-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True)
    datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=True)
    manifest = prepare_splits(train.targets, args.split_root, args.num_clients, args.alpha, args.proxy_size, args.seed)
    print(json.dumps(manifest, indent=2))
    print(f"Prepared proxy and client indices at {args.split_root}")


if __name__ == "__main__":
    main()
