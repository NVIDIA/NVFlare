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

"""Split the wikitext-2-raw-v1 training set among N simulated clients.

Run before starting the simulation:

    python prepare_data.py --n_clients 4

Output layout (default --output_dir /tmp/swarm_data):

    /tmp/swarm_data/
        site-1/train/          # Arrow dataset (HuggingFace load_from_disk)
        site-2/train/
        site-3/train/
        site-4/train/
        validation/            # shared validation split (all clients)

Each client receives a disjoint shard of the training split so the
simulation reflects real-world data silos.  The validation set is written
once and is not partitioned (useful for per-round evaluation).

The client.py script loads from these directories when --data_dir is
supplied, falling back to an in-memory shard if no --data_dir is given.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Partition wikitext-2 among N swarm clients.")
    parser.add_argument("--n_clients", type=int, default=4, help="Number of clients to create splits for")
    parser.add_argument(
        "--output_dir",
        default="/tmp/swarm_data",
        help="Root directory to write per-client splits (default: /tmp/swarm_data)",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional HuggingFace cache directory",
    )
    args = parser.parse_args()

    if args.n_clients < 1:
        raise ValueError("--n_clients must be >= 1")

    from datasets import load_dataset

    print(f"Loading wikitext-2-raw-v1 ...")
    train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=args.cache_dir)
    val_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", cache_dir=args.cache_dir)

    # Drop empty lines
    train_ds = train_ds.filter(lambda x: len(x["text"].strip()) > 0)
    val_ds = val_ds.filter(lambda x: len(x["text"].strip()) > 0)

    print(f"Train examples (after filtering): {len(train_ds):,}")
    print(f"Validation examples (after filtering): {len(val_ds):,}")
    print(f"Splitting into {args.n_clients} shards ...\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Write per-client training shards
    for i in range(args.n_clients):
        site_name = f"site-{i + 1}"
        shard = train_ds.shard(num_shards=args.n_clients, index=i)
        shard_dir = os.path.join(args.output_dir, site_name, "train")
        shard.save_to_disk(shard_dir)
        print(f"  {site_name}: {len(shard):,} examples → {shard_dir}")

    # Write shared validation split
    val_dir = os.path.join(args.output_dir, "validation")
    val_ds.save_to_disk(val_dir)
    print(f"\n  validation: {len(val_ds):,} examples → {val_dir}")

    print(f"\nData preparation complete. Pass --data_dir {args.output_dir} to client.py.")


if __name__ == "__main__":
    main()
