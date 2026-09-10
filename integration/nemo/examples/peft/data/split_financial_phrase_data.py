# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# This Dirichlet sampling strategy for creating a heterogeneous partition is adopted
# from FedMA (https://github.com/IBM/FedMA).

# MIT License

# Copyright (c) 2020 International Business Machines

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import glob
import hashlib
import json
import os
from pprint import pprint

import numpy as np
import pandas as pd


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sentences(path):
    if not path:
        return set()
    data = pd.read_json(path, lines=True)
    return {str(sentence).strip() for sentence in data["sentence"]}


def clean_files(data_root, ext):
    files = glob.glob(os.path.join(data_root, "**", ext), recursive=True)
    for file in files:
        os.remove(file)


def clean_memmap(data_root):
    clean_files(data_root, "*.npy")
    clean_files(data_root, "*.info")


def get_site_class_summary(train_labels, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_labels[data_idx], return_counts=True)
        tmp = {unq[i]: int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[f"site-{site + 1}"] = tmp
    return class_sum


def write_class_summary(path, num_clients, alpha, class_sum):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as sum_file:
        sum_file.write(f"Number of clients: {num_clients} \n")
        sum_file.write(f"Dirichlet sampling parameter: {alpha} \n")
        sum_file.write("Class counts for each client: \n")
        sum_file.write(json.dumps(class_sum))


def partition_data(train_labels, label_names, num_sites, alpha, sum_file_name: str = None):
    min_size = 0
    N = len(train_labels)
    site_idx = {}
    train_labels = np.asarray(train_labels)

    # split
    while min_size < 10:
        idx_batch = [[] for _ in range(num_sites)]
        # for each class in the dataset
        for k in label_names:
            idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_sites))
            # Balance
            proportions = np.array([p * (len(idx_j) < N / num_sites) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # shuffle
    for j in range(num_sites):
        np.random.shuffle(idx_batch[j])
        site_idx[j] = idx_batch[j]

    # collect class summary
    class_sum = get_site_class_summary(train_labels, site_idx)

    # write summary info
    if sum_file_name:
        split_dir = os.path.dirname(sum_file_name)
        if not os.path.isdir(split_dir):
            os.makedirs(split_dir)
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {num_sites} \n")
            sum_file.write(f"Dirichlet sampling parameter: {alpha} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

    return site_idx, class_sum


def array_to_dataframe(data_array):
    data_dict = {"sentence": [], "label": []}
    for p in data_array:
        p = p.tolist()
        data_dict["sentence"].append(p[-2])
        data_dict["label"].append(p[-1])

    return pd.DataFrame(data_dict)


def split_data(
    data_path,
    out_dir,
    num_clients,
    site_name_prefix,
    seed,
    alpha,
    validation_path=None,
    test_path=None,
    remove_train_overlap=False,
):
    np.random.seed(seed)

    # use pandas to read jsonl format
    train_data = pd.read_json(data_path, lines=True)
    assert len(train_data) > 0, f"No data loaded from {data_path}"
    print(f"Loaded training data with {len(train_data)} entries")

    os.makedirs(out_dir, exist_ok=True)
    validation_sentences = load_sentences(validation_path)
    test_sentences = load_sentences(test_path)
    reserved_sentences = validation_sentences | test_sentences
    normalized_train_sentences = train_data["sentence"].astype(str).str.strip()
    overlap_mask = normalized_train_sentences.isin(reserved_sentences)
    removed_rows = int(overlap_mask.sum())
    removed_sentences = sorted(set(normalized_train_sentences[overlap_mask]))
    if removed_rows:
        if not remove_train_overlap:
            raise ValueError(
                "Sentence-level dataset split overlap: "
                f"train contains {removed_rows} rows for {len(removed_sentences)} validation/test sentences; "
                f"examples={removed_sentences[:3]}"
            )
        train_data = train_data.loc[~overlap_mask].copy()
        print(f"Removed {removed_rows} training rows overlapping validation/test sentences.")

    prepared_train_path = os.path.join(out_dir, "financial_phrase_bank_train_disjoint.jsonl")
    train_data.to_json(prepared_train_path, orient="records", lines=True)

    # shuffle the data
    train_data = train_data.sample(frac=1, random_state=seed)

    train_labels = train_data["label"]

    label_names = [" negative", " neutral", " positive"]

    # Clean NeMo memmap data before running a new data split
    clean_memmap(out_dir)

    sentence_groups = {}
    for row_idx, row in enumerate(train_data[["sentence", "label"]].itertuples(index=False)):
        sentence_groups.setdefault(str(row.sentence).strip(), []).append(row_idx)
    group_labels = []
    group_rows = []
    for sentence, row_indices in sentence_groups.items():
        labels = {str(train_labels.iloc[row_idx]) for row_idx in row_indices}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for duplicate training sentence {sentence!r}: {sorted(labels)}")
        group_labels.append(next(iter(labels)))
        group_rows.append(row_indices)

    group_site_idx, _ = partition_data(
        group_labels,
        label_names,
        num_clients,
        alpha=alpha,
        sum_file_name=None,
    )
    site_idx = {
        site: [row_idx for group_idx in group_indices for row_idx in group_rows[group_idx]]
        for site, group_indices in group_site_idx.items()
    }
    class_sum = get_site_class_summary(np.asarray(train_labels), site_idx)
    write_class_summary(os.path.join(out_dir, f"summary_alpha{alpha}.txt"), num_clients, alpha, class_sum)
    print(f"After split Dirichlet sampling with alpha={alpha}")
    pprint(class_sum)

    train_data = np.asarray(train_data)
    output_files = []
    for idx in range(num_clients):
        train_indices = site_idx[idx]
        split = train_data[train_indices]

        df = array_to_dataframe(split)

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, f"alpha{alpha}_{site_name_prefix}{idx + 1}.jsonl")

        df.to_json(out_file, orient="records", lines=True)
        print(f"Save split {idx + 1} of {num_clients} with {len(split)} entries to {out_file}")
        output_files.append(out_file)

    split_sentence_sets = [load_sentences(path) for path in output_files]
    for left in range(len(split_sentence_sets)):
        for right in range(left + 1, len(split_sentence_sets)):
            overlap = split_sentence_sets[left] & split_sentence_sets[right]
            if overlap:
                raise ValueError(
                    f"Sentence overlap between site-{left + 1} and site-{right + 1}: {sorted(overlap)[:3]}"
                )
    train_sentences = set().union(*split_sentence_sets)
    split_pairs = {
        "train_validation": train_sentences & validation_sentences,
        "train_test": train_sentences & test_sentences,
        "validation_test": validation_sentences & test_sentences,
    }
    overlaps = {name: sorted(values)[:3] for name, values in split_pairs.items() if values}
    if overlaps:
        raise ValueError(f"Sentence-level dataset split overlap: {overlaps}")

    files = {"source_train": data_path, "prepared_train": prepared_train_path}
    files.update({f"site-{index + 1}": path for index, path in enumerate(output_files)})
    if validation_path:
        files["validation"] = validation_path
    if test_path:
        files["test"] = test_path
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "alpha": alpha,
        "num_clients": num_clients,
        "sentence_level_disjoint": True,
        "overlap_resolution": {
            "policy": "remove_from_train" if remove_train_overlap else "reject",
            "removed_rows": removed_rows,
            "removed_unique_sentences": len(removed_sentences),
        },
        "files": {},
    }
    for name, path in files.items():
        frame = pd.read_json(path, lines=True)
        manifest["files"][name] = {
            "path": os.path.abspath(path),
            "sha256": file_sha256(path),
            "rows": len(frame),
            "class_counts": {str(key).strip(): int(value) for key, value in frame["label"].value_counts().items()},
        }
    manifest_path = os.path.join(out_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Saved reproducibility manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data split for dataset")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--out_dir", type=str, help="Path to output directory", default=".")
    parser.add_argument("--num_clients", type=int, help="Total number of clients", default=3)
    parser.add_argument("--random_seed", type=int, help="Random seed", default=0)
    parser.add_argument("--site_name_prefix", type=str, help="Site name prefix", default="site-")
    parser.add_argument("--validation_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument(
        "--remove_train_overlap",
        action="store_true",
        help="Remove training rows whose sentence occurs in validation or test; validation and test remain unchanged.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Alpha value to control the Dirichlet sampling strategy for creating a heterogeneous partition. "
        "Smaller values of alpha cause higher heterogeneity.",
        default=10.0,
    )
    args = parser.parse_args()

    split_data(
        data_path=args.data_path,
        out_dir=args.out_dir,
        num_clients=args.num_clients,
        site_name_prefix=args.site_name_prefix,
        seed=args.random_seed,
        alpha=args.alpha,
        validation_path=args.validation_path,
        test_path=args.test_path,
        remove_train_overlap=args.remove_train_overlap,
    )
