# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import time

import _pickle
import numpy as np
import tensorflow as tf
from filelock import FileLock, Timeout
from tensorflow.keras import datasets


def load_cifar10_with_retry(max_retries=3, retry_delay=5):
    """
    Load CIFAR10 dataset with retry mechanism and proper error handling.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (train_images, train_labels), (test_images, test_labels)
    """
    lock_path = os.path.join("/tmp", "cifar10_download.lock")
    lock = FileLock(lock_path)

    for attempt in range(max_retries):
        try:
            with lock:
                # Clear any existing corrupted downloads
                if attempt > 0:
                    cache_dir = os.path.expanduser("~/.keras/datasets")
                    cifar10_path = os.path.join(cache_dir, "cifar-10-batches-py")
                    if os.path.exists(cifar10_path):
                        import shutil

                        shutil.rmtree(cifar10_path)

                # Load the dataset
                return datasets.cifar10.load_data()

        except (Timeout, _pickle.UnpicklingError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to load CIFAR10 dataset after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    raise RuntimeError("Failed to load CIFAR10 dataset")


def cifar10_split(split_dir: str = None, num_sites: int = 8, alpha: float = 0.5, seed: int = 0):
    """
    Partition CIFAR-10 dataset into multiple sites using Dirichlet sampling.

    This Dirichlet sampling strategy for creating a heterogeneous partition is adopted
    from FedMA (https://github.com/IBM/FedMA).

    Args:
        split_dir: Directory to save split indices
        num_sites: Number of client sites
        alpha: Dirichlet sampling parameter (lower = more heterogeneous)
        seed: Random seed

    Returns:
        List of paths to .npy files containing training indices for each site
    """
    if split_dir is None:
        raise ValueError("You need to define a valid `split_dir` for splitting the data.")
    if not os.path.isabs(split_dir):
        raise ValueError("`split_dir` needs to be absolute path.")
    if alpha < 0.0:
        raise ValueError(f"Alpha should be larger or equal 0.0 but was {alpha}!")

    np.random.seed(seed)

    train_idx_paths = []

    print(f"Partition CIFAR-10 dataset into {num_sites} sites with Dirichlet sampling under alpha {alpha}")
    site_idx, class_sum = _partition_data(num_sites, alpha)

    # write to files
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    sum_file_name = os.path.join(split_dir, "summary.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write(f"Number of clients: {num_sites} \n")
        sum_file.write(f"Dirichlet sampling parameter: {alpha} \n")
        sum_file.write("Class counts for each client: \n")
        sum_file.write(json.dumps(class_sum))

    # save site data files
    site_file_path = os.path.join(split_dir, "site-")
    for site in range(num_sites):
        site_file_name = site_file_path + str(site + 1) + ".npy"
        print(f"Save split index {site + 1} of {num_sites} to {site_file_name}")
        np.save(site_file_name, np.array(site_idx[site]))
        train_idx_paths.append(site_file_name)

    print("\nData splitting completed successfully!")
    print("\nClass distribution summary:")
    for site, classes in class_sum.items():
        total_samples = sum(classes.values())
        print(f"  Site {site + 1}: {total_samples} samples - {classes}")

    print(f"Split data saved to: {split_dir}")
    return train_idx_paths


def _get_site_class_summary(train_label, site_idx):
    """Get class distribution summary for each site."""
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


def _partition_data(num_sites, alpha):
    """Partition data using Dirichlet sampling."""
    # only training label is needed for doing split
    (train_images, train_labels), (test_images, test_labels) = load_cifar10_with_retry()

    min_size = 0
    K = 10
    N = train_labels.shape[0]
    site_idx = {}

    # split
    while min_size < 10:
        idx_batch = [[] for _ in range(num_sites)]
        # for each class in the dataset
        for k in range(K):
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
    class_sum = _get_site_class_summary(train_labels, site_idx)

    return site_idx, class_sum


def preprocess_dataset(dataset, is_training, batch_size=1):
    """
    Apply pre-processing transformations to CIFAR10 dataset.

    Same pre-processings are used as in the PyTorch tutorial
    on CIFAR10: https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/cifar10-sim

    Training time pre-processings are (in-order):
    - Image padding with 4 pixels in "reflect" mode on each side
    - RandomCrop of 32 x 32 images
    - RandomHorizontalFlip
    - Normalize to [0, 1]: dividing pixels values by given CIFAR10 data mean & std
    - Random shuffle

    Testing/Validation time pre-processings are:
    - Normalize: dividing pixels values by 255

    Args:
        dataset: tf.data.Dataset - Tensorflow Dataset
        is_training: bool - Boolean flag indicating if current phase is training phase
        batch_size: int - Batch size

    Returns:
        tf.data.Dataset - Tensorflow Dataset with pre-processings applied
    """
    # Values from: https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/cifar10/pt/learners/cifar10_model_learner.py#L147
    mean_cifar10 = tf.constant([125.3, 123.0, 113.9], dtype=tf.float32)
    std_cifar10 = tf.constant([63.0, 62.1, 66.7], dtype=tf.float32)

    if is_training:
        # Padding each dimension by 4 pixels each side
        dataset = dataset.map(
            lambda image, label: (
                tf.stack(
                    [
                        tf.pad(tf.squeeze(t, [2]), [[4, 4], [4, 4]], mode="REFLECT")
                        for t in tf.split(image, num_or_size_splits=3, axis=2)
                    ],
                    axis=2,
                ),
                label,
            )
        )
        # Random crop of 32 x 32 x 3
        dataset = dataset.map(lambda image, label: (tf.image.random_crop(image, size=(32, 32, 3)), label))
        # Random horizontal flip
        dataset = dataset.map(lambda image, label: (tf.image.random_flip_left_right(image), label))
        # Normalize by dividing by given mean & std
        dataset = dataset.map(lambda image, label: ((tf.cast(image, tf.float32) - mean_cifar10) / std_cifar10, label))
        # Random shuffle
        dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True)
        # Convert to batches
        return dataset.batch(batch_size)
    else:
        # For validation / test only do normalization
        return dataset.map(
            lambda image, label: ((tf.cast(image, tf.float32) - mean_cifar10) / std_cifar10, label)
        ).batch(batch_size)


def load_site_data(site_name, train_idx_root="/tmp/cifar10_splits"):
    """
    Load CIFAR10 data for a specific site.

    Args:
        site_name: Name of the site (e.g., "site-1")
        train_idx_root: Root directory containing the data split indices

    Returns:
        train_images, train_labels, test_images, test_labels
    """
    (train_images, train_labels), (test_images, test_labels) = load_cifar10_with_retry()

    # Load site-specific training data if split file exists
    train_idx_path = os.path.join(train_idx_root, f"{site_name}.npy")

    if os.path.exists(train_idx_path):
        print(f"Loading train indices from {train_idx_path}")
        train_idx = np.load(train_idx_path)
        train_images = train_images[train_idx]
        train_labels = train_labels[train_idx]

        unq, unq_cnt = np.unique(train_labels, return_counts=True)
        print(
            f"Loaded {len(train_idx)} training indices from {train_idx_path} "
            f"with label distribution:\nUnique labels: {unq}\nUnique Counts: {unq_cnt}"
        )
    else:
        print(f"Warning: Split file {train_idx_path} not found. Using full training dataset.")

    return train_images, train_labels, test_images, test_labels
