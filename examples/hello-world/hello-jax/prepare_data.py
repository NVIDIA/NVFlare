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

"""
Prepare MNIST .npy files for hello-jax without requiring TensorFlow.
"""

import argparse
import gzip
import os
import struct
import urllib.request

import numpy as np

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
DEFAULT_DATA_DIR = "/tmp/nvflare/data/hello-jax/mnist"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    return parser.parse_args()


def _download_if_missing(data_dir: str, filename: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(f"{MNIST_BASE_URL}/{filename}", file_path)
    return file_path


def _read_images(file_path: str) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols, 1)


def _read_labels(file_path: str) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_labels)


def main():
    args = parse_args()
    raw_dir = os.path.join(args.data_dir, "raw")
    os.makedirs(args.data_dir, exist_ok=True)

    train_images_path = _download_if_missing(raw_dir, MNIST_FILES["train_images"])
    train_labels_path = _download_if_missing(raw_dir, MNIST_FILES["train_labels"])
    test_images_path = _download_if_missing(raw_dir, MNIST_FILES["test_images"])
    test_labels_path = _download_if_missing(raw_dir, MNIST_FILES["test_labels"])

    np.save(os.path.join(args.data_dir, "train_images.npy"), _read_images(train_images_path))
    np.save(os.path.join(args.data_dir, "train_labels.npy"), _read_labels(train_labels_path))
    np.save(os.path.join(args.data_dir, "test_images.npy"), _read_images(test_images_path))
    np.save(os.path.join(args.data_dir, "test_labels.npy"), _read_labels(test_labels_path))


if __name__ == "__main__":
    main()
