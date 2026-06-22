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

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from typing import Iterable

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is available on CI Linux/macOS.
    fcntl = None


DEFAULT_CACHE_ROOT = "/tmp/nvf-test-data"
CIFAR10_DIR_NAME = "cifar-10-batches-py"


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _cache_dataset_root(cache_root: str) -> str:
    return os.path.join(_normalize_path(cache_root), "cifar10")


@contextlib.contextmanager
def _file_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        if fcntl:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _is_valid_cifar10_root(root: str) -> bool:
    from torchvision.datasets import CIFAR10

    try:
        CIFAR10(root=root, train=True, download=False)
        CIFAR10(root=root, train=False, download=False)
    except RuntimeError:
        return False
    return True


def _download_cifar10(root: str):
    from torchvision.datasets import CIFAR10

    print(f"Preparing CIFAR-10 cache at {root}")
    CIFAR10(root=root, train=True, download=True)
    CIFAR10(root=root, train=False, download=True)

    if not _is_valid_cifar10_root(root):
        raise RuntimeError(f"CIFAR-10 cache at {root} failed integrity validation")


def _remove_stale_target(target_data_dir: str):
    if os.path.islink(target_data_dir) or os.path.isfile(target_data_dir):
        os.unlink(target_data_dir)
    elif os.path.isdir(target_data_dir):
        shutil.rmtree(target_data_dir)


def _link_or_copy_cache(cache_data_dir: str, root: str):
    root = _normalize_path(root)
    target_data_dir = os.path.join(root, CIFAR10_DIR_NAME)

    if os.path.realpath(target_data_dir) == os.path.realpath(cache_data_dir):
        return

    if os.path.exists(target_data_dir):
        if _is_valid_cifar10_root(root):
            print(f"CIFAR-10 data already exists at {root}")
            return
        _remove_stale_target(target_data_dir)
    elif os.path.islink(target_data_dir):
        os.unlink(target_data_dir)

    os.makedirs(root, exist_ok=True)
    try:
        os.symlink(cache_data_dir, target_data_dir)
        print(f"Linked CIFAR-10 data into {root}")
    except OSError:
        shutil.copytree(cache_data_dir, target_data_dir)
        print(f"Copied CIFAR-10 data into {root}")

    if not _is_valid_cifar10_root(root):
        raise RuntimeError(f"CIFAR-10 data at {root} failed integrity validation")


def prepare_cifar10(roots: Iterable[str] = (), cache_root: str | None = None):
    """Prepare CIFAR-10 once in a shared cache and expose it at each requested root."""

    cache_root = cache_root or os.environ.get("NVFLARE_TEST_DATA_CACHE", DEFAULT_CACHE_ROOT)
    cache_dataset_root = _cache_dataset_root(cache_root)
    lock_path = os.path.join(_normalize_path(cache_root), ".cifar10.lock")

    with _file_lock(lock_path):
        if not _is_valid_cifar10_root(cache_dataset_root):
            _download_cifar10(cache_dataset_root)
        else:
            print(f"Using existing CIFAR-10 cache at {cache_dataset_root}")

        cache_data_dir = os.path.join(cache_dataset_root, CIFAR10_DIR_NAME)
        for root in roots:
            _link_or_copy_cache(cache_data_dir, root)

    return cache_dataset_root


def main():
    parser = argparse.ArgumentParser(description="Prepare reusable CIFAR-10 data for integration tests")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Dataset root to populate. Can be specified more than once.",
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("NVFLARE_TEST_DATA_CACHE", DEFAULT_CACHE_ROOT),
        help=f"Shared cache root. Defaults to NVFLARE_TEST_DATA_CACHE or {DEFAULT_CACHE_ROOT}.",
    )
    args = parser.parse_args()

    prepare_cifar10(roots=args.root, cache_root=args.cache_root)


if __name__ == "__main__":
    main()
