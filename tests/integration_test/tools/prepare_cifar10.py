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

import argparse
import shutil
from pathlib import Path

CACHE_ARCHIVE = Path("/tmp/nvf-test-data/cifar10/cifar-10-python.tar.gz")


def prepare_cifar10(root):
    from torchvision.datasets import CIFAR10

    root = Path(root).expanduser()
    if CACHE_ARCHIVE.is_file():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CACHE_ARCHIVE, root / CIFAR10.filename)
    # Torchvision validates and extracts the local archive, or downloads it when absent/invalid.
    CIFAR10(root=str(root), download=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CIFAR10 using the CI archive when available")
    parser.add_argument("--root", required=True)
    prepare_cifar10(parser.parse_args().root)
