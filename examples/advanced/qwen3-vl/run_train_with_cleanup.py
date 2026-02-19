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
Thin wrapper that runs the Qwen3-VL train_qwen.train() and calls
torch.distributed.destroy_process_group() on exit to avoid the PyTorch warning
"destroy_process_group() was not called before program exit".
Run this with torchrun instead of train_qwen.py so the process group is torn down cleanly.
"""
import os
import sys


def main():
    # Ensure Qwen finetune package is importable; train_qwen.py uses "from trainer import ..."
    # so the qwenvl/train directory must be on the path.
    finetune_dir = os.environ.get("QWEN_FINETUNE_DIR")
    if finetune_dir:
        finetune_dir = os.path.abspath(finetune_dir)
        if finetune_dir not in sys.path:
            sys.path.insert(0, finetune_dir)
        train_dir = os.path.join(finetune_dir, "qwenvl", "train")
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)

    import torch

    try:
        from qwenvl.train import train_qwen

        train_qwen.train(attn_implementation="flash_attention_2")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
