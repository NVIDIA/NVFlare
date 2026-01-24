# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from model import CIFAR10DataModule, LitNet
from pytorch_lightning import Trainer, seed_everything

# (0) import nvflare lightning client API
import nvflare.client.lightning as flare

seed_everything(7)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)

    return parser.parse_args()


def main():
    args = define_parser()
    batch_size = args.batch_size

    model = LitNet()
    cifar10_dm = CIFAR10DataModule(batch_size=batch_size)
    trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1 if torch.cuda.is_available() else None)

    # (1) patch the lightning trainer
    flare.patch(trainer)

    # (2) evaluate the current global model to allow server-side model selection
    print("--- validate global model ---")
    trainer.validate(model, datamodule=cifar10_dm)


if __name__ == "__main__":
    main()
