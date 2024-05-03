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
#
# MONAI Example adopted from https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/monai_101.ipynb
#
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler, TensorBoardStatsHandler
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import Compose, EnsureChannelFirstD, LoadImageD, ScaleIntensityD

# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

print_config()


def main():
    # (2) initializes NVFlare client API
    flare.init()

    # Setup data directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    # Use MONAI transforms to preprocess data
    transform = Compose(
        [
            LoadImageD(keys="image", image_only=True),
            EnsureChannelFirstD(keys="image"),
            ScaleIntensityD(keys="image"),
        ]
    )

    # Prepare datasets using MONAI Apps
    dataset = MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=True)

    # Define a network and a supervised trainer

    # If available, we use GPU to speed things up.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    max_epochs = 1  # rather than 5 epochs, we run 5 FL rounds with 1 local epoch each.
    model = densenet121(spatial_dims=2, in_channels=1, out_channels=6).to(DEVICE)

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    trainer = SupervisedTrainer(
        device=torch.device(DEVICE),
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
        loss_function=torch.nn.CrossEntropyLoss(),
        inferer=SimpleInferer(),
        train_handlers=StatsHandler(),
    )

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    summary_writer = SummaryWriter()
    train_tensorboard_stats_handler = TensorBoardStatsHandler(summary_writer=summary_writer)
    train_tensorboard_stats_handler.attach(trainer)

    # (optional) calculate total steps
    steps = max_epochs * len(train_loader)
    # Run the training

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (4) loads model from NVFlare and sends it to GPU
        trainer.network.load_state_dict(input_model.params)
        trainer.network.to(DEVICE)

        trainer.run()

        # (5) wraps evaluation logic into a method to re-use for
        #       evaluation on both trained and received model
        def evaluate(input_weights):
            # Create model for evaluation
            eval_model = densenet121(spatial_dims=2, in_channels=1, out_channels=6).to(DEVICE)
            eval_model.load_state_dict(input_weights)

            # Check the prediction on the test dataset
            dataset_dir = Path(root_dir, "MedNIST")
            class_names = sorted(f"{x.name}" for x in dataset_dir.iterdir() if x.is_dir())
            testdata = MedNISTDataset(
                root_dir=root_dir, transform=transform, section="test", download=False, runtime_cache=True
            )
            correct = 0
            total = 0
            max_items_to_print = 10
            _print = 0
            with eval_mode(eval_model):
                for item in DataLoader(testdata, batch_size=512, num_workers=0):  # changed to do batch processing
                    prob = np.array(eval_model(item["image"].to(DEVICE)).detach().to("cpu"))
                    pred = [class_names[p] for p in prob.argmax(axis=1)]
                    gt = item["class_name"]
                    # changed the logic a bit from tutorial to compute accuracy on full test set
                    # but only print for some.
                    for _gt, _pred in zip(gt, pred):
                        if _print < max_items_to_print:
                            print(f"Class prediction is {_pred}. Ground-truth: {_gt}")
                            _print += 1

                        # compute accuracy
                        total += 1
                        correct += float(_pred == _gt)

            print(f"Accuracy of the network on the {total} test images: {100 * correct // total} %")
            return correct / total

        # (6) evaluate on received model for model selection
        accuracy = evaluate(input_model.params)
        summary_writer.add_scalar(tag="global_model_accuracy", scalar=accuracy, global_step=input_model.current_round)

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=trainer.network.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
