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
import numpy as np
import os
from pathlib import Path
import sys
import tempfile
import torch

from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import LoadImageD, EnsureChannelFirstD, ScaleIntensityD, Compose

# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

print_config()

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

max_epochs = 5
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

# (optional) calculate total steps
steps = max_epochs * len(train_loader)
# Run the training

summary_writer = SummaryWriter()
while flare.is_running():
    # (3) receives FLModel from NVFlare
    input_model = flare.receive()
    print(f"current_round={input_model.current_round}")

    # (4) loads model from NVFlare
    trainer.model.load_state_dict(input_model.params)

    trainer.run()

    # (5) wraps evaluation logic into a method to re-use for
    #       evaluation on both trained and received model
    def evaluate(input_weights):
        model.load_state_dict(input_weights)

        # Check the prediction on the test dataset
        dataset_dir = Path(root_dir, "MedNIST")
        class_names = sorted(f"{x.name}" for x in dataset_dir.iterdir() if x.is_dir())
        testdata = MedNISTDataset(root_dir=root_dir, transform=transform, section="test", download=False,
                                  runtime_cache=True)
        correct = 0
        total = 0
        with eval_mode(model):
            for item in DataLoader(testdata, batch_size=1, num_workers=0):
                prob = np.array(model(item["image"].to(DEVICE)).detach().to("cpu"))[0]
                pred = class_names[prob.argmax()]
                gt = item["class_name"][0]
                print(f"Class prediction is {pred}. Ground-truth: {gt}")

                # the class with the highest energy is what we choose as prediction
                total += gt.size(0)
                correct += (pred == gt).sum().item()

        print(f"Accuracy of the network on the {total} test images: {100 * correct // total} %")
        return 100 * correct // total

    # (6) evaluate on received model for model selection
    accuracy = evaluate(input_model.params)
    summary_writer.add_scalar(tag="global_model_accuracy", scalar=accuracy, global_step=input_model.current_round)

    # (7) construct trained FL model
    output_model = flare.FLModel(
        params=model.cpu().state_dict(),
        metrics={"accuracy": accuracy},
        meta={"NUM_STEPS_CURRENT_ROUND": steps},
    )
    # (8) send model back to NVFlare
    flare.send(output_model)


