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

import importlib

from nvflare.fuel.utils.import_utils import optional_import
from nvflare.recipe.spec import Recipe

TRACKING_REGISTRY = {
    "mlflow": {
        "package": "mlflow",
        "receiver_module": "nvflare.app_opt.tracking.mlflow.mlflow_receiver",
        "receiver_class": "MLflowReceiver",
    },
    "tensorboard": {
        "package": "tensorboard",
        "receiver_module": "nvflare.app_opt.tracking.tb.tb_receiver",
        "receiver_class": "TBAnalyticsReceiver",
    },
    "wandb": {
        "package": "wandb",
        "receiver_module": "nvflare.app_opt.tracking.wandb.wandb_receiver",
        "receiver_class": "WandBReceiver",
    },
}


def add_experiment_tracking(recipe: Recipe, tracking_type: str, tracking_config: dict = None):
    """Enable experiment tracking.

    Args:
        tracking_type: the type of tracking to enable
        tracking_config: the configuration for the tracking
    """
    tracking_config = tracking_config or {}
    if tracking_type not in TRACKING_REGISTRY:
        raise ValueError(f"Invalid tracking type: {tracking_type}")

    _, flag = optional_import(TRACKING_REGISTRY[tracking_type]["package"])
    if not flag:
        raise ValueError(
            f"{TRACKING_REGISTRY[tracking_type]['package']} is not installed. Please install it using `pip install {TRACKING_REGISTRY[tracking_type]['package']}`"
        )

    module = importlib.import_module(TRACKING_REGISTRY[tracking_type]["receiver_module"])
    receiver_class = getattr(module, TRACKING_REGISTRY[tracking_type]["receiver_class"])
    receiver = receiver_class(**tracking_config)
    recipe.job.to_server(receiver, "receiver")
