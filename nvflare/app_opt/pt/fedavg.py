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

from typing import Optional, Union

import torch

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs


class PTFedAvg(FedAvg):
    """PyTorch FedAvg Controller with Early Stopping and Model Selection.

    This is a PyTorch-specific wrapper around the generic FedAvg controller.
    It adds PyTorch-specific model serialization using torch.save/torch.load.

    The FedAvg controller includes:
    - InTime (streaming) aggregation for memory efficiency
    - Early stopping support
    - Best model selection and saving
    - Custom aggregator support

    Args:
        num_clients (int, optional): The number of clients. Defaults to 3.
        num_rounds (int, optional): The total number of training rounds. Defaults to 5.
        stop_cond (str, optional): Early stopping condition based on metric. String
            literal in the format of '<key> <op> <value>' (e.g. "accuracy >= 80").
            If None, early stopping is disabled.
        patience (int, optional): The number of rounds with no improvement after which
            FL will be stopped. Only applies if stop_cond is set. Defaults to None.
        task_name (str, optional): Task name for training. Defaults to "train".
        save_filename (str, optional): Filename for saving the best model.
            Defaults to "FL_global_model.pt".
        initial_model (nn.Module, optional): Initial PyTorch model. Can be an nn.Module
            (will call .state_dict()) or a dict of parameters.

    Example:
        ```python
        from model import Net
        from nvflare import FedJob
        from nvflare.app_opt.pt.fedavg import PTFedAvg

        job = FedJob(name="pt_fedavg")
        controller = PTFedAvg(
            num_clients=2,
            num_rounds=10,
            stop_cond="accuracy >= 80",
            patience=3,
            initial_model=Net(),
        )
        job.to(controller, "server")
        ```
    """

    def __init__(
        self,
        *args,
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        task_name: Optional[str] = "train",
        save_filename: Optional[str] = "FL_global_model.pt",
        initial_model: Optional[Union[torch.nn.Module, dict, FLModel]] = None,
        **kwargs,
    ) -> None:
        # Convert PyTorch model to dict if needed
        if initial_model is None:
            initial_model_params = None
        elif isinstance(initial_model, torch.nn.Module):
            initial_model_params = initial_model.state_dict()
        elif isinstance(initial_model, dict):
            initial_model_params = initial_model
        elif isinstance(initial_model, FLModel):
            initial_model_params = initial_model
        else:
            raise TypeError(
                f"initial_model must be torch.nn.Module, dict, FLModel, or None, "
                f"but got {type(initial_model).__name__}"
            )

        super().__init__(
            *args,
            initial_model=initial_model_params,
            save_filename=save_filename,
            stop_cond=stop_cond,
            patience=patience,
            task_name=task_name,
            **kwargs,
        )

    def run(self) -> None:
        """Run FedAvg workflow with PyTorch tensor serialization support."""
        # Register TensorDecomposer for FOBS serialization of PyTorch tensors
        fobs.register(TensorDecomposer)
        super().run()

    def save_model_file(self, model: FLModel, filepath: str) -> None:
        """Save model using PyTorch's torch.save.

        Saves parameters via torch.save and FLModel metadata via FOBS.

        Args:
            model (FLModel): model to save
            filepath (str): path to save the model
        """
        # Save parameters with torch.save
        torch.save(model.params, filepath)

        # Save FLModel metadata (metrics, params_type, etc.) separately
        # Save FLModel metadata (metrics, params_type, etc.) separately
        params = model.params
        try:
            model.params = {}  # Temporarily remove params to save only metadata
            fobs.dumpf(model, f"{filepath}.metadata")
        finally:
            model.params = params  # Restore params

    def load_model_file(self, filepath: str) -> FLModel:
        """Load model using PyTorch's torch.load.

        Loads parameters via torch.load and FLModel metadata via FOBS.

        Args:
            filepath (str): path to load the model from

        Returns:
            FLModel: loaded model with params and metadata
        """
        import os

        # Load parameters with torch.load
        params = torch.load(filepath, weights_only=True)

        # Load FLModel metadata if exists
        metadata_path = f"{filepath}.metadata"
        if os.path.exists(metadata_path):
            model: FLModel = fobs.loadf(metadata_path)
            model.params = params
        else:
            model = FLModel(params=params)

        return model


# Backward compatibility alias
PTFedAvgEarlyStopping = PTFedAvg
