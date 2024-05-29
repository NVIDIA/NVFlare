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

import time
from typing import Union

import torch

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.security.logging import secure_format_exception


class FedOpt(FedAvg):
    def __init__(
        self,
        *args,
        source_model: Union[str, torch.nn.Module],
        optimizer_args: dict = {
            "path": "torch.optim.SGD",
            "args": {"lr": 1.0, "momentum": 0.6},
        },
        lr_scheduler_args: dict = {
            "path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "args": {"T_max": 3, "eta_min": 0.9},
        },
        device=None,
        **kwargs,
    ):
        """Implement the FedOpt algorithm. Based on FedAvg ModelController.

        The algorithm is proposed in Reddi, Sashank, et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).
        After each round, update the global model using the specified PyTorch optimizer and learning rate scheduler.
        Note: This class will use FedOpt to optimize the global trainable parameters (i.e. `self.torch_model.named_parameters()`)
        but use FedAvg to update any other layers such as batch norm statistics.

        Args:
            source_model: component id of torch model object or a valid torch model object
            optimizer_args: dictionary of optimizer arguments, with keys of 'optimizer_path' and 'args.
            lr_scheduler_args: dictionary of server-side learning rate scheduler arguments, with keys of 'lr_scheduler_path' and 'args.
            device: specify the device to run server-side optimization, e.g. "cpu" or "cuda:0"
                (will default to cuda if available and no device is specified).

        Raises:
            TypeError: when any of input arguments does not have correct type
        """
        super().__init__(*args, **kwargs)

        self.source_model = source_model
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.torch_model = None
        self.optimizer = None
        self.lr_scheduler = None

    def run(self):
        # set up source model
        if isinstance(self.source_model, str):
            self.torch_model = self.get_component(self.source_model)
        else:
            self.torch_model = self.source_model

        if self.torch_model is None:
            self.panic("Model is not available")
            return
        elif not isinstance(self.torch_model, torch.nn.Module):
            self.panic(f"expect model to be torch.nn.Module but got {type(self.torch_model)}")
            return
        else:
            print("server model", self.torch_model)
        self.torch_model.to(self.device)

        # set up optimizer
        try:
            if "args" not in self.optimizer_args:
                self.optimizer_args["args"] = {}
            self.optimizer_args["args"]["params"] = self.torch_model.parameters()
            self.optimizer = self.build_component(self.optimizer_args)
        except Exception as e:
            error_msg = f"Exception while constructing optimizer: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return

        # set up lr scheduler
        try:
            if "args" not in self.lr_scheduler_args:
                self.lr_scheduler_args["args"] = {}
            self.lr_scheduler_args["args"]["optimizer"] = self.optimizer
            self.lr_scheduler = self.build_component(self.lr_scheduler_args)
        except Exception as e:
            error_msg = f"Exception while constructing lr_scheduler: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return

        super().run()

    def optimizer_update(self, model_diff):
        """Updates the global model using the specified optimizer.

        Args:
            model_diff: the aggregated model differences from clients.

        Returns:
            The updated PyTorch model state dictionary.

        """
        self.torch_model.train()
        self.optimizer.zero_grad()

        # Apply the update to the model. We must multiply weights_delta by -1.0 to
        # view it as a gradient that should be applied to the server_optimizer.
        updated_params = []
        for name, param in self.torch_model.named_parameters():
            if name in model_diff:
                param.grad = torch.tensor(-1.0 * model_diff[name]).to(self.device)
                updated_params.append(name)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.torch_model.state_dict(), updated_params

    def update_model(self, global_model: FLModel, aggr_result: FLModel):
        model_diff = aggr_result.params

        start = time.time()
        weights, updated_params = self.optimizer_update(model_diff)
        secs = time.time() - start

        # convert to numpy dict of weights
        start = time.time()
        for key in weights:
            weights[key] = weights[key].detach().cpu().numpy()
        secs_detach = time.time() - start

        # update unnamed parameters such as batch norm layers if there are any using the averaged update
        n_fedavg = 0
        for key, value in model_diff.items():
            if key not in updated_params:
                weights[key] = global_model.params[key] + value
                n_fedavg += 1

        self.info(
            f"FedOpt ({type(self.optimizer)} {self.device}) server model update "
            f"round {self.current_round}, "
            f"{type(self.lr_scheduler)} "
            f"lr: {self.optimizer.param_groups[-1]['lr']}, "
            f"fedopt layers: {len(updated_params)}, "
            f"fedavg layers: {n_fedavg}, "
            f"update: {secs} secs., detach: {secs_detach} secs.",
        )

        global_model.params = weights
        global_model.meta = aggr_result.meta

        return global_model
