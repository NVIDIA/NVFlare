# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.security.logging import secure_format_exception


class PTFedOptModelShareableGenerator(FullModelShareableGenerator):
    def __init__(
        self,
        optimizer_args: dict = None,
        lr_scheduler_args: dict = None,
        source_model="model",
        device=None,
    ):
        """Implement the FedOpt algorithm.

        The algorithm is proposed in Reddi, Sashank, et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).
        This SharableGenerator will update the global model using the specified
        PyTorch optimizer and learning rate scheduler.

        Args:
            optimizer_args: dictionary of optimizer arguments, e.g.
                {'path': 'torch.optim.SGD', 'args': {'lr': 1.0}} (default).
            lr_scheduler_args: dictionary of server-side learning rate scheduler arguments, e.g.
                {'path': 'torch.optim.lr_scheduler.CosineAnnealingLR', 'args': {'T_max': 100}} (default: None).
            source_model: either a valid torch model object or a component ID of a torch model object
            device: specify the device to run server-side optimization, e.g. "cpu" or "cuda:0"
                (will default to cuda if available and no device is specified).

        Raises:
            TypeError: when any of input arguments does not have correct type
        """
        super().__init__()
        if not optimizer_args:
            self.logger("No optimizer_args provided. Using FedOpt with SGD and lr 1.0")
            optimizer_args = {"name": "SGD", "args": {"lr": 1.0}}

        if not isinstance(optimizer_args, dict):
            raise TypeError(
                "optimizer_args must be a dict of format, e.g. {'path': 'torch.optim.SGD', 'args': {'lr': 1.0}}."
            )
        if lr_scheduler_args is not None:
            if not isinstance(lr_scheduler_args, dict):
                raise TypeError(
                    "optimizer_args must be a dict of format, e.g. "
                    "{'path': 'torch.optim.lr_scheduler.CosineAnnealingLR', 'args': {'T_max': 100}}."
                )
        self.source_model = source_model
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.optimizer_name = None
        self.lr_scheduler_name = None

    def _get_component_name(self, component_args):
        if component_args is not None:
            name = component_args.get("path", None)
            if name is None:
                name = component_args.get("name", None)
            return name
        else:
            return None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            # Initialize the optimizer with current global model params
            engine = fl_ctx.get_engine()

            if isinstance(self.source_model, str):
                self.model = engine.get_component(self.source_model)
            else:
                self.model = self.source_model

            if self.model is None:
                self.system_panic(
                    "Model is not available",
                    fl_ctx,
                )
                return
            elif not isinstance(self.model, torch.nn.Module):
                self.system_panic(
                    f"Expected model to be a torch.nn.Module but got {type(self.model)}",
                    fl_ctx,
                )
                return
            else:
                print("server model", self.model)

            self.model.to(self.device)

            # set up optimizer
            try:
                # use provided or default optimizer arguments and add the model parameters
                if "args" not in self.optimizer_args:
                    self.optimizer_args["args"] = {}
                self.optimizer_args["args"]["params"] = self.model.parameters()
                self.optimizer = engine.build_component(self.optimizer_args)
                # get optimizer name for log
                self.optimizer_name = self._get_component_name(self.optimizer_args)
            except Exception as e:
                self.system_panic(
                    f"Exception while parsing `optimizer_args`({self.optimizer_args}): {secure_format_exception(e)}",
                    fl_ctx,
                )
                return

            # set up lr scheduler
            if self.lr_scheduler_args is not None:
                try:
                    self.lr_scheduler_name = self._get_component_name(self.lr_scheduler_args)
                    # use provided or default lr scheduler argument and add the optimizer
                    if "args" not in self.lr_scheduler_args:
                        self.lr_scheduler_args["args"] = {}
                    self.lr_scheduler_args["args"]["optimizer"] = self.optimizer
                    self.lr_scheduler = engine.build_component(self.lr_scheduler_args)
                except Exception as e:
                    self.system_panic(
                        f"Exception while parsing `lr_scheduler_args`({self.lr_scheduler_args}): {secure_format_exception(e)}",
                        fl_ctx,
                    )
                    return

    def server_update(self, model_diff):
        """Updates the global model using the specified optimizer.

        Args:
            model_diff: the aggregated model differences from clients.

        Returns:
            The updated PyTorch model state dictionary.

        """
        self.model.train()
        self.optimizer.zero_grad()

        # Apply the update to the model. We must multiply weights_delta by -1.0 to
        # view it as a gradient that should be applied to the server_optimizer.
        for name, param in self.model.named_parameters():
            param.grad = torch.tensor(-1.0 * model_diff[name]).to(self.device)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.model.state_dict()

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        """Convert Shareable to Learnable while doing a FedOpt update step.

        Supporting data_kind == DataKind.WEIGHT_DIFF

        Args:
            shareable (Shareable): Shareable to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Model: Updated global ModelLearnable.
        """
        # check types
        dxo = from_shareable(shareable)

        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            self.system_panic(
                "FedOpt is only implemented for " "data_kind == DataKind.WEIGHT_DIFF",
                fl_ctx,
            )
            return Learnable()

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            self.system_panic(
                f"FedOpt is not implemented for shareable processed by {processed_algorithm}",
                fl_ctx,
            )
            return Learnable()

        model_diff = dxo.data

        start = time.time()
        weights = self.server_update(model_diff)
        secs = time.time() - start

        # convert to numpy dict of weights
        start = time.time()
        for key in weights:
            weights[key] = weights[key].detach().cpu().numpy()
        secs_detach = time.time() - start

        self.log_info(
            fl_ctx,
            f"FedOpt ({self.optimizer_name}, {self.device}) server model update "
            f"round {fl_ctx.get_prop(AppConstants.CURRENT_ROUND)}, "
            f"{self.lr_scheduler_name if self.lr_scheduler_name else ''} "
            f"lr: {self.optimizer.param_groups[-1]['lr']}, "
            f"update: {secs} secs., detach: {secs_detach} secs.",
        )
        # TODO: write server-side lr to tensorboard

        return make_model_learnable(weights, dxo.get_meta_props())
