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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parameter import Parameter

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants

from .autofedrl_constants import AutoFedRLConstants


class PTAutoFedRLSearchSpace(FLComponent):
    def __init__(
        self,
        optimizer_args: dict = None,
        lr_scheduler_args: dict = None,
        device=None,
        search_lr=False,
        lr_range=None,
        search_ne=False,
        ne_range=None,
        search_aw=False,
        aw_range=None,
        search_slr=False,
        slr_range=None,
        cutoff_interval=5,
        n_clients=8,
        initial_precision=85.0,
        search_type="cs",
    ):
        """Implement the Auto-FedRL algorithm (https://arxiv.org/abs/2203.06338).

        The algorithm is proposed in Reddi, Sashank,
        et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).
        This SharableGenerator will update the global model using the specified
        PyTorch optimizer and learning rate scheduler.

        Args:
            optimizer_args: dictionary of optimizer arguments, e.g.
                {'path': 'torch.optim.SGD', 'args': {'lr': 1.0}} (default).
            lr_scheduler_args: dictionary of server-side learning rate scheduler arguments, e.g.
                {'path': 'torch.optim.lr_scheduler.CosineAnnealingLR', 'args': {'T_max': 100}} (default: None).
            device: specify the device to run server-side optimization, e.g. "cpu" or "cuda:0"
                (will default to cuda if available and no device is specified).

        Raises:
            TypeError: when any of input arguments does not have correct type
        """
        super().__init__()
        if not optimizer_args:
            self.logger("No optimizer_args provided. Using FedOpt with SGD and lr 0.01")
            optimizer_args = {"name": "Adam", "args": {"lr": 0.01, "betas": (0.7, 0.7)}}

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
        if search_type not in ["cs", "drl"]:
            raise NotImplementedError("Currently, we only implemented continuous search space")
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.optimizer = None
        self.lr_scheduler = None
        self.search_lr = search_lr
        self.lr_range = lr_range
        self.search_ne = search_ne
        self.ne_range = ne_range
        self.search_aw = search_aw
        self.aw_range = aw_range
        self.search_slr = search_slr
        self.slr_range = slr_range
        self.n_clients = n_clients
        self.initial_precision = initial_precision
        self.search_type = search_type
        self.cutoff_interval = cutoff_interval

        # Set default search ranges
        if self.lr_range is None:
            self.lr_range = [0.0005, 0.05]
        if self.ne_range is None:
            self.ne_range = [2, 40]
        if self.aw_range is None:
            self.aw_range = [0.1, 1.0]
        if self.slr_range is None:
            self.slr_range = [0.5, 1.5]

        # TODO: add checks for valid parameter ranges

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
            # Define RL search space
            hyperparams_points = []
            if self.search_lr:
                hyperparams_points += [self.lr_range]
            if self.search_ne:
                hyperparams_points += [self.ne_range]
            if self.search_aw:
                hyperparams_points += [self.aw_range for _ in range(self.n_clients)]
            if self.search_slr:
                hyperparams_points += [self.slr_range]

            if self.search_type == "cs":
                self.hp_dist = LearnableGaussianContinuousSearch(
                    hyperparams_points, self.initial_precision, self.device
                )
                self.hp_dist.to(self.device)
                self.log_info(fl_ctx, "Initialized Continuous Search Space")
            elif self.search_type == "drl":
                # TODO: Deep  RL agent requires torch==1.4.0
                self.hp_dist = LearnableGaussianContinuousSearchDRL(
                    hyperparams_points, self.initial_precision, self.device
                )
                self.hp_dist.to(self.device)
                self.log_info(fl_ctx, "Initialized DRL Continuous Search space")
            else:
                raise NotImplementedError

            # Set up optimizer
            try:
                # Use provided or default optimizer arguments and add the model parameters
                if "args" not in self.optimizer_args:
                    self.optimizer_args["args"] = {}
                self.optimizer_args["args"]["params"] = self.hp_dist.parameters()
                self.optimizer = engine.build_component(self.optimizer_args)
                # Get optimizer name for log
                self.optimizer_name = self._get_component_name(self.optimizer_args)
            except Exception as e:
                self.system_panic(
                    f"Exception while parsing `optimizer_args`: " f"{self.optimizer_args} with Exception {e}",
                    fl_ctx,
                )
                return
            # Initialize
            self.logprob_history = []
            self.val_losses = [-np.inf]
            self.log_info(fl_ctx, "Initialized validation loss fpr Search space")

    def sample_hyperparamters(self, fl_ctx: FLContext) -> None:
        """Convert Shareable to Learnable while doing a FedOpt update step.

        Args:
            shareable (Shareable): Shareable to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Model: Updated global ModelLearnable.
        """

        hparam, logprob = self.hp_dist.forward()

        hparam_list = list(hparam)
        if self.search_lr:
            lrate = hparam_list.pop(0)
        if self.search_ne:
            train_iters_per_round = hparam_list.pop(0)
            train_iters_per_round = int(train_iters_per_round + 0.5)
        if self.search_aw:
            aw = [hparam_list.pop(0) for _ in range(self.n_clients)]
            aw_tensor = torch.tensor([aw])
            aw_tensor = F.softmax(aw_tensor, dim=1)
            weight = [aw_tensor[:, i].item() for i in range(self.n_clients)]
        if self.search_slr:
            slr = hparam_list.pop(0)
        # Add constrains to prevent negative value
        if self.search_lr:
            lrate = lrate if lrate > 0.0001 else 0.0001
        if self.search_ne:
            train_iters_per_round = int(train_iters_per_round + 0.5) if train_iters_per_round >= 1 else 1
        if self.search_slr:
            slr = slr if slr > 0.0001 else 0.0001
        self.logprob_history.append(logprob)

        self.log_info(fl_ctx, f"Hyperparameter Search at round {fl_ctx.get_prop(AppConstants.CURRENT_ROUND)}")
        if self.search_lr:
            self.log_info(fl_ctx, f"Learning rate: {lrate}")
        if self.search_ne:
            self.log_info(fl_ctx, f"Number of local epochs: {train_iters_per_round}")
        if self.search_aw:
            self.log_info(fl_ctx, f"Aggregation weights: {weight}")
        if self.search_lr:
            self.log_info(fl_ctx, f"Server learning rate {slr}")
        if self.search_lr:
            self.log_info(fl_ctx, f"dist mean: {self.hp_dist.mean}")
        if self.search_lr:
            self.log_info(fl_ctx, f"precision component: {self.hp_dist.precision_component}")

        hps = {
            "lr": lrate if self.search_lr else None,
            "ne": train_iters_per_round if self.search_ne else None,
            "aw": weight if self.search_aw else None,
            "slr": slr if self.search_slr else None,
        }

        fl_ctx.set_prop(AutoFedRLConstants.HYPERPARAMTER_COLLECTION, hps, private=True, sticky=False)

    def update_search_space(self, shareable, fl_ctx: FLContext) -> None:
        if not isinstance(shareable, Shareable):
            raise TypeError("shareable must be Shareable, but got {}.".format(type(shareable)))

        dxo = from_shareable(shareable)
        if dxo.data_kind == DataKind.METRICS:
            val_loss = dxo.data["val_loss"]
        else:
            raise ValueError("data_kind should be DataKind.METRICS, but got {}".format(dxo.data_kind))
        self.val_losses.append(torch.tensor(val_loss, dtype=torch.float32, device=self.device))

        start = time.time()
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        # Get cutoff val losses for windowed updates
        cutoff_round = max(0, current_round - self.cutoff_interval)
        # Ignore initial loss
        val_losses_torch = torch.tensor(list(self.val_losses[1:]), dtype=torch.float32, device=self.device)
        val_losses_cut = val_losses_torch[cutoff_round:]

        current_loss = 0
        current_improvements = [0]
        # Compute hp search loss
        if len(val_losses_cut) > 1:
            current_improvements = -((val_losses_cut[1:] / val_losses_cut[:-1]) - 1)
            current_mean_improvements = current_improvements.mean()
            updates_limit = len(current_improvements) + 1
            for j in range(1, updates_limit):
                current_loss = self.logprob_history[-j - 1] * (current_improvements[-j] - current_mean_improvements)
        # Update search space
        if not (type(current_loss) == int):  # TODO: Is this needed?
            self.optimizer.zero_grad()
            (-current_loss).backward(retain_graph=True)
            self.optimizer.step()
            # We need release the memory based on cutoff interval
            if len(self.logprob_history) > self.cutoff_interval:
                self.log_info(fl_ctx, (f"Release Memory......at round {current_round}"))
                release_list = self.logprob_history[: -self.cutoff_interval]
                keep_list = self.logprob_history[-self.cutoff_interval :]
                tmp_list = [logpro.detach() for logpro in release_list]
                self.logprob_history = tmp_list + keep_list

        secs = time.time() - start

        self.log_info(
            fl_ctx,
            f"Finished Auto-FedRL search space update ({self.optimizer_name}, {self.device}) "
            f"round {current_round}, HP loss {current_loss} "
            f"lr: {self.optimizer.param_groups[-1]['lr']}, "
            f"update: {secs} secs.",
        )
        self.log_debug(fl_ctx, f"val loss: {self.val_losses}) ")
        self.log_debug(fl_ctx, f"logprob_history: {self.logprob_history}) ")


class LearnableGaussianContinuousSearch(torch.nn.Module):
    def __init__(self, hyperparams_points, initial_precision=None, device="cpu"):
        super(LearnableGaussianContinuousSearch, self).__init__()

        self.dim = len(hyperparams_points)
        self.hps = [np.array(x) for x in hyperparams_points]

        self.hps_center = torch.tensor([(x[0] + x[-1]) / 2 for x in self.hps]).to(device)
        self.hps_scale = torch.tensor([x[-1] - x[0] for x in self.hps]).to(device)

        self.mean = Parameter(torch.zeros(self.dim))

        precision_val = 5.0 if initial_precision is None else initial_precision
        precision_component = torch.sqrt(torch.eye(self.dim) * precision_val)
        self.precision_component = Parameter(precision_component)

    def forward(self):
        self.mean.data.copy_(torch.clamp(self.mean.data, -0.5, 0.5))
        self.dist = MultivariateNormal(loc=self.mean, precision_matrix=self.precision_component)

        sample = self.dist.sample()
        logprob = self.dist.log_prob(sample)
        sample = sample * self.hps_scale + self.hps_center

        return sample, logprob


class LearnableGaussianContinuousSearchDRL(torch.nn.Module):
    def __init__(self, hyperparams_points, initial_precision=None, device="cpu", rl_nettype="mlp"):
        super(LearnableGaussianContinuousSearchDRL, self).__init__()

        self.dim = len(hyperparams_points)
        if rl_nettype == "mlp":
            self.PolicyNet = PolicyNet(self.dim).to(device)
        else:
            raise NotImplementedError

        self.hps = [np.array(x) for x in hyperparams_points]

        self.hps_center = torch.tensor([(x[0] + x[-1]) / 2 for x in self.hps]).to(device)
        self.hps_scale = torch.tensor([x[-1] - x[0] for x in self.hps]).to(device)

        self.mean = torch.zeros(self.dim) + 10e-8

        precision_val = 5.0 if initial_precision is None else initial_precision
        precision_component = torch.sqrt(torch.eye(self.dim) * precision_val) + 10e-8
        self.precision_component = precision_component

    def forward(self):

        mean_update, precision_component_update = self.PolicyNet(self.mean, self.precision_component)
        self.mean = self.mean + mean_update
        self.precision_component = self.precision_component + precision_component_update
        self.mean.data.copy_(torch.clamp(self.mean.data, -1.0, 1.0))

        dist = MultivariateNormal(
            loc=self.mean, precision_matrix=torch.mm(self.precision_component, self.precision_component.t())
        )
        sample = dist.sample()
        logprob = dist.log_prob(sample)
        sample = sample * self.hps_scale + self.hps_center

        return sample, logprob


class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(PolicyNet, self).__init__()

        self.input_dim = input_dim
        in_chanel = input_dim * input_dim + input_dim
        self.fc_layer = nn.Sequential(
            nn.Linear(in_chanel, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, in_chanel),
            nn.Tanh(),
        )

    def forward(self, mean, precision_component):
        tmp = torch.cat([mean, precision_component.reshape((-1,))])
        input = torch.unsqueeze(tmp, 0)
        x = torch.squeeze(self.fc_layer(input)) / 100.0
        mean_update = x[: self.input_dim]
        precision_component_update = x[self.input_dim :].reshape((self.input_dim, self.input_dim))

        return mean_update, precision_component_update
