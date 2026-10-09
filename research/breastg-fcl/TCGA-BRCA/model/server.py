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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model.modules import GraphDNet

logger = logging.getLogger("GFedCL")


class Server:
    """
    Server Class for GFedCL

    The server:
    1. Maintains a global discriminator
    2. Receives encoded samples from clients
    3. Trains the discriminator on those encodings
    4. Sends updated discriminator back to clients
    """

    def __init__(self, opt):
        """
        Initialize the server with a global discriminator

        Args:
            opt: Configuration options
        """
        self.opt = opt
        self.device = torch.device(opt.device)

        # Initialize the global discriminator
        self.global_discriminator = GraphDNet(opt).to(self.device)

        # Set up optimizer
        self.optimizer_D = optim.Adam(
            self.global_discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, getattr(opt, "beta2", 0.999))
        )

        # Set up learning rate scheduler
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))

    def _prepare_batches(self, encoded_samples, graph_embeddings):
        """Pair each latent batch with its own graph rows before concatenation."""
        if len(encoded_samples) != len(graph_embeddings):
            raise ValueError(
                "Discriminator requires one graph batch per latent batch: "
                f"got {len(encoded_samples)} latent and {len(graph_embeddings)} graph batches"
            )
        if not encoded_samples:
            raise ValueError("Discriminator requires at least one nonempty latent batch")

        parameter = next(self.global_discriminator.parameters())
        latent_batches, target_batches = [], []
        for index, (latent, graph) in enumerate(zip(encoded_samples, graph_embeddings)):
            for name, tensor, width in (
                ("latent", latent, self.opt.nh),
                ("graph", graph, self.opt.nt),
            ):
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Discriminator {name} batch {index} must be a tensor")
                if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] != width:
                    raise ValueError(
                        f"Discriminator {name} batch {index} must have shape [B, {width}] "
                        f"with B > 0; got {tuple(tensor.shape)}"
                    )

            if graph.shape[0] not in (1, latent.shape[0]):
                raise ValueError(
                    f"Discriminator graph batch {index} has {graph.shape[0]} rows; "
                    f"expected 1 or {latent.shape[0]} to match its latent batch"
                )
            latent = latent.detach().to(device=self.device, dtype=parameter.dtype)
            graph = graph.detach().to(device=self.device, dtype=parameter.dtype)
            if not torch.isfinite(latent).all() or not torch.isfinite(graph).all():
                raise ValueError(f"Discriminator batch {index} contains nonfinite values")
            if graph.shape[0] == 1:
                graph = graph.expand(latent.shape[0], -1)
            latent_batches.append(latent)
            target_batches.append(graph)

        return torch.cat(latent_batches, dim=0), torch.cat(target_batches, dim=0)

    def _discriminator_loss(self, latent, target):
        predicted = self.global_discriminator(latent)
        if predicted.shape != target.shape:
            raise ValueError(
                f"Discriminator prediction shape {tuple(predicted.shape)} "
                f"does not match graph target shape {tuple(target.shape)}"
            )
        if not torch.isfinite(predicted).all():
            raise ValueError("Discriminator predictions contain nonfinite values")
        loss = F.mse_loss(predicted, target, reduction="mean")
        if not torch.isfinite(loss):
            raise ValueError("Discriminator loss is nonfinite")
        return loss

    def train_discriminator(self, encoded_samples, graph_embeddings):
        """Update only D using paired latent batches and graph-row targets.

        Each graph batch must contain either one row for its entire latent
        batch or one row per latent sample. Inputs are detached so this update
        cannot propagate into client encoders, generators, or graph attention.
        """
        latent, target = self._prepare_batches(encoded_samples, graph_embeddings)
        self.global_discriminator.train()
        self.optimizer_D.zero_grad(set_to_none=True)
        loss_D = self._discriminator_loss(latent, target)
        loss_D.backward()
        self.optimizer_D.step()
        logger.debug(
            "Discriminator training: latent shape %s, target shape %s, loss %.4f",
            tuple(latent.shape),
            tuple(target.shape),
            loss_D.item(),
        )
        return loss_D.item()

    def get_discriminator(self):
        """
        Get the global discriminator state_dict

        Returns:
            state_dict: The state dictionary of the global discriminator
        """
        return copy.deepcopy(self.global_discriminator.state_dict())

    def set_discriminator(self, state_dict):
        """
        Update the global discriminator with a new state_dict

        Args:
            state_dict: The state dictionary to load
        """
        self.global_discriminator.load_state_dict(state_dict)

    def update_learning_rate(self):
        """
        Update the learning rate using the scheduler
        """
        self.lr_scheduler_D.step()

    def evaluate_discriminator(self, encoded_samples, graph_embeddings):
        """Evaluate D with the same paired graph-row alignment used in training."""
        latent, target = self._prepare_batches(encoded_samples, graph_embeddings)
        self.global_discriminator.eval()
        with torch.no_grad():
            return self._discriminator_loss(latent, target).item()
