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

"""Local E/F/G training with a frozen copy of the server discriminator."""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import FeatureEncoder, GNet, GraphDNet, PredNet
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from utils.privacy_utils import add_laplace_noise

logger = logging.getLogger("GFedCL")


class ModifiedClient(nn.Module):
    def __init__(self, client_id, opt):
        super().__init__()
        self.client_id = client_id
        self.opt = opt
        self.device = torch.device(opt.device)
        self.batch_size = opt.batch_size
        self.netE = FeatureEncoder(opt).to(self.device)
        self.netF = PredNet(opt).to(self.device)
        self.netG = GNet(opt).to(self.device)
        for network in (self.netE, self.netF, self.netG):
            self.__init_weight__(network)

        self.optimizer_EFG = optim.Adam(
            [
                {"params": self.netE.parameters(), "lr": opt.lr_e},
                {"params": self.netF.parameters(), "lr": opt.lr_f},
                {"params": self.netG.parameters(), "lr": getattr(opt, "lr_g", opt.lr_e)},
            ],
            betas=(opt.beta1, getattr(opt, "beta2", 0.999)),
        )
        self.lr_scheduler_EFG = ExponentialLR(self.optimizer_EFG, gamma=0.5 ** (1 / 100))
        self.loss_names = ["E_pred", "E_gan", "G_pred", "G_gan"]
        self.server_discriminator = None
        # Replay retains label counts and batch sizes, never historical raw data.
        self.task_label_counts = {}
        self.task_batch_sizes = {}

    def getId(self):
        return self.client_id

    def train(self, mode=True):
        super().train(mode)
        if self.server_discriminator is not None:
            self.server_discriminator.eval()
        return self

    def set_server_discriminator(self, discriminator_state_dict):
        if self.server_discriminator is None:
            self.server_discriminator = GraphDNet(self.opt).to(self.device)
        self.server_discriminator.load_state_dict(discriminator_state_dict)
        # Freeze D's weights, while retaining derivatives with respect to its input.
        self.server_discriminator.requires_grad_(False)
        self.server_discriminator.eval()

    def register_task(self, task, dataloader):
        """Retain the current task's label distribution for future synthetic replay."""
        if task in self.task_label_counts:
            return
        counts = torch.zeros(self.opt.num_classes, dtype=torch.long)
        batch_sizes = []
        for _, labels in dataloader:
            labels = labels.detach().cpu().long().reshape(-1)
            if (labels < 0).any() or (labels >= self.opt.num_classes).any():
                raise ValueError("Task labels are outside the configured class range")
            counts += torch.bincount(labels, minlength=self.opt.num_classes)
            batch_sizes.append(labels.numel())
        if not batch_sizes or counts.sum() == 0:
            raise ValueError(f"Client {self.client_id}: task {task} has no training samples")
        self.task_label_counts[task] = counts
        self.task_batch_sizes[task] = batch_sizes

    def _graph_row(self, relational_graphs, task, batch_size):
        if not isinstance(relational_graphs, (list, tuple)) or not 0 <= task < len(relational_graphs):
            raise ValueError(f"Missing relational graph for task {task}")
        if relational_graphs[task] is None:
            raise ValueError(f"Missing relational graph for task {task}")
        graph = torch.as_tensor(relational_graphs[task], device=self.device, dtype=torch.float32)
        expected = (self.opt.num_clients, self.opt.num_clients)
        if tuple(graph.shape) != expected or not torch.isfinite(graph).all():
            raise ValueError(f"Task {task} graph must be a finite matrix with shape {expected}")
        return graph[self.client_id].detach().unsqueeze(0).expand(batch_size, -1)

    def _synthetic_tasks(self, task):
        return range(task + 1) if self.opt.replay else [task]

    def _generate_latent(self, task, relational_graphs, batch_size, labels=None):
        if labels is None:
            if task not in self.task_label_counts:
                raise ValueError(f"Register task {task} before requesting synthetic replay")
            labels = torch.multinomial(
                self.task_label_counts[task].float(),
                batch_size,
                replacement=True,
            ).to(self.device)
        graph_row = self._graph_row(relational_graphs, task, batch_size)
        noise = torch.randn(batch_size, self.netG.noise_dim, device=self.device)
        return self.netG(noise, labels, graph_row), labels, graph_row

    def _representation_loss(self, latent, labels, graph_row):
        prediction_loss = F.nll_loss(self.netF(latent), labels.long())
        # D is frozen, not wrapped in no_grad: this loss must reach E and G.
        adversarial_loss = -F.mse_loss(self.server_discriminator(latent), graph_row)
        return prediction_loss, adversarial_loss

    def learn(self, epoch, task, relational_graphs, dataloader, generate=False):
        """Train E/F/G on current real data and task-conditioned synthetic latents.

        With replay enabled, synthetic losses cover tasks 0..task, as in the
        paper. With replay disabled, G still learns on the current task.
        ``generate=True`` is a compatibility path for synthetic-only learning
        on one registered task and never reads a historical dataloader.
        """
        if self.server_discriminator is None:
            raise RuntimeError("Set the server discriminator before client training")
        if generate:
            if task not in self.task_batch_sizes:
                raise ValueError(f"Register task {task} before synthetic-only training")
            batches = ((None, size) for size in self.task_batch_sizes[task])
        else:
            self.register_task(task, dataloader)
            batches = ((data, len(data[1])) for data in dataloader)
        self.train()
        losses = {name: 0.0 for name in self.loss_names}
        encodings, graph_rows = [], []
        count = 0

        for data, batch_size in batches:
            self.optimizer_EFG.zero_grad(set_to_none=True)
            zero = torch.zeros((), device=self.device)
            self.loss_E_pred, self.loss_E_gan = zero, zero
            if data is not None:
                inputs, labels = data
                labels = labels.to(self.device).long()
                self.z_seq = self._graph_row(relational_graphs, task, batch_size)
                self.e_seq = self.netE(inputs.to(self.device), self.z_seq)
                self.loss_E_pred, self.loss_E_gan = self._representation_loss(
                    self.e_seq,
                    labels,
                    self.z_seq,
                )
                encodings.append(self.e_seq.detach())
                graph_rows.append(self.z_seq.detach())

            self.loss_G_pred, self.loss_G_gan = zero, zero
            for replay_task in ([task] if generate else self._synthetic_tasks(task)):
                # Current labels come from this batch; old labels are sampled
                # from saved counts without accessing historical raw inputs.
                condition = labels if data is not None and replay_task == task else None
                latent, synthetic_labels, graph_row = self._generate_latent(
                    replay_task,
                    relational_graphs,
                    batch_size,
                    condition,
                )
                pred_loss, gan_loss = self._representation_loss(latent, synthetic_labels, graph_row)
                self.loss_G_pred = self.loss_G_pred + pred_loss
                self.loss_G_gan = self.loss_G_gan + gan_loss
                encodings.append(latent.detach())
                graph_rows.append(graph_row.detach())

            loss = self.loss_E_pred + self.loss_G_pred + self.opt.lambda_gan * (self.loss_E_gan + self.loss_G_gan)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite E/F/G training loss")
            loss.backward()
            self.optimizer_EFG.step()
            for name, value in zip(
                self.loss_names,
                (
                    self.loss_E_pred,
                    self.loss_E_gan,
                    self.loss_G_pred,
                    self.loss_G_gan,
                ),
            ):
                losses[name] += value.item()
            count += 1

        if count:
            losses = {name: value / count for name, value in losses.items()}
            self.lr_scheduler_EFG.step()
        logger.info("Client %s, Task %s, Epoch %s: %s", self.client_id, task, epoch, losses)
        return {"loss_values": losses, "encodings": encodings, "graph_embeddings": graph_rows}

    def generate_encodings(self, task, relational_graphs, dataloader, generate=False):
        """Perturb real and G-produced latents locally before returning an upload."""
        if generate:
            if task not in self.task_batch_sizes:
                raise ValueError(f"Register task {task} before requesting synthetic replay")
            batches = ((None, size) for size in self.task_batch_sizes[task])
        else:
            self.register_task(task, dataloader)
            batches = ((data, len(data[1])) for data in dataloader)
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        encodings, graph_rows = [], []
        try:
            with torch.no_grad():
                for data, batch_size in batches:
                    if data is not None:
                        inputs, labels = data
                        labels = labels.to(self.device).long()
                        row = self._graph_row(relational_graphs, task, batch_size)
                        encodings.append(self.netE(inputs.to(self.device), row))
                        graph_rows.append(row.clone())
                    for replay_task in ([task] if generate else self._synthetic_tasks(task)):
                        condition = labels if data is not None and replay_task == task else None
                        latent, _, row = self._generate_latent(
                            replay_task,
                            relational_graphs,
                            batch_size,
                            condition,
                        )
                        encodings.append(latent)
                        graph_rows.append(row.clone())
        finally:
            for module, training in modes:
                module.training = training
        # Every real/current/replay latent receives an independent perturbation
        # before it can leave this client through the Collab encode response.
        uploaded_encodings = [add_laplace_noise(encoding, self.opt.b) for encoding in encodings]
        return {"encodings": uploaded_encodings, "graph_embeddings": graph_rows}

    def test(self, task_id, dataloader, relational_graphs):
        self.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.to(self.device).long()
                row = self._graph_row(relational_graphs, task_id, len(labels))
                # Labels are used only for metrics, never as encoder inputs.
                logits = self.netF(self.netE(inputs.to(self.device), row))
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)
                total_loss += F.nll_loss(logits, labels, reduction="sum").item()
        return {"loss": total_loss / total if total else 0.0, "acc": 100.0 * correct / total if total else 0.0}

    def get_weights(self):
        return {
            "encoder": copy.deepcopy(self.netE.state_dict()),
            "predictor": copy.deepcopy(self.netF.state_dict()),
            "generator": copy.deepcopy(self.netG.state_dict()),
        }

    def set_weights(self, weights):
        for name, network in (("encoder", self.netE), ("predictor", self.netF), ("generator", self.netG)):
            if name in weights:
                network.load_state_dict(weights[name])

    def get_training_state(self):
        """Keep each client's Adam state and task metadata across NVFlare operations."""
        return copy.deepcopy(
            {
                "optimizer": self.optimizer_EFG.state_dict(),
                "scheduler": self.lr_scheduler_EFG.state_dict(),
                "task_label_counts": self.task_label_counts,
                "task_batch_sizes": self.task_batch_sizes,
            }
        )

    def set_training_state(self, state):
        state = copy.deepcopy(state)
        self.optimizer_EFG.load_state_dict(state["optimizer"])
        self.lr_scheduler_EFG.load_state_dict(state["scheduler"])
        self.task_label_counts = state["task_label_counts"]
        self.task_batch_sizes = state["task_batch_sizes"]

    @staticmethod
    def __init_weight__(net):
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
