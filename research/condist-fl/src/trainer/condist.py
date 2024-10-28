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

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from losses import ConDistDiceLoss, MarginalDiceCELoss
from monai.losses import DeepSupervisionLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.get_model import get_model


class ConDistTrainer(object):
    def __init__(self, task_config: Dict):
        self.init_lr = task_config["training"].get("lr", 1e-2)
        self.max_steps = task_config["training"]["max_steps"]
        self.max_rounds = task_config["training"]["max_rounds"]

        self.use_half_precision = task_config["training"].get("use_half_precision", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_half_precision)

        num_classes = len(task_config["classes"])
        foreground = task_config["condist_config"]["foreground"]
        background = task_config["condist_config"]["background"]
        temperature = task_config["condist_config"].get("temperature", 2.0)
        self.model_config = task_config["model"]
        self.weight_range = task_config["condist_config"]["weight_schedule_range"]

        self.condist_loss_fn = ConDistDiceLoss(
            num_classes, foreground, background, temperature=temperature, smooth_nr=0.0, batch=True
        )
        self.marginal_loss_fn = MarginalDiceCELoss(foreground, softmax=True, smooth_nr=0.0, batch=True)
        self.ds_loss_fn = DeepSupervisionLoss(self.marginal_loss_fn, weights=[0.5333, 0.2667, 0.1333, 0.0667])

        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None

    def update_condist_weight(self):
        left = min(self.weight_range)
        right = max(self.weight_range)
        intv = (right - left) / (self.max_rounds - 1)
        self.weight = left + intv * self.current_round

    def configure_optimizer(self):
        self.opt = SGD(self.model.parameters(), lr=self.init_lr, momentum=0.99, nesterov=True, weight_decay=1e-5)
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)

        self.sch = CosineAnnealingLR(self.opt, T_max=self.max_steps, eta_min=1e-7)
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def training_step(self, model: nn.Module, batch: Dict, device: str = "cuda:0"):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        preds = model(image)
        if preds.dim() == 6:
            preds = [preds[:, i, ::] for i in range(preds.shape[1])]
        ds_loss = self.ds_loss_fn(preds, label)

        with torch.no_grad():
            targets = self.global_model(image)
            if targets.dim() == 6:
                targets = targets[:, 0, ::]
        condist_loss = self.condist_loss_fn(preds[0], targets, label)

        loss = ds_loss + self.weight * condist_loss

        # Log training information
        if self.logger is not None:
            step = self.current_step
            self.logger.add_scalar("loss", loss, step)
            self.logger.add_scalar("loss_sup", ds_loss, step)
            self.logger.add_scalar("loss_condist", condist_loss, step)
            self.logger.add_scalar("lr", self.sch.get_last_lr()[-1], step)
            self.logger.add_scalar("condist_weight", self.weight, step)

        return loss

    def get_batch(self, data_loader: DataLoader, num_steps: int):
        it = iter(data_loader)
        for i in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            yield batch

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        self.global_model = self.global_model.to(device)

        target_step = self.current_step + num_steps
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for batch in self.get_batch(data_loader, num_steps):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    loss = self.training_step(self.model, batch)

                # Backward
                self.opt.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                # Apply gradient
                self.scaler.step(self.opt)
                self.sch.step()
                self.scaler.update()

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

    def setup(self, model: nn.Module, logger: SummaryWriter, abort_signal: Any):
        self.model = model
        # self.global_model = deepcopy(model).eval()
        self.global_model = get_model(self.model_config)
        self.global_model.load_state_dict(deepcopy(model.state_dict()))
        self.global_model.eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()
        self.update_condist_weight()

    def cleanup(self):
        # Save opt & sch states
        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch_state = deepcopy(self.sch.state_dict())

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None
        self.global_model = None

        self.logger = None
        self.abort_signal = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)

        model.load_state_dict(ckpt["model"])
        return model

    def run(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_steps: int,
        device: str = "cuda:0",
        logger: Optional[SummaryWriter] = None,
        abort_signal: Optional[Any] = None,
    ):
        self.setup(model, logger, abort_signal)

        # Run training
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1

        self.cleanup()
