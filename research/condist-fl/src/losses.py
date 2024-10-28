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

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from monai.losses import DiceCELoss, MaskedDiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class ConDistTransform(object):
    def __init__(
        self,
        num_classes: int,
        foreground: Sequence[int],
        background: Sequence[Union[int, Sequence[int]]],
        temperature: float = 2.0,
    ):
        self.num_classes = num_classes

        self.foreground = foreground
        self.background = background

        if temperature < 0.0:
            raise ValueError("Softmax temperature must be a postive number!")
        self.temperature = temperature

    def softmax(self, data: Tensor):
        return torch.softmax(data / self.temperature, dim=1)

    def reduce_channels(self, data: Tensor, eps: float = 1e-5):
        batch, channels, *shape = data.shape
        if channels != self.num_classes:
            raise ValueError(f"Expect input with {self.num_classes} channels, get {channels}")

        fg_shape = [batch] + [1] + shape
        bg_shape = [batch] + [len(self.background)] + shape

        # Compute the probability for the union of local foreground
        fg = torch.zeros(fg_shape, dtype=torch.float32, device=data.device)
        for c in self.foreground:
            fg += data[:, c, ::].view(*fg_shape)

        # Compute the raw probabilities for each background group
        bg = torch.zeros(bg_shape, dtype=torch.float32, device=data.device)
        for i, g in enumerate(self.background):
            if isinstance(g, int):
                bg[:, i, ::] = data[:, g, ::]
            else:
                for c in g:
                    bg[:, i, ::] += data[:, c, ::]

        # Compute condistional probability for background groups
        return bg / (1.0 - fg + eps)

    def generate_mask(self, targets: Tensor, ground_truth: Tensor):
        targets = torch.argmax(targets, dim=1, keepdim=True)

        # The mask covers the background but excludes false positive areas
        condition = torch.zeros_like(targets, device=targets.device)
        for c in self.foreground:
            condition = torch.where(torch.logical_or(targets == c, ground_truth == c), 1, condition)
        mask = 1 - condition

        return mask.astype(torch.float32)

    def __call__(self, preds: Tensor, targets: Tensor, ground_truth: Tensor) -> Tuple[Tensor]:
        mask = self.generate_mask(targets, ground_truth)

        preds = self.softmax(preds)
        preds = self.reduce_channels(preds)

        targets = self.softmax(targets)
        targets = self.reduce_channels(targets)

        return preds, targets, mask


class MarginalTransform(object):
    def __init__(self, foreground: Sequence[int], softmax: bool = False):
        self.foreground = foreground
        self.softmax = softmax

    def reduce_background_channels(self, tensor: Tensor) -> Tensor:
        n_chs = tensor.shape[1]
        slices = torch.split(tensor, 1, dim=1)

        fg = [slices[i] for i in self.foreground]
        bg = sum([slices[i] for i in range(n_chs) if i not in self.foreground])

        output = torch.cat([bg] + fg, dim=1)
        return output

    def __call__(self, preds: Tensor, target: Tensor) -> Tuple[Tensor]:
        n_pred_ch = preds.shape[1]
        if n_pred_ch == 1:
            # Marginal loss is not intended for single channel output
            return preds, target

        if self.softmax:
            preds = torch.softmax(preds, 1)

        if target.shape[1] == 1:
            target = one_hot(target, num_classes=n_pred_ch)
        elif target.shape != n_pred_ch:
            raise ValueError(f"Number of channels of label must be 1 or {n_pred_ch}.")

        preds = self.reduce_background_channels(preds)
        target = self.reduce_background_channels(target)

        return preds, target


class ConDistDiceLoss(_Loss):
    def __init__(
        self,
        num_classes: int,
        foreground: Sequence[int],
        background: Sequence[Union[int, Sequence[int]]],
        temperature: float = 2.0,
        include_background: bool = True,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__()

        self.transform = ConDistTransform(num_classes, foreground, background, temperature=temperature)
        self.dice = MaskedDiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=False,
            softmax=False,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

    def forward(self, preds: Tensor, targets: Tensor, ground_truth: Tensor):
        n_chs = preds.shape[1]
        if (ground_truth.shape[1] > 1) and (ground_truth.shape[1] == n_chs):
            ground_truth = torch.argmax(ground_truth, dim=1, keepdim=True)

        preds, targets, mask = self.transform(preds, targets, ground_truth)
        return self.dice(preds, targets, mask=mask)


class MarginalDiceCELoss(_Loss):
    def __init__(
        self,
        foreground: Sequence[int],
        include_background: bool = True,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ):
        super().__init__()

        self.transform = MarginalTransform(foreground, softmax=softmax)
        self.dice_ce = DiceCELoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=False,
            softmax=False,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            ce_weight=ce_weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(self, preds: Tensor, targets: Tensor):
        preds, targets = self.transform(preds, targets)
        return self.dice_ce(preds, targets)


class MoonContrasiveLoss(torch.nn.Module):
    def __init__(self, tau: float = 1.0):
        super().__init__()

        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = tau

    def forward(self, z: Tensor, z_prev: Tensor, z_glob: Tensor):
        sim_prev = F.cosine_similarity(z, z_prev, dim=1)
        sim_glob = F.cosine_similarity(z, z_glob, dim=1)

        exp_prev = torch.exp(sim_prev / self.tau)
        exp_glob = torch.exp(sim_glob / self.tau)

        loss = -torch.log(exp_glob / (exp_glob + exp_prev))
        return loss.mean()
