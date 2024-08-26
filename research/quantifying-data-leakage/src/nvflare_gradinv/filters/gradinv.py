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

import json
import os
from copy import deepcopy

import numpy as np
import torch
from fl_gradient_inversion import FLGradientInversion
from monai.networks.nets.torchvision_fc import TorchVisionFCModel
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    NormalizeIntensity,
    Resize,
    SaveImage,
    ScaleIntensity,
    Transform,
)
from monai.utils import ImageMetaKey

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learner_spec import Learner


class Inverter(FLComponent):
    def __init__(self):
        super().__init__()

    def __call__(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext):
        """Subclass must implement this method to filter the provided DXO

        Args:
            dxo: the DXO to containing the data to be inverted
            shareable: the shareable that the dxo belongs to
            fl_ctx: the FL context

        Returns:
            Inversions

        """
        raise NotImplementedError(f"Subclass of {self.__class__.__name__} must implement this method.")


class AddName(Transform):
    def __init__(self, num_images=1):
        self.num_images = num_images
        self.idx = 0

    def __call__(self, img):
        if self.idx == self.num_images:
            self.idx = 0
        img.meta[ImageMetaKey.FILENAME_OR_OBJ] = f"recon_b{self.idx}"
        self.idx += 1
        return img


class GradInversionInverter(Inverter):
    def __init__(
        self,
        learner_name: str = "learner",
        cfg_file: str = "config/config_inversion.json",
        bn_momentum: float = 0.1,
        compute_update_sums: bool = True,
        print_names: bool = True,
        use_determinism: bool = True,
        prior_transforms=None,
        save_transforms=None,
        save_fmt=".png",
    ):
        """Wrapper class calling gradient inversion. Assumes being used with `CXRLearner` or
        Learners that have the same member variables to be accessed in `__call__()`.

        Args:
            learner_name: ID of the `Learner` component used to get training hyperparameters.
            cfg_file: Configuration file used by `FLGradientInversion` class.
            bn_momentum: Batch norm momentum used by the local trainer code in `Learner`. Defaults to 0.1.
            compute_update_sums: Whether to print the absolute sum of the model updates. Defaults to `True`.
            print_names: Whether to print the layer variable names of the model. Defaults to `True`.
            use_determinism: Whether to use deterministic functions for the reconstruction. Defaults to `True`.
            prior_transforms: Optional custom transforms to read prior images. Defaults to `None`.
            save_transforms: Optional custom transforms to save the reconstructed images. Defaults to `None`.
            save_fmt: Output format to save individual reconstructions. Defaults to ".png".

        Returns:
            Reconstructions
        """
        super().__init__()
        if not isinstance(learner_name, str):
            raise ValueError(f"Expected `learner_name` of type `str` but received type {type(learner_name)}")
        if not isinstance(cfg_file, str):
            raise ValueError(f"Expected `cfg_file` of type `str` but received type {type(cfg_file)}")
        if not isinstance(bn_momentum, float):
            raise ValueError(f"Expected `bn_momentum` of type `float` but received type {type(bn_momentum)}")
        self.learner_name = learner_name
        self.cfg_file = cfg_file
        self.bn_momentum = bn_momentum
        self.compute_update_sums = compute_update_sums
        self.print_names = print_names
        self.prior_transforms = prior_transforms
        self.save_transforms = save_transforms
        self.save_fmt = save_fmt

        self.cfg = None
        self.save_path = None

        self.use_determinism = use_determinism
        if self.use_determinism:
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def run_inversion(
        cfg, updates, global_weights, bn_momentum=0.1, prior_transforms=None, save_transforms=None, save_fmt=".png"
    ):
        """Wrapper function calling `FLGradientInversion`.

        Args:
            cfg: Configuration dictionary used by `FLGradientInversion` class.
            updates: The model updates sent by the client.
            global_weights: The current state dict of global model the `updates` are with respect to.
            bn_momentum: Batch norm momentum used by the local trainer code in `Learner`. Defaults to 0.1.
            prior_transforms: Optional custom transforms to read prior images. Defaults to `None`.
            save_transforms: Optional custom transforms to save the reconstructed images. Defaults to `None`.
            save_fmt: Output format to save individual reconstructions. Defaults to ".png".

        Returns:
            Reconstructions
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = TorchVisionFCModel(
            model_name=cfg["model_name"],
            num_classes=cfg["num_classes"],
            pretrained=False,
        )

        # get global weights
        if "model" in global_weights:
            net.load_state_dict(global_weights["model"])
        else:
            net.load_state_dict(global_weights)

        # compute weight changes
        update_sum = 0.0
        for name, _ in net.named_parameters():
            update_sum += np.sum(np.abs(updates[name]))
        assert update_sum > 0.0, "All updates are zero!"
        model_bn = deepcopy(net).cuda()
        update_sum = 0.0
        new_state_dict = model_bn.state_dict()
        for n in updates.keys():
            val = updates[n]
            update_sum += np.sum(np.abs(val))
            new_state_dict[n] = new_state_dict[n] + torch.tensor(val, device=new_state_dict[n].device)
        model_bn.load_state_dict(new_state_dict)
        assert update_sum > 0.0, "All updates are zero!"
        n_bn_updated = 0
        global_state_dict = net.state_dict()

        # Compute full BN stats
        bn_stats = {}
        for param_name in updates:
            if "bn" in param_name or "batch" in param_name or "running" in param_name:
                bn_stats[param_name] = global_weights[param_name] + updates[param_name]
        for n in bn_stats.keys():
            if "running" in n:
                xt = (bn_stats[n] - (1 - bn_momentum) * global_state_dict[n].numpy()) / bn_momentum
                n_bn_updated += 1
                bn_stats[n] = xt

        # move weight updates and model to gpu
        net = net.to(device)
        grad_lst = []
        for name, _ in net.named_parameters():
            val = torch.from_numpy(updates[name]).cuda()
            grad_lst.append([name, val])

        # Use same transforms to load prior as used in training routine
        # TODO: make configurable
        if prior_transforms is None:
            prior_transforms = Compose(
                [
                    LoadImage(image_only=True),
                    EnsureChannelFirst(),
                    NormalizeIntensity(subtrahend=0, divisor=255, dtype="float32"),
                    Resize(spatial_size=[224, 224]),
                ]
            )

        if save_transforms is None:
            save_transforms = Compose(
                [
                    ScaleIntensity(minv=0, maxv=255),
                    AddName(num_images=cfg["local_num_images"]),
                    SaveImage(
                        output_dir=cfg["save_path"],
                        output_ext=save_fmt,
                        separate_folder=False,
                        output_postfix="",
                    ),
                ]
            )

        # Compute inversion
        grad_inversion_engine = FLGradientInversion(
            network=net,
            grad_lst=grad_lst,
            bn_stats=bn_stats,
            model_bn=model_bn,
            prior_transforms=prior_transforms,
            save_transforms=save_transforms,
        )
        best_inputs, targets = grad_inversion_engine(cfg)

        return best_inputs, targets

    def __call__(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext):
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        if not self.cfg:
            self.cfg_file = os.path.join(app_root, self.cfg_file)
            if os.path.isfile(self.cfg_file):
                with open(self.cfg_file, "r") as f:
                    self.cfg = json.load(f)
            else:
                raise ValueError(f"`cfg_file` file does not exist at {self.cfg_file}")
            self.save_path = os.path.join(app_root, self.cfg["save_path"])

        self.logger.info(f"Using full BN stats with momentum {self.bn_momentum} ! \n")

        # get current learner & global model
        engine = fl_ctx.get_engine()
        _learner = engine.get_component(self.learner_name)
        if not _learner:
            raise ValueError(f"No Learner available with name {self.learner_name}")
        elif not isinstance(_learner, Learner):
            raise ValueError(f"Expected `learner` to be of type `Learner` but got type {type(_learner)}")

        if _learner:
            global_model = _learner.global_weights
            if "weights" in global_model:
                global_model = global_model["weights"]

        if global_model is None:
            raise ValueError("No global model exists!")

        # get updates
        weight_updates = dxo.data
        if weight_updates is None or len(weight_updates) == 0:
            raise ValueError(f"No weight_updates available or empty: {weight_updates}")
        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            raise ValueError(f"Expected weight updates to be of data_kind `WEIGHT_DIFF` but got {dxo.data_kind}")
        if self.compute_update_sums:
            sum_updates, sum_bn_updates = 0.0, 0.0
            for k in weight_updates.keys():
                if self.print_names:
                    print(f"Inverting {k}")
                if "bn" in k or "batch" in k or "running" in k:
                    sum_bn_updates += np.sum(np.abs(weight_updates[k]))
                else:
                    sum_updates += np.sum(np.abs(weight_updates[k]))
            self.log_info(
                fl_ctx,
                f"weight update sum {sum_updates}, bn update sum {sum_bn_updates}",
            )
            if sum_updates == 0.0:
                self.log_warning(fl_ctx, "Gradient update sum is zero!")
            if sum_bn_updates == 0.0:
                self.log_warning(fl_ctx, "BN sum is zero!")

        # run inversion
        self.cfg["save_path"] = os.path.join(
            self.save_path,
            f"{fl_ctx.get_identity_name()}_rnd{_learner.current_round}",
        )
        self.cfg["batch_size"] = _learner.batch_size
        self.cfg["local_bs"] = _learner.batch_size
        self.cfg["num_classes"] = _learner.num_class
        self.cfg["lr_local"] = _learner.get_lr()[0]  # use learning rate from first layer
        self.cfg["local_epoch"] = _learner.aggregation_epochs
        self.cfg["local_num_images"] = int(len(_learner.train_loader) * _learner.batch_size)

        self.log_info(fl_ctx, f"Run inversion with config {self.cfg}")
        best_images, _ = self.run_inversion(
            cfg=self.cfg,
            updates=weight_updates,
            global_weights=global_model,
            bn_momentum=self.bn_momentum,
            prior_transforms=self.prior_transforms,
            save_transforms=self.save_transforms,
            save_fmt=self.save_fmt,
        )

        return best_images
