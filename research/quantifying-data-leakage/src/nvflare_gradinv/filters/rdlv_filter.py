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

import os
from typing import Union

import numpy as np
import torch
from monai.data import CacheDataset, load_decathlon_datalist
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    RepeatChannel,
    RepeatChanneld,
    Resize,
    Resized,
    ScaleIntensity,
)

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants

from .gradinv import Inverter
from .image_sim import SimMetric


class RelativeDataLeakageValueFilter(DXOFilter):
    def __init__(
        self,
        data_root: str,
        dataset_json: str,
        frequency: int = 1,
        start_round: int = 0,
        inverter_name: str = "grad_inverter",
        sim_metric_name: str = "sim_metric",
        prior_filename: str = None,
        image_key: str = "image",
        rdlv_reduce: str = "max",
        rdlv_threshold: float = 1.0,
        data_kinds: [str] = None,
        save_best_matches: bool = False,
    ):
        """Filter calling gradient inversion and computing the "Relative Data Leakage Value" (RDLV).
        Data inside the DXO will be removed if any RDLV is above the `rdlv_threshold`.
        See https://arxiv.org/abs/2202.06924 for details.

        Args:
            data_root: Data root for local training set.
            dataset_json: Data list of local training set.
            frequency: Frequency in FL rounds for which to run inversion code and the filter. Defaults to 1.
            start_round: FL round to start inversion. Defaults to 0.
            inverter_name: ID of inverter component.
            sim_metric_name: ID of the similarity metric component.
            prior_filename: Prior image used to initialize the attack and to compute RDLV.
            image_key: Dictionary key used by the data loader.
            rdlv_reduce: Which operation to use to reduce the RDLV values. Defaults to "max".
            rdlv_threshold: Threshold on RDLV to determine whether dxo.data will be passed on or filtered out.
            data_kinds: kinds of DXO data to filter. If None,
                `[DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]` is used.
            save_best_matches: Whether to save the best reconstruction.

        Returns:
            Filtered DXO data. Empty DXO if any of the RDLV values is above `rdlv_threshold`.
            The computed RDLV values and hyperparameters will be saved as NumPy-file (*.npy) in the app_root.
        """
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(
            supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF],
            data_kinds_to_filter=data_kinds,
        )

        self.frequency = frequency
        self.start_round = start_round
        self.inverter_name = inverter_name
        self.sim_metric_name = sim_metric_name
        self.inverter = None
        self.sim_metric = None
        self.image_key = image_key
        self.rdlv_reduce = rdlv_reduce
        self.rdlv_threshold = rdlv_threshold
        self.prior_filename = prior_filename
        self.save_best_matches = save_best_matches

        # TODO: make configurable
        self.data_root = data_root
        self.dataset_json = dataset_json
        self.train_set = "training"
        self.cache_rate = 1.0
        self.num_workers = 2

        # some input checks
        if isinstance(data_root, str):
            if not os.path.isdir(data_root):
                raise ValueError(f"`data_root` directory does not exist at {data_root}")
        else:
            raise ValueError(f"Expected `data_root` of type `str` but received type {type(data_root)}")
        if isinstance(dataset_json, str):
            if not os.path.isfile(dataset_json):
                raise ValueError(f"`dataset_json` file does not exist at {dataset_json}")
        else:
            raise ValueError(f"Expected `dataset_json` of type `str` but received type {type(dataset_json)}")

        self.transform_train = None
        self.train_dataset = None
        self.train_loader = None
        self.prior = None
        self.app_root = None

        self.recon_transforms = Compose([ScaleIntensity(minv=0, maxv=255)])

    def _create_train_loader(self, fl_ctx: FLContext):
        if self.train_loader is None:
            # create data loader for computing RDLV
            self.transform_train = Compose(
                [
                    LoadImaged(keys=[self.image_key]),
                    EnsureChannelFirstd(keys=[self.image_key]),
                    Resized(keys=[self.image_key], spatial_size=[224, 224]),
                    RepeatChanneld(keys=[self.image_key], repeats=3),
                ]
            )

            train_list = load_decathlon_datalist(
                data_list_file_path=self.dataset_json,
                is_segmentation=False,
                data_list_key=self.train_set,
                base_dir=self.data_root,
            )

            self.train_dataset = CacheDataset(
                data=train_list,
                transform=self.transform_train,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=False,
            )

            self.log_info(fl_ctx, f"Training Size ({self.train_set}): {len(train_list)}")

    def _setup(self, fl_ctx: FLContext):
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        self._create_train_loader(fl_ctx)

        # get inverter during first process
        if self.inverter is None:
            engine = fl_ctx.get_engine()
            self.inverter = engine.get_component(self.inverter_name)
            if not self.inverter:
                raise ValueError(f"No Inverter available with name {self.inverter_name}")
            elif not isinstance(self.inverter, Inverter):
                raise ValueError(f"Expected `inverter` to be of type `Inverter` but got type {type(self.inverter)}")

        # get sim_metric during first process
        if self.sim_metric is None:
            engine = fl_ctx.get_engine()
            self.sim_metric = engine.get_component(self.sim_metric_name)
            if not self.sim_metric:
                raise ValueError(f"No SimMetric available with name {self.sim_metric_name}")
            elif not isinstance(self.sim_metric, SimMetric):
                raise ValueError(
                    f"Expected `sim_metric` to be of type `SimMetric` but got type {type(self.sim_metric)}"
                )

        # get prior
        if self.prior_filename and self.prior is None:
            prior_transforms = Compose(
                [
                    LoadImage(image_only=True),
                    EnsureChannelFirst(),
                    Resize(spatial_size=[224, 224]),
                    RepeatChannel(repeats=3),
                ]
            )
            self.prior = prior_transforms(self.prior_filename)

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Compute gradient inversions and compute relative data leakage value (RDLV).
           Filter result based on the given threshold.

        Args:
            dxo: information from client
            shareable: that the dxo belongs to
            fl_ctx: context provided by workflow

        Returns: filtered result.
        """
        self._setup(fl_ctx)

        # Compute inversions
        current_round = dxo.get_meta_prop(AppConstants.CURRENT_ROUND)
        if current_round is None:
            raise ValueError("No current round available!")

        if current_round % self.frequency == 0 and current_round >= self.start_round:
            recons = self.inverter(dxo=dxo, shareable=shareable, fl_ctx=fl_ctx)
            self.log_info(fl_ctx, f"Created reconstructions of shape {np.shape(recons)}")

            # compute (relative) data leakage value
            try:
                self.log_info(
                    fl_ctx,
                    f"Computing sim metrics for {len(self.train_loader)}x{len(recons)} pairs of images",
                )
                (img_recon_sim_reduced, img_recon_sim, best_matches, closest_idx,) = self.compute_rdlv(
                    train_loader=self.train_loader,
                    recons=recons,
                    sim_metric=self.sim_metric,
                    reduce=self.rdlv_reduce,
                    recon_transforms=self.recon_transforms,
                    image_key=self.image_key,
                    prior=self.prior,
                )
                if self.rdlv_reduce == "max":
                    rdlv = np.max(img_recon_sim_reduced, axis=0)
                elif self.rdlv_reduce == "min":
                    rdlv = np.min(img_recon_sim_reduced, axis=0)
                else:
                    raise ValueError(f"No such `rdlv_reduce` supported {self.rdlv_reduce}")
            except Exception as e:
                raise RuntimeError("Computing RDLV failed!") from e

            self.log_info(fl_ctx, f"RDLV: {rdlv}")

            results = {
                "img_recon_sim_reduced": img_recon_sim_reduced,
                "img_recon_sim": img_recon_sim,
                "closest_idx": closest_idx,
                "site": fl_ctx.get_identity_name(),
                "round": current_round,
            }
            if self.save_best_matches:
                results["best_matches"] = best_matches

            save_path = os.path.join(self.app_root, f"rdvl_round{current_round}.npy")
            np.save(save_path, results)

            # Remove data if above threshold
            if np.any(rdlv > self.rdlv_threshold):
                self.log_warning(
                    fl_ctx,
                    f"At least one RDLV of {rdlv} is over the threshold {self.rdlv_threshold}! Remove data.",
                )
                dxo.data = {}

        return dxo

    @staticmethod
    def compute_rdlv(
        train_loader,
        recons,
        sim_metric,
        reduce="max",
        recon_transforms=None,
        image_key="image",
        prior=None,
        fl_ctx: FLContext = None,
    ):
        # TODO: Enforce using only one metric
        # TODO: use recon/reference terminology
        img_recon_sim = np.nan * np.ones((len(train_loader), len(recons), len(sim_metric.metrics)))
        if prior is not None:
            img_prior_sim = np.nan * np.ones((len(train_loader), len(sim_metric.metrics)))
        pairs = []
        imgs = []
        for i, batch_data in enumerate(train_loader):
            img = batch_data[image_key]
            if img.shape[0] != 1:
                raise ValueError(f"Assume batch dimension to be 1 but received {image_key} batch of shape {img.shape}")
            img = img[0, ...]  # assume batch size 1
            imgs.append(img)

            # compute similarity to prior
            if prior is not None:
                _outputs = sim_metric(
                    source=prior,
                    reference=img,
                    fl_ctx=fl_ctx,
                    is_channel_first=True,
                )
                if isinstance(_outputs, dict):
                    for m_idx, m_name in enumerate(sim_metric.metrics):  # TODO: make metrics part of the SimMetrics API
                        img_prior_sim[i, m_idx] = _outputs[m_name]
                else:
                    img_prior_sim[i, ...] = _outputs

            # compute similarity to reconstructions
            for r, recon in enumerate(recons):
                if recon_transforms:
                    recon = recon_transforms(recon)
                _outputs = sim_metric(
                    source=recon,
                    reference=img,
                    fl_ctx=fl_ctx,
                    is_channel_first=True,
                )
                if isinstance(_outputs, dict):
                    for m_idx, m_name in enumerate(sim_metric.metrics):  # TODO: make metrics part of the SimMetrics API
                        img_recon_sim[i, r, m_idx] = _outputs[m_name]
                else:
                    img_recon_sim[i, r, ...] = _outputs
                pairs.append([i, r])

                # Compute relative value
                if prior is not None:
                    for m_idx in range(len(sim_metric.metrics)):
                        img_recon_sim[i, r, m_idx] = (
                            img_recon_sim[i, r, m_idx] - img_prior_sim[i, m_idx]
                        ) / img_prior_sim[i, m_idx]

            if (i + 1) % 32 == 0:
                print(f"processing original image {i + 1} of {len(train_loader)}...")

        if reduce == "max":
            closest_idx = np.argmax(img_recon_sim, axis=1)
        elif reduce == "min":
            closest_idx = np.argmin(img_recon_sim, axis=1)
        elif reduce is None:
            closest_idx = None
        else:
            raise NotImplementedError(f"No such reduce function implemented `{reduce}`")

        if closest_idx is not None:
            img_recon_sim_reduced = []
            best_matches = []
            for i, idx in enumerate(closest_idx):
                # record best value for each image using first metric
                img_recon_sim_reduced.append(img_recon_sim[i, idx[0], ...])
                best_matches.append([imgs[i], recons[idx[0]]])
        else:
            img_recon_sim_reduced = img_recon_sim
            best_matches = []

        return (
            np.array(img_recon_sim_reduced),
            img_recon_sim,
            best_matches,
            closest_idx,
        )
