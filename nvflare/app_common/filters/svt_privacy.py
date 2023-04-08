# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class SVTPrivacy(DXOFilter):
    def __init__(
        self, fraction=0.1, epsilon=0.1, noise_var=0.1, gamma=1e-5, tau=1e-6, data_kinds: [str] = None, replace=True
    ):
        """Implementation of the standard Sparse Vector Technique (SVT) differential privacy algorithm.

        lambda_rho = gamma * 2.0 / epsilon
        threshold = tau + np.random.laplace(scale=lambda_rho)

        Args:
            fraction (float, optional): used to determine dataset threshold. Defaults to 0.1.
            epsilon (float, optional): Defaults to 0.1.
            noise_var (float, optional): additive noise. Defaults to 0.1.
            gamma (float, optional): Defaults to 1e-5.
            tau (float, optional): Defaults to 1e-6.
            data_kinds (str, optional): Defaults to None.
            replace (bool): whether to sample with replacement. Defaults to True.
        """
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF], data_kinds_to_filter=data_kinds)

        self.frac = fraction  # fraction of the model to upload
        self.eps_1 = epsilon
        self.eps_2 = None  # to be derived from eps_1
        self.eps_3 = noise_var
        self.gamma = gamma
        self.tau = tau
        self.replace = replace

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Compute the differentially private SVT.

        Args:
            dxo: information from client
            shareable: that the dxo belongs to
            fl_ctx: context provided by workflow

        Returns: filtered result.
        """
        self.log_debug(fl_ctx, "inside filter")
        model_diff = dxo.data
        total_steps = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1)

        delta_w = np.concatenate([model_diff[name].ravel() / np.float64(total_steps) for name in sorted(model_diff)])
        self.log_info(
            fl_ctx,
            "Delta_w: Max abs: {}, Min abs: {}, Median abs: {}.".format(
                np.max(np.abs(delta_w)), np.min(np.abs(delta_w)), np.median(np.abs(delta_w))
            ),
        )

        # precompute thresholds
        n_upload = np.minimum(np.ceil(np.float64(delta_w.size) * self.frac), np.float64(delta_w.size))

        # eps_1: threshold with noise
        lambda_rho = self.gamma * 2.0 / self.eps_1
        threshold = self.tau + np.random.laplace(scale=lambda_rho)
        # eps_2: query with noise
        self.eps_2 = self.eps_1 * (2.0 * n_upload) ** (2.0 / 3.0)
        lambda_nu = self.gamma * 4.0 * n_upload / self.eps_2
        self.logger.info(
            "total params: %s, epsilon: %s, "
            "perparam budget %s, threshold tau: %s + f(eps_1) = %s, "
            "clip gamma: %s",
            delta_w.size,
            self.eps_1,
            self.eps_1 / n_upload,
            self.tau,
            threshold,
            self.gamma,
        )

        # selecting weights with additive noise
        accepted, candidate_idx = [], np.arange(delta_w.size)
        _clipped_w = np.abs(np.clip(delta_w, a_min=-self.gamma, a_max=self.gamma))
        while len(accepted) < n_upload:
            nu_i = np.random.laplace(scale=lambda_nu, size=candidate_idx.shape)
            above_threshold = (_clipped_w[candidate_idx] + nu_i) >= threshold
            accepted += candidate_idx[above_threshold].tolist()
            candidate_idx = candidate_idx[~above_threshold]
            self.log_info(fl_ctx, "selected {} responses, requested {}".format(len(accepted), n_upload))
        accepted = np.random.choice(accepted, size=np.int64(n_upload), replace=self.replace)
        # eps_3 return with noise
        noise = np.random.laplace(scale=self.gamma * 2.0 / self.eps_3, size=accepted.shape)
        self.log_info(fl_ctx, "noise max: {}, median {}".format(np.max(np.abs(noise)), np.median(np.abs(noise))))
        delta_w[accepted] = np.clip(delta_w[accepted] + noise, a_min=-self.gamma, a_max=self.gamma)
        candidate_idx = list(set(np.arange(delta_w.size)) - set(accepted))
        delta_w[candidate_idx] = 0.0

        # resume original format
        dp_w, _start = {}, 0
        for name in sorted(model_diff):
            if np.ndim(model_diff[name]) == 0:
                dp_w[name] = model_diff[name]
                _start += 1
                continue
            value = delta_w[_start : (_start + model_diff[name].size)]
            dp_w[name] = value.reshape(model_diff[name].shape) * np.float64(total_steps)
            _start += model_diff[name].size

        # We update the shareable weights only.  Headers are unchanged.
        dxo.data = dp_w
        return dxo
