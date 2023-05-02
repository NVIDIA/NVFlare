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

from typing import Dict

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.app_constant import AppConstants


class FedCEAssembler(Assembler):
    def __init__(self, fedce_mode, model):
        super().__init__(data_kind=DataKind.WEIGHT_DIFF)
        # mode, plus or times
        self.fedce_mode = fedce_mode
        self.model = model
        self.fedce_cos_param_list = []
        # Aggregator needs to keep record of historical
        # cosine similarity for FedCM coefficients
        self.fedce_cos_sim = {}
        self.fedce_coef = {}

    def _initialize(self, fl_ctx: FLContext):
        # convert str model description to model
        if isinstance(self.model, str):
            # treat it as model component ID
            model_component_id = self.model
            engine = fl_ctx.get_engine()
            self.model = engine.get_component(model_component_id)
            if not self.model:
                self.system_panic(
                    reason=f"cannot find model component '{model_component_id}'",
                    fl_ctx=fl_ctx,
                )
                return
            if not isinstance(self.model, torch.nn.Module):
                self.system_panic(
                    reason=f"expect model component '{model_component_id}' to be torch.nn.Module but got {type(self.model_selector)}",
                    fl_ctx=fl_ctx,
                )
                return
        elif self.model and not isinstance(self.model, torch.nn.Module):
            self.system_panic(
                reason=f"expect model to be torch.nn.Module but got {type(self.model)}",
                fl_ctx=fl_ctx,
            )
            return
        # only include params requires_grad for cosine similarity computation
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fedce_cos_param_list.append(name)
        self.log_info(fl_ctx, "FedCE model assembler initialized")

    def get_model_params(self, dxo: DXO):
        data = dxo.data
        meta = dxo.meta
        return {"model": data, "fedce_minus_val": meta["fedce_minus_val"]}

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        site_list = data.keys()

        fedce_minus_vals = []
        self.fedce_cos_sim[current_round] = {}

        for site in site_list:
            if current_round == 0:
                # round 0, initialize uniform fedce_coef
                self.fedce_coef[site] = 1 / len(site_list)
            # get minus_val from submissions
            fedce_minus_vals.append(data[site]["fedce_minus_val"])

        # generate consensus gradient with current FedCE coefficients
        consensus_grad = []
        global_weights = self.model.state_dict()
        for idx, name in enumerate(global_weights):
            if name in self.fedce_cos_param_list:
                temp = torch.zeros_like(global_weights[name])
                for site in site_list:
                    temp += self.fedce_coef[site] * torch.as_tensor(data[site]["model"][name])
                consensus_grad.append(temp.data.view(-1))

        # flatten for cosine similarity computation
        consensus_grads_vec = torch.cat(consensus_grad).to("cpu")

        # generate minus gradients and compute cosine similarity
        for site in site_list:
            site_grad = []
            for name in self.fedce_cos_param_list:
                site_grad.append(torch.as_tensor(data[site]["model"][name]).data.view(-1))
            site_grads_vec = torch.cat(site_grad).to("cpu")
            # minus gradient
            minus_grads_vec = consensus_grads_vec - self.fedce_coef[site] * site_grads_vec
            # compute cosine similarity
            fedce_cos_sim_site = (
                torch.cosine_similarity(site_grads_vec, minus_grads_vec, dim=0).detach().cpu().numpy().item()
            )
            # append to record dict
            self.fedce_cos_sim[current_round][site] = fedce_cos_sim_site

        # compute cos_weights and minus_vals based on the record for each site
        fedce_cos_weights = []
        for site in site_list:
            # cosine similarity
            cos_accu_avg = np.mean([self.fedce_cos_sim[i][site] for i in range(current_round + 1)])
            fedce_cos_weights.append(1.0 - cos_accu_avg)

        # normalize
        fedce_cos_weights /= np.sum(fedce_cos_weights)
        fedce_cos_weights = np.clip(fedce_cos_weights, a_min=1e-3, a_max=None)
        fedce_minus_vals /= np.sum(fedce_minus_vals)
        fedce_minus_vals = np.clip(fedce_minus_vals, a_min=1e-3, a_max=None)

        # two aggregation strategies
        if self.fedce_mode == "times":
            new_fedce_coef = [c_w * mv_w for c_w, mv_w in zip(fedce_cos_weights, fedce_minus_vals)]
        elif self.fedce_mode == "plus":
            new_fedce_coef = [c_w + mv_w for c_w, mv_w in zip(fedce_cos_weights, fedce_minus_vals)]
        else:
            raise NotImplementedError

        # normalize again
        new_fedce_coef /= np.sum(new_fedce_coef)
        new_fedce_coef = np.clip(new_fedce_coef, a_min=1e-3, a_max=None)

        # update fedce_coef
        fedce_coef = {}
        idx = 0
        for site in site_list:
            fedce_coef[site] = new_fedce_coef[idx]
            idx += 1

        # compute global model update with the new fedce weights
        global_updates = {}
        for idx, name in enumerate(global_weights):
            temp = torch.zeros_like(global_weights[name], dtype=torch.float32)
            for site in site_list:
                weight = fedce_coef[site]
                temp += weight * data[site]["model"][name]
            global_updates[name] = temp.detach().cpu().numpy()

        meta = {"fedce_coef": fedce_coef}
        dxo = DXO(data_kind=self.expected_data_kind, data=global_updates, meta=meta)

        return dxo
