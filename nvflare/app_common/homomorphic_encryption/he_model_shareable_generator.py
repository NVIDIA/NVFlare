# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import tenseal as ts

import nvflare.app_common.homomorphic_encryption.he_constant as he
from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.homomorphic_encryption.homomorphic_encrypt import load_tenseal_context_from_workspace
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator


class HEModelShareableGenerator(FullModelShareableGenerator):
    def __init__(self, tenseal_context_file="server_context.tenseal"):
        """This ShareableGenerator converts between Shareable and Learnable objects.

        This conversion is done with homomorphic encryption (HE) support using TenSEAL https://github.com/OpenMined/TenSEAL.

        Args:
            tenseal_context_file: tenseal context files containing decryption keys and parameters
        """
        super().__init__()
        self.tenseal_context = None
        self.tenseal_context_file = tenseal_context_file
        self.is_encrypted = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.tenseal_context = None

    def add_to_global_weights(self, fl_ctx: FLContext, new_val, base_weights, v_name, encrypt_layers):
        if encrypt_layers is None:
            raise ValueError("encrypted layers info missing!")

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        start_round = fl_ctx.get_prop(AppConstants.START_ROUND, 0)
        if current_round > start_round and encrypt_layers[v_name]:
            if encrypt_layers.get(v_name, False):
                try:
                    binary_global_var = base_weights[v_name]
                    global_var = ts.ckks_vector_from(
                        self.tenseal_context, binary_global_var
                    )  # now the global model weights are encrypted
                    n_vars_total = global_var.size()
                except BaseException as e:
                    raise ValueError("add_to_global_weights Exception", str(e))
        else:
            global_var = base_weights[v_name].ravel()
            n_vars_total = np.size(global_var)

        # update the global model
        updated_vars = new_val + global_var

        if encrypt_layers[v_name]:  # only works with standard aggregation
            self.log_info(fl_ctx, f"serialize encrypted {v_name}")
            updated_vars = updated_vars.serialize()

        return updated_vars, n_vars_total

    def _shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        dxo = from_shareable(shareable)
        enc_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if enc_algorithm != he.HE_ALGORITHM_CKKS:
            raise ValueError("expected encryption algorithm {} but got {}".format(he.HE_ALGORITHM_CKKS, enc_algorithm))

        encrypt_layers = dxo.get_meta_prop(MetaKey.PROCESSED_KEYS)
        if encrypt_layers is None:
            raise ValueError("DXO in shareable missing PROCESSED_KEYS property")

        if len(encrypt_layers) == 0:
            raise ValueError(f"encrypt_layers is empty: {encrypt_layers}")

        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
            return base_model

        base_weights = base_model[ModelLearnableKey.WEIGHTS]

        if dxo.data_kind == DataKind.WEIGHT_DIFF:
            start_time = time.time()
            model_diff = dxo.data
            if not model_diff:
                raise ValueError(f"{self._name} DXO data is empty!")

            n_vars = len(model_diff.items())
            n_params = 0
            for v_name, v_value in model_diff.items():
                self.log_debug(fl_ctx, f"adding {v_name} to global model...")
                # v_value += model[v_name]
                updated_vars, n_vars_total = self.add_to_global_weights(
                    fl_ctx, v_value, base_weights, v_name, encrypt_layers
                )
                n_params += n_vars_total
                base_weights[v_name] = updated_vars
                self.log_debug(fl_ctx, f"assigned new {v_name}")

            end_time = time.time()
            self.log_info(
                fl_ctx,
                f"Updated global model {n_vars} vars with {n_params} params in {end_time - start_time} seconds",
            )
        elif dxo.data_kind == DataKind.WEIGHTS:
            weights = dxo.data
            for v_name in weights.keys():
                if encrypt_layers[v_name]:
                    self.log_info(fl_ctx, f"serialize encrypted {dxo.data_kind}: {v_name}")
                    weights[v_name] = weights[v_name].serialize()
            base_model[ModelLearnableKey.WEIGHTS] = weights
        else:
            raise NotImplementedError(f"data type {dxo.data_kind} not supported!")

        self.log_debug(fl_ctx, "returning model")
        base_model[ModelLearnableKey.META] = dxo.get_meta_props()
        return base_model

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        """Updates the global model in `Learnable` in encrypted space.

        Args:
            shareable: shareable
            fl_ctx: FLContext

        Returns:
            Learnable object
        """
        self.log_info(fl_ctx, "shareable_to_learnable...")
        try:
            return self._shareable_to_learnable(shareable, fl_ctx)
        except BaseException as e:
            self.log_exception(fl_ctx, "error converting shareable to model")
            raise ValueError(f"{self._name} Exception {e}")
