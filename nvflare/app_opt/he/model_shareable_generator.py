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
import tenseal as ts

from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, model_learnable_to_dxo
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.he import decomposers
from nvflare.app_opt.he.constant import HE_ALGORITHM_CKKS
from nvflare.app_opt.he.homomorphic_encrypt import (
    deserialize_nested_dict,
    load_tenseal_context_from_workspace,
    serialize_nested_dict,
)
from nvflare.security.logging import secure_format_exception


def add_to_global_weights(new_val, base_weights, v_name):
    try:
        global_var = base_weights[v_name]

        if isinstance(new_val, np.ndarray):
            new_val = new_val.ravel()

        if isinstance(global_var, np.ndarray):
            global_var = global_var.ravel()
            n_vars_total = np.size(global_var)
        elif isinstance(global_var, ts.CKKSVector):
            n_vars_total = global_var.size()
        else:
            raise ValueError(f"global_var has type {type(global_var)} which is not supported.")

        # update the global model
        updated_vars = new_val + global_var

    except Exception as e:
        raise ValueError(f"add_to_global_weights Exception: {secure_format_exception(e)}") from e

    return updated_vars, n_vars_total


class HEModelShareableGenerator(ShareableGenerator):
    def __init__(self, tenseal_context_file="server_context.tenseal"):
        """This ShareableGenerator converts between Shareable and Learnable objects.

        This conversion is done with homomorphic encryption (HE) support using
        TenSEAL https://github.com/OpenMined/TenSEAL.

        Args:
            tenseal_context_file: tenseal context files containing TenSEAL context
        """
        super().__init__()
        self.tenseal_context = None
        self.tenseal_context_file = tenseal_context_file

        decomposers.register()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.tenseal_context = None

    def _shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        dxo = from_shareable(shareable)
        enc_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if enc_algorithm != HE_ALGORITHM_CKKS:
            raise ValueError("expected encryption algorithm {} but got {}".format(HE_ALGORITHM_CKKS, enc_algorithm))

        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
            return base_model
        deserialize_nested_dict(base_model, context=self.tenseal_context)

        base_weights = base_model[ModelLearnableKey.WEIGHTS]

        if dxo.data_kind == DataKind.WEIGHT_DIFF:
            start_time = time.time()
            model_diff = dxo.data
            if not model_diff:
                raise ValueError(f"{self._name} DXO data is empty!")

            deserialize_nested_dict(model_diff, context=self.tenseal_context)

            n_vars = len(model_diff.items())
            n_params = 0
            for v_name, v_value in model_diff.items():
                self.log_debug(fl_ctx, f"adding {v_name} to global model...")
                updated_vars, n_vars_total = add_to_global_weights(v_value, base_weights, v_name)
                n_params += n_vars_total
                base_weights[v_name] = updated_vars
                self.log_debug(fl_ctx, f"assigned new {v_name}")

            end_time = time.time()
            self.log_info(
                fl_ctx,
                f"Updated global model {n_vars} vars with {n_params} params in {end_time - start_time} seconds",
            )
        elif dxo.data_kind == DataKind.WEIGHTS:
            base_model[ModelLearnableKey.WEIGHTS] = dxo.data
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
        except Exception as e:
            self.log_exception(fl_ctx, "error converting shareable to model")
            raise ValueError(f"{self._name} Exception {secure_format_exception(e)}") from e

    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """
        # serialize model_learnable
        serialize_nested_dict(model_learnable)
        dxo = model_learnable_to_dxo(model_learnable)
        return dxo.to_shareable()
