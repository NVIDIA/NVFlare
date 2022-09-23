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
from typing import Union

import numpy as np
import tenseal as ts
from tenseal.tensors.ckksvector import CKKSVector

import nvflare.app_common.homomorphic_encryption.he_constant as he
from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.homomorphic_encryption.homomorphic_encrypt import (
    count_encrypted_layers,
    load_tenseal_context_from_workspace,
)


class HEModelDecryptor(DXOFilter):
    def __init__(self, tenseal_context_file="client_context.tenseal", data_kinds: [str] = None):
        """Filter to decrypt Shareable object using homomorphic encryption (HE) with TenSEAL https://github.com/OpenMined/TenSEAL.

        Args:
            tenseal_context_file: tenseal context files containing decryption keys and parameters
            data_kinds: kinds of DXOs to filter

        """
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF], data_kinds_to_filter=data_kinds)

        self.logger.info("Using HE model decryptor.")
        self.tenseal_context = None
        self.tenseal_context_file = tenseal_context_file

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.tenseal_context = None

    def decryption(self, params, encrypted_layers, fl_ctx: FLContext):

        n_params = len(params.keys())
        self.log_info(fl_ctx, f"Running HE Decryption algorithm {n_params} variables")
        if encrypted_layers is None:
            raise ValueError("encrypted_layers is None!")

        start_time = time.time()
        n_decrypted, n_total = 0, 0
        for i, param_name in enumerate(params.keys()):
            values = params[param_name]
            if encrypted_layers[param_name]:
                _n = values.size()
                n_total += _n
                if isinstance(values, CKKSVector):
                    self.log_info(fl_ctx, f"Decrypting vars {i+1} of {n_params}: {param_name} with {_n} values")
                    params[param_name] = values.decrypt()
                    n_decrypted += _n
                else:
                    self.log_info(
                        fl_ctx,
                        f"{i} of {n_params}: {param_name} = {np.shape(params[param_name])} already decrypted (RAW)!",
                    )
                    raise ValueError("Should be encrypted at this point!")
            else:
                params[param_name] = values
        end_time = time.time()
        self.log_info(fl_ctx, f"Decryption time for {n_decrypted} of {n_total} params {end_time - start_time} seconds.")

        return params

    def to_ckks_vector(self, params, encrypted_layers, fl_ctx: FLContext):
        """Convert encrypted arrays to CKKS vector."""
        if encrypted_layers is None:
            raise ValueError("encrypted_layers is None!")
        start_time = time.time()
        result = {}
        n_total = 0
        self.log_info(fl_ctx, f"params {len(params)} {type(params)}")
        for v in params:
            ndarray = params[v]
            if encrypted_layers[v]:
                if np.size(ndarray) > 1:
                    raise ValueError(f"size of {v} should not be larger 1 but is {np.size(ndarray)}!")
                result[v] = ts.ckks_vector_from(self.tenseal_context, ndarray)
                n = result[v].size()
            else:
                result[v] = ndarray
                n = np.size(ndarray)
            n_total += n
        end_time = time.time()
        self.log_info(fl_ctx, f"to_ckks_vector time for {n_total} values: {end_time - start_time} seconds.")
        return result

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Filter process apply to the Shareable object.

        Args:
            shareable: shareable
            fl_ctx: FLContext

        Returns: DXO object with decrypted weights
        """
        self.log_info(fl_ctx, "Running decryption...")
        encrypted_layers = dxo.get_meta_prop(key=MetaKey.PROCESSED_KEYS, default=None)
        if not encrypted_layers:
            self.log_warning(fl_ctx, "dxo does not contain PROCESSED_KEYS (do nothing)")
            return None

        encrypted_algo = dxo.get_meta_prop(key=MetaKey.PROCESSED_ALGORITHM, default=None)
        if encrypted_algo != he.HE_ALGORITHM_CKKS:
            self.log_error(fl_ctx, "shareable is not HE CKKS encrypted")
            return None

        n_encrypted, n_total = count_encrypted_layers(encrypted_layers)
        self.log_info(fl_ctx, f"{n_encrypted} of {n_total} layers encrypted")
        decrypted_params = self.decryption(
            params=self.to_ckks_vector(params=dxo.data, encrypted_layers=encrypted_layers, fl_ctx=fl_ctx),
            encrypted_layers=encrypted_layers,
            fl_ctx=fl_ctx,
        )

        dxo.data = decrypted_params
        dxo.remove_meta_props([MetaKey.PROCESSED_ALGORITHM, MetaKey.PROCESSED_KEYS])
        dxo.update_shareable(shareable)

        return dxo
