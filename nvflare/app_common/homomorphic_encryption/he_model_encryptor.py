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

import re
import time

import numpy as np
import tenseal as ts
from tenseal.tensors.ckksvector import CKKSVector

import nvflare.app_common.homomorphic_encryption.he_constant as he
from nvflare.apis.dxo import DXO, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.homomorphic_encryption.homomorphic_encrypt import (
    count_encrypted_layers,
    load_tenseal_context_from_workspace,
)


class HEModelEncryptor(Filter):
    def __init__(
        self,
        tenseal_context_file="client_context.tenseal",
        encrypt_layers=None,
        aggregation_weights=None,
        weigh_by_local_iter=True,
    ):
        """Filter to encrypt Shareable object using homomorphic encryption (HE) with TenSEAL https://github.com/OpenMined/TenSEAL.

        Args:
            tenseal_context_file: tenseal context files containing encryption keys and parameters
            encrypt_layers: if not specified (None), all layers are being encrypted;
                            if list of variable/layer names, only specified variables are encrypted;
                            if string containing regular expression (e.g. "conv"), only matched variables are being encrypted.
            aggregation_weights: dictionary of client aggregation `{"client1": 1.0, "client2": 2.0, "client3": 3.0}`;
                                 defaults to a weight of 1.0 if not specified.
            weigh_by_local_iter: If true, multiply client weights on first before encryption (default: `True` which is recommended for HE)

        """
        super().__init__()
        self.logger.info("Using HE model encryptor.")
        self.tenseal_context = None
        self.tenseal_context_file = tenseal_context_file
        self.aggregation_weights = aggregation_weights or {}
        self.logger.info(f"client weights control: {self.aggregation_weights}")
        self.weigh_by_local_iter = weigh_by_local_iter
        self.n_iter = None
        self.client_name = None
        self.aggregation_weight = None

        # choose which layers to encrypt
        if encrypt_layers is not None:
            if not (isinstance(encrypt_layers, list) or isinstance(encrypt_layers, str)):
                raise ValueError(
                    "Must provide a list of layer names or a string for regex matching, but got {}".format(
                        type(encrypt_layers)
                    )
                )
        if isinstance(encrypt_layers, list):
            for encrypt_layer in encrypt_layers:
                if not isinstance(encrypt_layer, str):
                    raise ValueError(
                        "encrypt_layers needs to be a list of layer names to encrypt, but found element of type {}".format(
                            type(encrypt_layer)
                        )
                    )
            self.encrypt_layers = encrypt_layers
            self.logger.info(f"Encrypting {len(encrypt_layers)} layers")
        elif isinstance(encrypt_layers, str):
            self.encrypt_layers = re.compile(encrypt_layers) if encrypt_layers else None
            self.logger.info(f'Encrypting all layers based on regex matches with "{encrypt_layers}"')
        else:
            self.encrypt_layers = [True]  # needs to be list for logic in encryption()
            self.logger.info("Encrypting all layers")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.tenseal_context = None

    def encryption(self, params, fl_ctx: FLContext):
        n_params = len(params.keys())
        self.log_info(fl_ctx, f"Running HE Encryption algorithm on {n_params} variables")

        # parse regex encrypt layers
        if isinstance(self.encrypt_layers, re.Pattern):
            re_pattern = self.encrypt_layers
            self.encrypt_layers = []
            for var_name in params:
                if re_pattern.search(var_name):
                    self.encrypt_layers.append(var_name)
            self.log_info(fl_ctx, f"Regex found {self.encrypt_layers} matching layers.")
            if len(self.encrypt_layers) == 0:
                raise ValueError(f"No matching layers found with regex {re_pattern}")

        start_time = time.time()
        n_encrypted, n_total = 0, 0
        encryption_dict = {}
        vmins, vmaxs = [], []
        for i, param_name in enumerate(params.keys()):
            values = params[param_name].ravel()
            _n = np.size(values)
            n_total += _n

            # weigh before encryption
            if self.aggregation_weight:
                values = values * np.float(self.aggregation_weight)
            if self.weigh_by_local_iter:
                values = values * np.float(self.n_iter)

            if param_name in self.encrypt_layers or self.encrypt_layers[0] is True:
                self.log_info(fl_ctx, f"Encrypting vars {i+1} of {n_params}: {param_name} with {_n} values")
                vmin = np.min(params[param_name])
                vmax = np.max(params[param_name])
                vmins.append(vmin)
                vmaxs.append(vmax)
                params[param_name] = ts.ckks_vector(self.tenseal_context, values).serialize()
                encryption_dict[param_name] = True
                n_encrypted += _n
            elif isinstance(values, CKKSVector):
                self.log_error(
                    fl_ctx, f"{i} of {n_params}: {param_name} = {np.shape(params[param_name])} already encrypted!"
                )
                raise ValueError("This should not happen!")
            else:
                params[param_name] = values
                encryption_dict[param_name] = False
        end_time = time.time()
        if n_encrypted == 0:
            raise ValueError("Nothing has been encrypted! Check provided encrypt_layers list of layer names or regex.")
        self.log_info(
            fl_ctx,
            f"Encryption time for {n_encrypted} of {n_total} params"
            f" (encrypted value range [{np.min(vmins)}, {np.max(vmaxs)}])"
            f" {end_time - start_time} seconds.",
        )
        # params is a dictionary.  keys are layer names.  values are either weights or serialized ckks_vector of weights.
        # encryption_dict: keys are layer names.  values are True for serialized ckks_vectors, False elsewhere.
        return params, encryption_dict

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Filter process apply to the Shareable object.

        Args:
            shareable: shareable
            fl_ctx: FLContext

        Returns:
            a Shareable object with encrypted model weights

        """
        rc = shareable.get_return_code()
        if rc != ReturnCode.OK:
            # don't process if RC not OK
            return shareable

        dxo = from_shareable(shareable)

        if self.aggregation_weights:
            self.client_name = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, default="?")
            self.aggregation_weight = self.aggregation_weights.get(self.client_name, 1.0)
            self.log_info(fl_ctx, f"weighting {self.client_name} by aggregation weight {self.aggregation_weight}")

        if self.weigh_by_local_iter:
            self.n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, None)
            if self.n_iter is None:
                raise ValueError("DXO data does not have local iterations for weighting!")
            self.log_info(fl_ctx, f"weighting by local iter before encryption with {self.n_iter}")
        try:
            new_dxo = self._process(dxo, fl_ctx)
            new_dxo.update_shareable(shareable)
        except BaseException as e:
            self.log_exception(fl_ctx, f"Exception occurred: {e}")

        return shareable

    def _process(self, dxo: DXO, fl_ctx: FLContext) -> DXO:
        self.log_info(fl_ctx, "Running HE encryption...")
        encrypted_params, encryption_dict = self.encryption(params=dxo.data, fl_ctx=fl_ctx)
        new_dxo = DXO(data_kind=dxo.data_kind, data=encrypted_params, meta=dxo.meta)
        new_dxo.set_meta_prop(key=MetaKey.PROCESSED_KEYS, value=encryption_dict)
        new_dxo.set_meta_prop(key=MetaKey.PROCESSED_ALGORITHM, value=he.HE_ALGORITHM_CKKS)
        n_encrypted, n_total = count_encrypted_layers(encryption_dict)
        self.log_info(fl_ctx, f"{n_encrypted} of {n_total} layers encrypted")
        return new_dxo
