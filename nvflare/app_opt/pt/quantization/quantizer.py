# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import threading
from typing import Union

import numpy as np
import torch
from bitsandbytes.functional import quantize_4bit, quantize_blockwise

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.pt.quantization.constant import DATA_TYPE, QUANTIZATION_TYPE

from .ada_quant import AdaQuantizer


class ModelQuantizer(DXOFilter):
    def __init__(
        self,
        quantization_type="float16",
    ):
        """Filter to quantize Shareable object to reduce communication burden.

        Args:
            quantization_type: method used for quantization

        """

        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)

        # assign quantization type and check if it is valid
        self.logger.info("Using model quantizer.")
        quantization_type = quantization_type.lower()
        if quantization_type.upper() not in QUANTIZATION_TYPE:
            raise ValueError(f"Invalid quantization type: {quantization_type}, valid: {QUANTIZATION_TYPE}")

        self.quantization_type = quantization_type

        # quantization constants
        self.NP_FP16_MIN = np.finfo(np.float16).min
        self.NP_FP16_MAX = np.finfo(np.float16).max
        self.TS_FP16_MIN = torch.finfo(torch.float16).min
        self.TS_FP16_MAX = torch.finfo(torch.float16).max

        # Lock to ensure thread-safe quantization for concurrent client requests
        # In 1-N server scenarios, multiple clients may process the same task data
        # Key by id(dxo.data) since different DXO wrappers may share the same data dict
        self._quantization_locks = {}  # data_id -> threading.Lock
        self._quantized_results = {}  # data_id -> quantized DXO (metadata for waiting threads)
        self._quantization_errors = {}  # data_id -> Exception (if quantization failed)
        self._data_refcounts = {}  # data_id -> number of threads using this data
        self._locks_lock = threading.Lock()  # protect the dictionaries

    def quantization(self, params: dict, fl_ctx: FLContext):
        n_params = len(params.keys())
        self.log_info(fl_ctx, f"Running quantization on {n_params} variables")
        n_bytes_before = 0
        n_bytes_after = 0
        n_bytes_meta = 0
        n_quant_params = 0
        quant_state = {}
        source_datatype = {}
        for i, param_name in enumerate(params.keys()):
            values = params[param_name]
            quant_state[param_name] = {}

            # check the data type, numpy or torch
            # otherwise error
            if isinstance(values, np.ndarray):
                # if numpy, convert to torch
                source_data_format = "numpy"
            elif isinstance(values, torch.Tensor):
                source_data_format = "torch"
            else:
                raise ValueError(f"Invalid source data type: {type(values)}, valid: numpy or torch")

            # get the data type of the values
            if source_data_format == "numpy":
                source_data_type = values.dtype.name
            elif source_data_format == "torch":
                source_data_type = str(values.dtype).split(".")[1]
            source_datatype[param_name] = source_data_type

            # check if the data type is valid, bool can be returned by adaquant so treating as exception
            if source_data_type != "bool" and source_data_type.upper() not in DATA_TYPE:
                raise ValueError(
                    f"Invalid source data type: {source_data_type}, valid: {DATA_TYPE}, param_name: {param_name}"
                )

            if source_data_type == "bool":
                continue

            if self.quantization_type != "adaquant":
                # get the bits information
                source_data_bits = int(re.findall(r"\d+", source_data_type)[0])
                quantization_bits = int(re.findall(r"\d+", self.quantization_type)[0])

                # only quantize if the quantization type is lower than the source data type
                if quantization_bits >= source_data_bits:
                    self.log_info(
                        fl_ctx,
                        f"Skipping quantization for {param_name}, quantization bit {self.quantization_type} >= source data bit {source_data_type}",
                    )
                    continue
            # add the number of bytes of the values
            n_bytes_before += values.nbytes
            n_quant_params += 1
            if self.quantization_type == "float16":
                if source_data_format == "numpy":
                    # first clamp the values to the range of float16
                    values = np.clip(values, self.NP_FP16_MIN, self.NP_FP16_MAX)
                    # then convert to float16
                    values = values.astype(np.float16)
                elif source_data_format == "torch":
                    # first clamp the values to the range of float16
                    values = torch.clamp(values, self.TS_FP16_MIN, self.TS_FP16_MAX)
                    # then convert to float16
                    values = values.to(torch.float16)
                params[param_name] = values
            elif self.quantization_type in ["blockwise8", "float4", "normfloat4"]:
                # use bitsandbytes to quantize the values
                # input is a tensor, output is a tuple of (quantized tensor, quantized_state)

                # CPU has limited support for 8- and 4-bits quantization
                # For general purpose, here we use GPU
                if source_data_format == "numpy":
                    # if numpy, first convert numpy array to tensor, need to use GPU
                    values_tensor = torch.as_tensor(values).cuda()
                elif source_data_format == "torch":
                    # if torch, directly use the tensor, need to use GPU
                    values_tensor = values.cuda()

                if self.quantization_type == "blockwise8":
                    # quantize the tensor
                    quantized, quantized_state = quantize_blockwise(values_tensor)
                    # add the quantization state and values, keep source data format
                    if source_data_format == "numpy":
                        quant_state[param_name]["absmax"] = quantized_state.absmax.cpu().numpy()
                        quant_state[param_name]["code"] = quantized_state.code.cpu().numpy()
                        values = quantized.cpu().numpy()
                    elif source_data_format == "torch":
                        quant_state[param_name]["absmax"] = quantized_state.absmax.cpu()
                        quant_state[param_name]["code"] = quantized_state.code.cpu()
                        values = quantized.cpu()
                    n_bytes_meta += quant_state[param_name]["absmax"].nbytes
                    n_bytes_meta += quant_state[param_name]["code"].nbytes
                else:
                    # then quantize the tensor
                    if self.quantization_type == "float4":
                        quantized, quantized_state = quantize_4bit(values_tensor, quant_type="fp4")
                    else:
                        quantized, quantized_state = quantize_4bit(values_tensor, quant_type="nf4")
                    # add the quantization state and values, keep source data format
                    quantized_state = quantized_state.as_dict()
                    # prepared the message
                    for state_name, state in quantized_state.items():
                        if isinstance(state, torch.Tensor):
                            if source_data_format == "numpy":
                                # if the state is a tensor, convert it to numpy array
                                quant_state[param_name][state_name] = state.cpu().numpy()
                            elif source_data_format == "torch":
                                # if the state is a tensor, keep it as tensor
                                quant_state[param_name][state_name] = state.cpu()
                            n_bytes_meta += state.nbytes
                        else:
                            quant_state[param_name][state_name] = state
                    # add values
                    if source_data_format == "numpy":
                        values = quantized.cpu().numpy()
                    elif source_data_format == "torch":
                        values = quantized.cpu()

                params[param_name] = values
            elif self.quantization_type == "adaquant":
                # if numpy, first convert numpy array to tensor
                if source_data_format == "numpy":
                    values_tensor = torch.as_tensor(values).cpu()
                elif source_data_format == "torch":
                    values_tensor = values.cpu()
                quantized, quantized_state_dict = AdaQuantizer().quantize(values_tensor)
                if source_data_format == "numpy":
                    params[param_name] = quantized.cpu().numpy() if isinstance(quantized, torch.Tensor) else quantized
                elif source_data_format == "torch":
                    params[param_name] = (
                        quantized.cpu() if isinstance(quantized, torch.Tensor) else torch.as_tensor(quantized)
                    )

                if quantized_state_dict:
                    quant_state[param_name] = quantized_state_dict

                for state_name, state in quantized_state_dict.items():
                    if isinstance(state, (torch.Tensor, np.ndarray)):
                        n_bytes_meta += state.nbytes
            n_bytes_after += params[param_name].nbytes

        self.log_info(
            fl_ctx,
            f"Quantized {n_quant_params}/{n_params} params."
            f" Before quantization: {n_bytes_before / (1024 ** 2):.2f} MB."
            f" After quantization: {n_bytes_after / (1024 ** 2):.2f} MB with meta: {n_bytes_meta / (1024 ** 2):.2f} MB.",
        )
        return params, quant_state, source_datatype

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Filter process apply to the Shareable object.

        Args:
            dxo: data to be processed
            shareable: that the dxo belongs to
            fl_ctx: FLContext

        Returns: DXO object with quantized weights

        """

        self.log_info(fl_ctx, "Running quantization...")

        # for already quantized message, skip quantization
        # The reason in this current example:
        # server job in this case is 1-N communication with identical quantization operation
        # the first communication to client will apply quantization and change the data on the server
        # thus the subsequent communications to the rest of clients will no longer need to apply quantization
        # This will not apply to client job, since the client job will be 1-1 and quantization applies to each client
        # Potentially:
        # - If clients talks to each other, it will also be 1-N and same rule applies
        # - If 1-N server-client filters can be different (Filter_1 applies to server-client_subset_1, etc.), then
        # a deep copy of the server data should be made by filter before applying a different filter

        # Use the data dictionary's id as the key for locking
        # In 1-N scenarios, different DXO wrapper objects may share the same underlying data dict
        # We need to synchronize based on the actual data being quantized, not the wrapper
        data_id = id(dxo.data)

        # Increment reference count - this thread is now using this data_id
        with self._locks_lock:
            if data_id not in self._quantization_locks:
                self._quantization_locks[data_id] = threading.Lock()
                self._data_refcounts[data_id] = 0
            self._data_refcounts[data_id] += 1
            data_lock = self._quantization_locks[data_id]

        try:
            # Acquire the data-specific lock to ensure only one thread quantizes this data
            # Other threads will wait here until quantization is complete
            with data_lock:
                # Check if quantization failed in another thread
                if data_id in self._quantization_errors:
                    self.log_info(fl_ctx, "Quantization failed in another thread, re-raising error")
                    raise self._quantization_errors[data_id]
                # Check if quantization result already exists (from another thread)
                elif data_id in self._quantized_results:
                    self.log_info(fl_ctx, "Already quantized by another thread, reusing result")
                    new_dxo = self._quantized_results[data_id]
                # Check if this DXO was already quantized (e.g., from previous filter)
                elif dxo.get_meta_prop("quantized_flag"):
                    self.log_info(fl_ctx, "Already quantized, skip quantization")
                    new_dxo = dxo
                # Perform quantization
                else:
                    try:
                        # Apply quantization
                        # The lock ensures mutual exclusion, so only one thread can execute this
                        quantized_params, quant_state, source_datatype = self.quantization(
                            params=dxo.data, fl_ctx=fl_ctx
                        )

                        # Compose new DXO with quantized data
                        # Add quant_state to the new DXO meta
                        new_dxo = DXO(data_kind=dxo.data_kind, data=quantized_params, meta=dxo.meta)
                        new_dxo.set_meta_prop(key=MetaKey.PROCESSED_ALGORITHM, value=self.quantization_type)
                        new_dxo.set_meta_prop(key="quant_state", value=quant_state)
                        new_dxo.set_meta_prop(key="source_datatype", value=source_datatype)
                        new_dxo.set_meta_prop(key="quantized_flag", value=True)

                        self.log_info(fl_ctx, f"Quantized to {self.quantization_type}")

                        # Store the result so other waiting threads can use it
                        self._quantized_results[data_id] = new_dxo

                    except Exception as e:
                        # Store the error so other waiting threads can re-raise it
                        # without retrying the failed quantization
                        self.log_error(fl_ctx, f"Quantization failed: {e}")
                        self._quantization_errors[data_id] = e
                        raise

            return new_dxo

        finally:
            # Decrement reference count and clean up if this was the last thread
            with self._locks_lock:
                self._data_refcounts[data_id] -= 1

                # If no more threads need this data, clean up
                if self._data_refcounts[data_id] == 0:
                    self._quantized_results.pop(data_id, None)
                    self._quantization_errors.pop(data_id, None)
                    self._quantization_locks.pop(data_id, None)
                    self._data_refcounts.pop(data_id, None)
