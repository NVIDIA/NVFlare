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
"""Decomposers for types from app_common and Machine Learning libraries."""
import os
from abc import ABC
from io import BytesIO
from typing import Any

import numpy as np

import nvflare.fuel.utils.fobs.dots as dots
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposers.via_file import ViaFileDecomposer

_NPZ_EXTENSION = ".npz"


class NumpyScalarDecomposer(fobs.Decomposer, ABC):
    """Decomposer base class for all numpy types with item method."""

    def decompose(self, target: Any, manager: DatumManager = None) -> Any:
        return target.item()

    def recompose(self, data: Any, manager: DatumManager = None) -> np.ndarray:
        return self.supported_type()(data)


class Float64ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.float64


class Float32ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.float32


class Int64ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.int64


class Int32ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.int32


class NumpyArrayDecomposer(ViaFileDecomposer):

    def __init__(self):
        # by default do not use file downloading.
        ViaFileDecomposer.__init__(self, 0, "np_")

    def supported_type(self):
        return np.ndarray

    def get_bytes_dot(self) -> int:
        return dots.NUMPY_BYTES

    def get_file_dot(self) -> int:
        return dots.NUMPY_FILE

    def dump_to_file(self, items: dict, path: str, fobs_ctx: dict):
        if not path.endswith(_NPZ_EXTENSION):
            path += _NPZ_EXTENSION
        self.logger.info(f"NP: dumping {len(items)} arrays to file {path}")
        try:
            np.savez(allow_pickle=False, file=path, **items)
            return path, None
        except Exception as e:
            self.logger.error(f"exception dumping NP to file: {e}")
            raise e

    def load_from_file(self, path: str, fobs_ctx: dict, meta: dict = None) -> Any:
        result = {}
        with np.load(path, allow_pickle=False) as npz_obj:
            for k in npz_obj.files:
                result[k] = npz_obj[k]
        self.logger.info(f"loaded {len(result)} array(s) from file {path}")
        return result

    def native_decompose(self, target: np.ndarray, manager: DatumManager = None) -> bytes:
        stream = BytesIO()
        np.save(stream, target, allow_pickle=False)
        return stream.getvalue()

    def native_recompose(self, data: bytes, manager: DatumManager = None) -> np.ndarray:
        stream = BytesIO(data)
        return np.load(stream, allow_pickle=False)


def register():
    if register.registered:
        return

    fobs.register_folder(os.path.dirname(__file__), __package__)

    register.registered = True


register.registered = False
