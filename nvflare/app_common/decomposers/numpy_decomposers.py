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
from typing import Any

import numpy as np

import nvflare.fuel.utils.fobs.dats as dats
from nvflare.fuel.utils.fobs.decomposers.via_file import ViaFileDecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager


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
    def supported_type(self):
        return np.ndarray

    def supported_dats(self):
        return [dats.LOCAL_NUMPY, dats.REMOTE_NUMPY]

    def get_local_dat(self) -> int:
        return dats.LOCAL_NUMPY

    def get_remote_dat(self) -> int:
        return dats.REMOTE_NUMPY

    def dump_to_file(self, items: dict, path: str):
        print(f"NP: dumping {len(items)} arrays to file {path}")
        try:
            np.savez(allow_pickle=False, file=path, **items)
            return path + ".npz"
        except Exception as e:
            print(f"exception dumping NP to file: {e}")

    def load_from_file(self, path: str) -> Any:
        result = {}
        with np.load(path, allow_pickle=False) as npz_obj:
            for k in npz_obj.files:
                result[k] = npz_obj[k]
        return result


def register():
    if register.registered:
        return

    fobs.register_folder(os.path.dirname(__file__), __package__)

    register.registered = True


register.registered = False
