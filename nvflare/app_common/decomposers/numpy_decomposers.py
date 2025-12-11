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
from typing import Any, Tuple

import numpy as np

import nvflare.fuel.utils.fobs.dots as dots
from nvflare.app_common.np.np_downloader import ArrayDownloadable, download_arrays
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Downloadable
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposers.via_downloader import ViaDownloaderDecomposer

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


class NumpyArrayDecomposer(ViaDownloaderDecomposer):

    def __init__(self):
        ViaDownloaderDecomposer.__init__(self, 1024 * 1024 * 2, "np_")

    def supported_type(self):
        return np.ndarray

    def get_download_dot(self) -> int:
        return dots.NUMPY_DOWNLOAD

    def to_downloadable(self, items: dict, max_chunk_size: int, fobs_ctx: dict) -> Downloadable:
        return ArrayDownloadable(items, max_chunk_size)

    def download(
        self,
        from_fqcn: str,
        ref_id: str,
        per_request_timeout: float,
        cell: Cell,
        secure=False,
        optional=False,
        abort_signal=None,
    ) -> Tuple[str, dict]:
        return download_arrays(
            from_fqcn,
            ref_id,
            per_request_timeout,
            cell,
            secure,
            optional,
            abort_signal,
        )

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
