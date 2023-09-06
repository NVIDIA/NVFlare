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

from typing import Dict, Hashable, Mapping, Optional

import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import MapTransform, Transform
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_tensor


class NormalizeIntensityRange(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, a_min: float, a_max: float, subtrahend: float, divisor: float, dtype: DtypeLike = np.float32):
        if a_min > a_max:
            raise ValueError("a_min must be lesser than a_max.")

        self.a_min = a_min
        self.a_max = a_max

        self.subtrahend = subtrahend
        self.divisor = divisor

        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        subtrahend: Optional[float] = None,
        divisor: Optional[float] = None,
        dtype: Optional[DtypeLike] = None,
    ) -> NdarrayOrTensor:
        if subtrahend is None:
            subtrahend = self.subtrahend
        if divisor is None:
            divisor = self.divisor
        if dtype is None:
            dtype = self.dtype

        img = convert_to_tensor(img, track_meta=get_track_meta())

        img = clip(img, self.a_min, self.a_max)
        img = (img - subtrahend) / divisor

        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]
        return ret


class NormalizeIntensityRanged(MapTransform):
    backend = NormalizeIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        subtrahend: float,
        divisor: float,
        dtype: Optional[DtypeLike] = np.float32,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = NormalizeIntensityRange(a_min, a_max, subtrahend, divisor, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.t(d[key])
        return d
