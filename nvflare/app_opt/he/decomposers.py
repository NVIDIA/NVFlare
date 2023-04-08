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
"""Decomposers for HE related classes"""
from typing import Any

import tenseal as ts

from nvflare.fuel.utils import fobs


class CKKSVectorDecomposer(fobs.Decomposer):
    def supported_type(self):
        return ts.CKKSVector

    def decompose(self, target: ts.CKKSVector) -> Any:
        return target.serialize(), target.context().serialize()

    def recompose(self, data: Any) -> ts.CKKSVector:
        vec_data, ctx_data = data
        context = ts.context_from(ctx_data)
        return ts.ckks_vector_from(context, vec_data)


def register():
    if register.registered:
        return

    fobs.register(CKKSVectorDecomposer)

    register.registered = True


register.registered = False
