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

import tenseal as ts

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentService


def load_tenseal_context_from_workspace(ctx_file_name: str, fl_ctx: FLContext):
    """Loads homomorphic encryption (HE) context from TenSEAL (https://github.com/OpenMined/TenSEAL) containing encryption keys and parameters.

    Args:
        ctx_file_name: filepath of TensSEAL context file
        fl_ctx: FL context

    Returns:
        TenSEAL context

    """
    is_secure_mode = fl_ctx.get_prop(FLContextKey.SECURE_MODE, True)
    data, rc = SecurityContentService.load_content(ctx_file_name)

    bad_rcs = [LoadResult.INVALID_CONTENT, LoadResult.NO_SUCH_CONTENT]
    if is_secure_mode:
        bad_rcs.extend([LoadResult.INVALID_SIGNATURE, LoadResult.NOT_SIGNED])

    if rc in bad_rcs:
        raise ValueError("Cannot load tenseal_context {}: {}".format(ctx_file_name, rc))

    context = ts.context_from(data)
    return context


def count_encrypted_layers(encrypted_layers: dict):
    """Count number of encrypted layers homomorphic encryption (HE) layers/variables."""
    n_total = len(encrypted_layers)
    n_encrypted = 0
    for e in encrypted_layers.keys():
        if encrypted_layers[e]:
            n_encrypted += 1
    return n_encrypted, n_total


def serialize_nested_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            serialize_nested_dict(v)
        else:
            if isinstance(v, ts.CKKSVector):
                d[k] = v.serialize()
    return d


def deserialize_nested_dict(d, context):
    for k, v in d.items():
        if isinstance(v, dict):
            deserialize_nested_dict(v, context)
        else:
            if isinstance(v, bytes):
                d[k] = ts.ckks_vector_from(context, v)
    return d
