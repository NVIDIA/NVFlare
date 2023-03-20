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

import tenseal as ts

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import Learnable
from nvflare.app_common.abstract.persistor_filter import PersistorFilter
from nvflare.app_opt.he.homomorphic_encrypt import load_tenseal_context_from_workspace


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


class HEModelSerializeFilter(PersistorFilter):
    def __init__(self, tenseal_context_file="server_context.tenseal"):
        """Used to serialize TenSEAL encrypted server models for use with
        homomorphic encryption (HE) support using TenSEAL https://github.com/OpenMined/TenSEAL.

        Args:
            tenseal_context_file: tenseal context files containing TenSEAL context
        """
        super().__init__()
        self.tenseal_context = None
        self.tenseal_context_file = tenseal_context_file

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.tenseal_context = None

    def process_post_load(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object after it was loaded.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return deserialize_nested_dict(learnable, self.tenseal_context)

    def process_pre_save(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object to support persisting when containing encrypted objects.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return serialize_nested_dict(learnable)

    def process_post_save(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object to support persisting when containing encrypted objects.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return deserialize_nested_dict(learnable, self.tenseal_context)

    def process_post_get(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object after it was returned.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return deserialize_nested_dict(learnable, self.tenseal_context)
