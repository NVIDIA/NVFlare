# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pickle

from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.protos.federated_pb2 import ModelData
from nvflare.private.fed.utils.numproto import bytes_to_proto


def shareable_to_modeldata(shareable, fl_ctx):
    # make_init_proto message
    model_data = ModelData()  # create an empty message

    model_data.params["data"].CopyFrom(make_shareeable_data(shareable))

    context_data = make_context_data(fl_ctx)
    model_data.params["fl_context"].CopyFrom(context_data)
    return model_data


def make_shareeable_data(shareable):
    return bytes_to_proto(shareable.to_bytes())


def make_context_data(fl_ctx):
    shared_fl_ctx = FLContext()
    shared_fl_ctx.set_public_props(fl_ctx.get_all_public_props())
    props = pickle.dumps(shared_fl_ctx)
    context_data = bytes_to_proto(props)
    return context_data
