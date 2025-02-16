# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.private.fed.utils.identity_utils import get_parent_site_name


def determine_parent_name(client_config: dict):
    fqsn = client_config.get("fqsn")
    return get_parent_site_name(fqsn)


def determine_parent_fqcn(client_config: dict, fl_ctx: FLContext):
    parent_client_name = determine_parent_name(client_config)
    if parent_client_name:
        engine = fl_ctx.get_engine()
        parent_client = engine.get_client_from_name(parent_client_name)
        if not parent_client:
            raise RuntimeError(f"cannot find parent '{parent_client_name}'")

        if not isinstance(parent_client, Client):
            raise RuntimeError(f"expect parent_client to be Client but got {type(parent_client)}")

        return parent_client.get_fqcn()
    else:
        return FQCN.ROOT_SERVER
