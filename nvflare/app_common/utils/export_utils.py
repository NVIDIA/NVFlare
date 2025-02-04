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
from nvflare.apis.fl_constant import ConnPropKey, FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.data_event.utils import get_scope_property


def update_export_props(props: dict, fl_ctx: FLContext):
    site_name = fl_ctx.get_identity_name()
    auth_token = get_scope_property(scope_name=site_name, key=FLMetaKey.AUTH_TOKEN, default="NA")
    signature = get_scope_property(scope_name=site_name, key=FLMetaKey.AUTH_TOKEN_SIGNATURE, default="NA")

    props[FLMetaKey.SITE_NAME] = site_name
    props[FLMetaKey.JOB_ID] = fl_ctx.get_job_id()
    props[FLMetaKey.AUTH_TOKEN] = auth_token
    props[FLMetaKey.AUTH_TOKEN_SIGNATURE] = signature

    root_conn_props = get_scope_property(site_name, ConnPropKey.ROOT_CONN_PROPS)
    if root_conn_props:
        props[ConnPropKey.ROOT_CONN_PROPS] = root_conn_props

    cp_conn_props = get_scope_property(site_name, ConnPropKey.CP_CONN_PROPS)
    if cp_conn_props:
        props[ConnPropKey.CP_CONN_PROPS] = cp_conn_props

    relay_conn_props = get_scope_property(site_name, ConnPropKey.RELAY_CONN_PROPS)
    if relay_conn_props:
        props[ConnPropKey.RELAY_CONN_PROPS] = relay_conn_props
