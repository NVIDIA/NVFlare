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

import json
import logging
import os

from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.config_service import ConfigService

LOG_STREAM_EVENT_TYPE = "stream_log"
LIVE_LOG_TOPIC = "live_log"

# Resources.json var name. When False (default), live log streaming is disabled
# at this site: JobLogStreamer no-ops, SystemLogStreamer skips injection and
# removes any pre-declared JobLogStreamer, and the server-side JobLogReceiver
# logs an error if a stream still arrives from this site.
ALLOW_LOG_STREAMING_VAR = "allow_log_streaming"

_logger = logging.getLogger(__name__)


def is_log_streaming_allowed(fl_ctx: FLContext = None) -> bool:
    """Check whether the site permits live log streaming.

    Tries ConfigService first (fast path, populated in production via
    FLClientStarterConfiger); falls back to reading resources.json directly
    via the workspace from fl_ctx, which is required in the simulator path
    where the section is not registered with ConfigService.
    """
    val = ConfigService.get_bool_var(name=ALLOW_LOG_STREAMING_VAR, conf=SystemConfigs.RESOURCES_CONF, default=None)
    if val is not None:
        return bool(val)

    if fl_ctx is None:
        return False

    workspace_root = fl_ctx.get_prop(FLContextKey.WORKSPACE_ROOT)
    site_name = fl_ctx.get_identity_name(default="")
    if not workspace_root or not site_name:
        return False

    try:
        ws = Workspace(root_dir=workspace_root, site_name=site_name)
        path = ws.get_resources_file_path()
        if not path or not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
    except Exception as ex:
        _logger.debug(f"failed to read resources.json for {ALLOW_LOG_STREAMING_VAR} check: {ex}")
        return False

    return bool(data.get(ALLOW_LOG_STREAMING_VAR, False))


class Channels(object):
    LOG_STREAMING_CHANNEL = "log_streaming"
    ERROR_LOG_LOG_TYPE = "ERRORLOG"
