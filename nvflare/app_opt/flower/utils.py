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
import os.path

import grpc

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace

from .defs import Constant


def get_applet_log_file_path(base_name: str, run_ctx: dict):
    """Determine the full path of the log file to be used by an applet process.

    Args:
        base_name: the base name of the log file
        run_ctx: the run context of the applet

    Returns: full path of the log file for the applet.

    The log file will be placed in the job's run dir.

    """
    fl_ctx = run_ctx.get(Constant.APP_CTX_FL_CONTEXT)
    if not isinstance(fl_ctx, FLContext):
        raise RuntimeError(f"{Constant.APP_CTX_FL_CONTEXT} should be FLContext but got {type(fl_ctx)}")

    ws = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    if not isinstance(ws, Workspace):
        raise RuntimeError(f"{FLContextKey.WORKSPACE_OBJECT} should be Workspace but got {type(ws)}")

    run_dir = ws.get_run_dir(fl_ctx.get_job_id())
    return os.path.join(run_dir, base_name)


def create_channel(server_addr, grpc_options, ready_timeout: float, test_only: bool):
    """Create gRPC channel and waits for the server to be ready

    Args:
        server_addr: the server address to connect to
        grpc_options: gRPC client connection options
        ready_timeout: how long to wait for the server to be ready
        test_only: whether for testing the server readiness only

    Returns: the gRPC channel created. Bit if test_only, the channel is closed and returns None.

    If the server does not become ready within ready_timeout, the RuntimeError exception will raise.

    """
    channel = grpc.insecure_channel(server_addr, options=grpc_options)

    # wait for channel ready
    try:
        grpc.channel_ready_future(channel).result(timeout=ready_timeout)
    except grpc.FutureTimeoutError:
        raise RuntimeError(f"cannot connect to server after {ready_timeout} seconds")

    if test_only:
        channel.close()
        channel = None
    return channel
