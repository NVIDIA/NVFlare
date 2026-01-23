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
from nvflare.security.logging import secure_log_traceback

from .app import ServerApp
from .constants import CollabMethodArgName, ContextKey
from .dec import supports_context


def run_server(server_app: ServerApp, logger):
    server_ctx = server_app.new_context(caller=server_app.name, callee=server_app.name)
    logger.info("initializing server app")
    server_app.initialize(server_ctx)

    if not server_app.mains:
        raise RuntimeError("server app does not have any algos!")

    result = None
    for name, f in server_app.mains:
        if server_ctx.is_aborted():
            break

        try:
            logger.info(f"Running algo {name}")
            kwargs = {CollabMethodArgName.CONTEXT: server_ctx}
            if not supports_context(f):
                kwargs = {}
            result = f(**kwargs)
            server_ctx.set_prop(ContextKey.RESULT, result)
        except Exception as ex:
            secure_log_traceback(logger)
            backend = server_app.backend
            if backend:
                backend.handle_exception(ex)
            break

    logger.info("finalizing server app")
    server_app.finalize(server_ctx)
    return result
