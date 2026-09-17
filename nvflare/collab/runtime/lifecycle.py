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
from nvflare.collab.api.app import ServerApp
from nvflare.collab.api.constants import CollabMethodArgName, ContextKey
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.decorators import supports_context
from nvflare.collab.api.exceptions import RunAborted
from nvflare.security.logging import secure_log_traceback


def run_server(server_app: ServerApp, logger):
    previous_ctx = get_call_context()
    server_ctx = server_app.new_context(caller=server_app.name, callee=server_app.name)
    result = None
    try:
        logger.info("initializing server app")
        server_app.initialize(server_ctx)

        if len(server_app.mains) != 1:
            raise RuntimeError(f"server app must have exactly one main function but got {len(server_app.mains)}")

        name, main_func = server_app.mains[0]
        if not server_ctx.is_aborted():
            try:
                logger.info(f"Running main {name}")
                kwargs = {CollabMethodArgName.CONTEXT: server_ctx}
                if not supports_context(main_func):
                    kwargs = {}
                result = main_func(**kwargs)
                server_ctx.set_prop(ContextKey.RESULT, result)
            except RunAborted:
                logger.info("server app run aborted")
            except Exception as ex:
                secure_log_traceback(logger)
                backend = server_app.backend
                if backend:
                    backend.handle_exception(ex)
    finally:
        try:
            logger.info("finalizing server app")
            server_app.finalize(server_ctx)
        finally:
            set_call_context(previous_ctx)

    return result
