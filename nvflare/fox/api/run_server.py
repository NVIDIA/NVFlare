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
import traceback

from .app import ServerApp
from .constants import ContextKey


def run_server(server_app: ServerApp, logger):
    server_ctx = server_app.new_context(caller=server_app.name, callee=server_app.name)
    logger.info("initializing server app")
    server_app.initialize(server_ctx)

    if not server_app.strategies:
        raise RuntimeError("server app does not have any strategies!")

    result = None
    for idx, strategy in enumerate(server_app.strategies):
        if server_ctx.is_aborted():
            break

        try:
            logger.info(f"Running Strategy #{idx + 1} - {type(strategy).__name__}")
            server_app.current_strategy = strategy
            result = strategy.execute(context=server_ctx)
            server_ctx.set_prop(ContextKey.INPUT, result)
        except:
            traceback.print_exc()
            break
    return result
