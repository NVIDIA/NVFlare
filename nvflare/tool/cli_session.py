# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""CLI-scoped session helpers."""

from nvflare.fuel.flare_api.flare_api import Session


def new_cli_session(
    username: str,
    startup_kit_location: str,
    timeout: float,
    study: str = "default",
    secure_mode: bool = True,
    debug: bool = False,
) -> Session:
    """Create a session for CLI commands and ensure cleanup on failed connect."""
    sess = Session(
        username=username,
        startup_path=startup_kit_location,
        secure_mode=secure_mode,
        debug=debug,
        study=study,
    )
    # CLI should fail fast: avoid long auto-login loops and time-box commands.
    try:
        if hasattr(sess, "api") and sess.api:
            if hasattr(sess.api, "auto_login_max_tries"):
                sess.api.auto_login_max_tries = 1
            if hasattr(sess.api, "set_command_timeout"):
                sess.api.set_command_timeout(timeout)
    except Exception:
        pass
    try:
        sess.try_connect(timeout)
    except Exception:
        try:
            sess.close()
        except Exception:
            pass
        raise
    return sess
