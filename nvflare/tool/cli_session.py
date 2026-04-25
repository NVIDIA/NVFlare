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

from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.tool.kit.kit_config import resolve_admin_user_and_dir_from_startup_kit, resolve_startup_kit_dir


def new_cli_session(
    username: str,
    startup_kit_location: str,
    timeout: float,
    study: str = "default",
    secure_mode: bool = True,
    debug: bool = False,
) -> Session:
    """Compatibility wrapper for CLI callers around the shared secure session factory."""
    return new_secure_session(
        username=username,
        startup_kit_location=startup_kit_location,
        debug=debug,
        study=study,
        timeout=timeout,
        auto_login_max_tries=1,
    )


def new_active_cli_session(timeout: float, study: str = "default", debug: bool = False) -> Session:
    """Create a CLI session using NVFLARE_STARTUP_KIT_DIR or the active registered startup kit."""
    startup_dir = resolve_startup_kit_dir()
    username, admin_user_dir = resolve_admin_user_and_dir_from_startup_kit(startup_dir)
    return new_cli_session(
        username=username,
        startup_kit_location=admin_user_dir,
        timeout=timeout,
        study=study,
        debug=debug,
    )
