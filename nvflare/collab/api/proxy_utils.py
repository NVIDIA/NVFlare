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

"""Shared utilities for creating Proxy objects.

This module provides common functionality for proxy preparation that is shared
between AppRunner (simulation path) and CollabExecutor (FLARE path).
"""

from typing import Dict, Optional

from nvflare.collab.api.backend import Backend
from nvflare.collab.api.proxy import Proxy


def create_proxy_with_children(
    app,
    target_name: str,
    target_fqn: str,
    main_backend: Backend,
    main_interface: dict,
    child_specs: Dict[str, dict],
) -> Proxy:
    """Create a Proxy with child proxies for collab objects.

    This is the shared logic for creating proxies, used by both simulation
    and FLARE execution paths.

    Args:
        app: The App that will use this proxy.
        target_name: Name of the target (e.g., "server", "site-1").
        target_fqn: Fully qualified name for CellNet routing.
        main_backend: The backend to use for main target calls.
        main_interface: Interface dict for the main target object.
        child_specs: Dict mapping child names to {"interface": dict, "backend": Backend}.
                    If "backend" is not provided, main_backend is used.

    Returns:
        Proxy with child proxies added.
    """
    # Create main proxy
    proxy = Proxy(
        app=app,
        target_name=target_name,
        target_fqn=target_fqn,
        backend=main_backend,
        target_interface=main_interface,
    )

    # Create child proxies for collab objects
    for name, spec in child_specs.items():
        # Child FQN: if parent has FQN, extend it; otherwise empty (in-process mode)
        child_fqn = f"{target_fqn}.{name}" if target_fqn else ""
        child_backend = spec.get("backend", main_backend)
        child_interface = spec.get("interface", {})
        child_proxy = Proxy(
            app=app,
            target_name=f"{target_name}.{name}",
            target_fqn=child_fqn,
            backend=child_backend,
            target_interface=child_interface,
        )
        proxy.add_child(name, child_proxy)

    return proxy


def get_worker_fqcn(site_name: str, worker_id: Optional[str] = None) -> str:
    """Get the CellNet FQCN for a subprocess worker.

    Args:
        site_name: Name of the site (e.g., "site-1").
        worker_id: Worker ID (defaults to site_name if not provided).

    Returns:
        Worker FQCN string (e.g., "site-1.worker.site-1").
    """
    if worker_id is None:
        worker_id = site_name
    return f"{site_name}.worker.{worker_id}"
