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

"""Helpers for projecting site_config metadata from a client's merged config.

The client's runtime config is the union of fed_client.json and resources.json.
project_site_config drops the structural / local-only top-level keys so the
remaining custom keys (labels, capabilities, custom resource hints, ...) can
be advertised to the server during registration.
"""

import copy

# Top-level keys in the merged fed_client.json + resources.json that should not
# be forwarded to the server as site metadata. These are either plumbed
# elsewhere, describe local component wiring, or hold paths/identities that
# don't transfer across machines.
SITE_CONFIG_EXCLUDED_TOP_LEVEL_KEYS = frozenset(
    {
        "format_version",  # config schema version, not site metadata
        "client",  # already forwarded as client_config; including it here would be circular
        "servers",  # connection info
        "components",  # local component wiring (class paths, args)
        "handlers",  # local handler wiring
        "snapshot_persistor",  # server-side persistence backend
        "admin",  # admin client config
        "relay_config",  # local connection topology
        "overseer_agent",  # local HA overseer wiring
    }
)


def project_site_config(config_data: dict) -> dict:
    """Project site_config from the merged client config.

    Drops keys in SITE_CONFIG_EXCLUDED_TOP_LEVEL_KEYS and deep-copies the rest
    so the result is independent of the live config. May be empty.
    """
    if not isinstance(config_data, dict):
        return {}
    return {k: copy.deepcopy(v) for k, v in config_data.items() if k not in SITE_CONFIG_EXCLUDED_TOP_LEVEL_KEYS}
