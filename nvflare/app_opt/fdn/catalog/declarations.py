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

"""Bounded, optional site-authored declarations, kept separate from reader configuration."""

import json

from nvflare.app_common.fdn.catalog.site.connector import ConnectorError

SIDECAR_NAME = "catalog.json"
MAX_SIDECAR_BYTES = 1024 * 1024
DECLARATION_KEYS = frozenset(
    {
        "display_name",
        "description",
        "owner",
        "modalities",
        "operations",
        "semantic_profile",
        "source_version",
        "lineage",
        "provenance",
        "observed_at",
        "tables",
    }
)


def read_declarations(path):
    if not path.is_file():
        return {}
    try:
        if path.stat().st_size > MAX_SIDECAR_BYTES:
            raise ConnectorError("sidecar is too large")
        with path.open("rb") as stream:
            raw = stream.read(MAX_SIDECAR_BYTES + 1)
        if len(raw) > MAX_SIDECAR_BYTES:
            raise ConnectorError("sidecar is too large")
        data = json.loads(raw)
    except ConnectorError:
        raise
    except Exception:
        raise ConnectorError("sidecar is unreadable") from None
    if not isinstance(data, dict) or set(data) - DECLARATION_KEYS:
        raise ConnectorError("sidecar has unsupported content")
    return data
