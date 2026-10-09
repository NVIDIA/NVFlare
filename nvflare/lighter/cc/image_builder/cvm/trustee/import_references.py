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

"""Import reviewed references into one profile namespace in shared Trustee."""

import datetime
from pathlib import Path

from ..artifacts.bundle import verify_bundle
from ..common.errors import require
from ..common.io import read_json, write_json
from ..common.linux import lock
from .references import check_profile, merge_records, profile_identity, reference_expirations


def import_references(bundle, store, state, expires):
    """Import reviewed values without changing a running Trustee instance."""
    bundle, store, state = Path(bundle), Path(store), Path(state)
    manifest = verify_bundle(bundle)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock(state / "publisher.lock"):
        reference_id = manifest["reference_value_id"]
        identity = state / "profiles" / (reference_id + ".json")
        if identity.exists():
            check_profile(read_json(identity), manifest)
        values = read_json(bundle / "reference_values.json")
        store.mkdir(parents=True, exist_ok=True, mode=0o700)
        record_path = store / reference_id
        records = []
        if record_path.exists():
            current = read_json(record_path)
            require(
                current.get("name") == reference_id and isinstance(current.get("value"), dict), "Invalid RVPS record"
            )
            current_values = current["value"].get("values", {})
            current_expirations = current["value"].get("expirations", {})
            require(set(current_values) == set(current_expirations), "Incomplete RVPS profile record")
            records = [
                {
                    "version": "0.1.0",
                    "name": name,
                    "expiration": datetime.datetime.fromtimestamp(
                        current_expirations[name], datetime.timezone.utc
                    ).isoformat(),
                    "value": value,
                }
                for name, value in current_values.items()
            ]
        merged = merge_records(records, values, expires)
        write_json(identity, profile_identity(manifest))
        write_json(
            record_path,
            {
                "version": "0.1.0",
                "name": reference_id,
                "expiration": expires,
                "value": {
                    "values": {record["name"]: record["value"] for record in merged},
                    "expirations": reference_expirations(merged),
                },
            },
        )
