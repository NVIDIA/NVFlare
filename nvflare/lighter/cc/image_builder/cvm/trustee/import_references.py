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

"""Import reviewed references into a stopped, single-profile Trustee instance."""

from pathlib import Path

from ..artifacts.bundle import verify_bundle
from ..common.io import read_json, write_json
from ..common.linux import lock
from ..common.references import EXPIRY_REFERENCE
from .references import check_profile, merge_records, profile_identity, reference_expirations


def import_references(bundle, store, state, expires):
    """Import reviewed values without changing a running Trustee instance."""
    bundle, store, state = Path(bundle), Path(store), Path(state)
    manifest = verify_bundle(bundle)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock(state / "publisher.lock"):
        identity = state / "security_profile.json"
        if identity.exists():
            check_profile(read_json(identity), manifest)
        values = read_json(bundle / "reference_values.json")
        store.mkdir(parents=True, exist_ok=True, mode=0o700)
        records = [read_json(path) for path in store.iterdir() if path.is_file() and path.name != EXPIRY_REFERENCE]
        merged = merge_records(records, values, expires)
        write_json(identity, profile_identity(manifest))
        write_json(
            store / EXPIRY_REFERENCE,
            {
                "version": "0.1.0",
                "name": EXPIRY_REFERENCE,
                "expiration": expires,
                "value": reference_expirations(merged),
            },
        )
        for record in merged:
            write_json(store / record["name"], record)
