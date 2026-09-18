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

import argparse
from pathlib import Path

from ..artifacts.bundle import verify_bundle
from ..common.io import read_json, write_json
from ..common.linux import lock
from ..common.references import EXPIRY_REFERENCE
from .references import check_profile, merge_records, profile_identity, reference_expirations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--store", type=Path, required=True, help="Trustee local_fs reference_value namespace directory"
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expires", required=True, help="Administrator-approved UTC expiry, e.g. 2026-12-01T00:00:00Z")
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle)
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock(args.state / "publisher.lock"):
        identity = args.state / "security_profile.json"
        if identity.exists():
            check_profile(read_json(identity), manifest)
        values = read_json(args.bundle / "reference_values.json")
        args.store.mkdir(parents=True, exist_ok=True, mode=0o700)
        records = [read_json(path) for path in args.store.iterdir() if path.is_file() and path.name != EXPIRY_REFERENCE]
        merged = merge_records(records, values, args.expires)
        write_json(identity, profile_identity(manifest))
        write_json(
            args.store / EXPIRY_REFERENCE,
            {
                "version": "0.1.0",
                "name": EXPIRY_REFERENCE,
                "expiration": args.expires,
                "value": reference_expirations(merged),
            },
        )
        for record in merged:
            write_json(args.store / record["name"], record)


if __name__ == "__main__":
    main()
