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

"""Harden only the reviewed, pinned Trustee Compose setup script, idempotently."""

import argparse
import hashlib
import os
import stat
import tempfile
from pathlib import Path

# confidential-containers/trustee@338610fbfed57b66c61a8a3a60e0e4386bdce793,
# kbs/config/docker-compose/setup.sh. Do not silently patch a changed upstream.
UPSTREAM_SHA256 = "c299639734bf68c7532a43aa403f77eacd13af976fd94074cab0b58c4e3a5768"
REPLACEMENTS = (
    ("set -euxo pipefail", "set +x\nset -euo pipefail\numask 077"),
    (
        'cd "${KEY_DIR}"',
        'cd "${KEY_DIR}"\n'
        'chmod 0700 "${KEY_DIR}"\n'
        "for secret in private.key admin-token ca.key token.key; do\n"
        '  if [ -e "${secret}" ]; then chmod 0600 "${secret}"; fi\n'
        "done",
    ),
    ("chmod 644 admin-token", "chmod 0600 admin-token"),
)


def harden(path: Path) -> bool:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("setup.sh must be a regular, non-linked file")
    source = path.read_bytes()
    text = source.decode("utf-8")
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA256:
        original = text
        for old, new in REPLACEMENTS:
            if original.count(new) != 1:
                raise ValueError("unexpected Trustee setup.sh; refusing to patch unreviewed source")
            original = original.replace(new, old, 1)
        if hashlib.sha256(original.encode("utf-8")).hexdigest() != UPSTREAM_SHA256:
            raise ValueError("unexpected Trustee setup.sh; refusing to patch unreviewed source")
        return False

    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError("unexpected Trustee setup.sh patch location")
        text = text.replace(old, new, 1)
    # A private same-directory temporary file prevents partial-script execution.
    descriptor, temporary = tempfile.mkstemp(prefix=".setup-hardened-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setup", type=Path, help="Pinned Trustee kbs/config/docker-compose/setup.sh")
    args = parser.parse_args()
    try:
        changed = harden(args.setup)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"Trustee setup hardening failed: {error}\n")
    print("Trustee setup hardened." if changed else "Trustee setup already hardened.")


if __name__ == "__main__":
    main()
