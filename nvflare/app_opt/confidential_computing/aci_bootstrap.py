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

"""Launch the single ACI server from an externally delivered, signed startup kit.

The image and its enforcement policy must fix this module, interpreter, public
key, input mount and launch command. Invoke with Python's ``-I`` option.
"""

import io
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from cryptography import x509
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.x509.oid import NameOID

from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.utils import verify_content, verify_folder_signature

INPUT_DIR = Path("/run/nvflare-input")
PUBLIC_KEY_FILE = Path("/opt/nvflare-trust/deployment-public-key.pem")
WORKSPACE_DIR = Path("/workspace")
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_EXTRACTED_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096


def _read_bounded(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError("Missing or oversized bootstrap input")
    return data


def _validated_members(archive):
    members, names, total = [], {}, 0
    for member in archive:
        if len(members) >= MAX_ARCHIVE_MEMBERS:
            raise ValueError("Too many startup archive entries")
        name = member.name.rstrip("/") if member.isdir() else member.name
        parts = name.split("/")
        if (
            not name
            or "\\" in name
            or any(part in ("", ".", "..") for part in parts)
            or PurePosixPath(name).is_absolute()
            or name in names
            or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE)
            or member.pax_headers
            or member.sparse is not None
        ):
            raise ValueError("Unsafe startup archive entry")
        names[name] = member.isdir()
        if member.isfile():
            total += member.size
            if member.size < 0 or total > MAX_EXTRACTED_BYTES:
                raise ValueError("Startup archive contents are oversized")
        members.append((member, name))
    if not members:
        raise ValueError("Empty startup archive")
    for name in names:
        if any(names.get(str(parent)) is False for parent in PurePosixPath(name).parents):
            raise ValueError("Conflicting startup archive entries")
    return members


def _extract_archive(data, destination):
    # Only uncompressed archives are accepted. Validate the full member list
    # before writing any contents; never call tarfile.extract or extractall.
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = _validated_members(archive)
        for member, name in members:
            target = destination.joinpath(*PurePosixPath(name).parts)
            parent = destination
            for part in PurePosixPath(name).parts[:-1]:
                parent = parent / part
                parent.mkdir(mode=0o700, exist_ok=True)
            if member.isdir():
                target.mkdir(mode=0o700, exist_ok=True)
                continue
            with archive.extractfile(member) as source, target.open("xb") as output:
                os.chmod(target, 0o600)
                shutil.copyfileobj(source, output)


def prepare_workspace():
    """Authenticate the delivered bytes, then extract and verify the kit."""
    data = _read_bounded(INPUT_DIR / "startup-kit.tar", MAX_ARCHIVE_BYTES)
    signature = _read_bounded(INPUT_DIR / "startup-kit.sig", 4096)
    public_key = load_pem_public_key(_read_bounded(PUBLIC_KEY_FILE, 64 * 1024))
    # The trust key is part of the measured image, never supplied by the kit.
    verify_content(data, signature, public_key)
    destination = Path(tempfile.mkdtemp(prefix="nvflare-server-", dir=WORKSPACE_DIR))
    try:
        _extract_archive(data, destination)
        for required in ("startup", "local"):
            if not (destination / required).is_dir():
                raise ValueError("Incomplete server startup kit")
        if not (destination / ProvFileName.SIGNATURE_JSON).is_file() or not verify_folder_signature(
            str(destination),
            str(destination / "startup/rootCA.pem"),
            single_signer=True,
            signature_file=ProvFileName.SIGNATURE_JSON,
        ):
            raise ValueError("Invalid server startup kit signatures")
        return destination
    except Exception:
        shutil.rmtree(destination)
        raise


def _server_command(workspace):
    certificate = x509.load_pem_x509_certificate((workspace / "startup/server.crt").read_bytes())
    organizations = certificate.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
    if len(organizations) != 1 or not organizations[0].value:
        raise ValueError("Server certificate must identify its organization")
    return [
        sys.executable,
        "-I",
        "-u",
        "-m",
        "nvflare.private.fed.app.server.server_train",
        "-m",
        str(workspace),
        "-s",
        "fed_server.json",
        "--set",
        "secure_train=true",
        "config_folder=config",
        "org=" + organizations[0].value,
    ]


def main():
    """Use fixed image-owned inputs and replace this process with the server."""
    destination = None
    try:
        if not sys.flags.isolated or len(sys.argv) != 1:
            raise ValueError("Bootstrap requires isolated Python and no arguments")
        destination = prepare_workspace()
        command = _server_command(destination)
        # Neither archive content nor inherited interpreter/loader variables
        # may select code. The launch policy must likewise fix the initial env.
        environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(destination), "LANG": "C.UTF-8"}
        os.chdir(destination)
        os.execve(sys.executable, command, environment)
    except Exception:
        if destination is not None:
            shutil.rmtree(destination)
        # Do not print archive contents, credentials, or sensitive input paths.
        print("ACI server bootstrap failed; server was not started", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
