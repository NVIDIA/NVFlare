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

"""Seal application vaults and assemble standalone deliveries."""

import contextlib
import errno
import hashlib
import json
import os
import posixpath
import re
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path

from ..artifacts.bundle import load_profile_set, verify_approval, verify_bundle
from ..artifacts.oci import CVM_ARTIFACT_TYPE, materialize
from ..artifacts.packaging import package_deliveries
from ..common.contracts import HEADER_BYTES, STORAGE_PROFILE, binding, resource_path
from ..common.errors import BuildError, ConfigurationError, require, require_config
from ..common.io import is_sha256, write_json
from ..common.linux import memory_file, protect_process, run
from ..common.luks import scan, snapshot_header
from ..common.validation import runtime_config
from ..trustee import client as trustee_client
from . import config
from .payload import HOST_MODULES, copy_modules
from .storage import (
    content_digest,
    copy_tree,
    create_image,
    format_vault,
    mounted,
    nbd,
    opened_vault,
    set_tree_owner,
    sidecar,
)

GIB = 1024**3
FILESYSTEM_BLOCK_BYTES = 4096
MINIMUM_VAULT_HEADROOM_BYTES = 2 * GIB
VAULT_USABLE_PERCENT = 90


def populate(root, app):
    for directory in ("docker", "docker/data", "containerd", "config", "application", "services", "scripts"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(app["docker_archive"], root / "docker/application.tar")
    os.chmod(root / "docker/application.tar", 0o600)
    write_json(root / "config/application.json", runtime_config(app))
    if app.get("application_files"):
        copy_tree(app["application_files"], root / "application")
    for name in ("runtime", "data"):
        path = root / "application" / name
        require(not path.is_symlink(), "Writable application directory cannot be a symlink")
        path.mkdir(exist_ok=True)
    uid, gid = app["_application_owner"]
    set_tree_owner(root / "application", uid, gid)
    for service in app["services"]:
        shutil.copyfile(service, root / "services" / Path(service).name)


def _numeric_owner(user):
    match = re.fullmatch(r"([0-9]+)(?::([0-9]+))?", user) if user else None
    if match:
        uid = int(match.group(1))
        gid = int(match.group(2) or match.group(1))
    else:
        require(user in ("", "root", "root:root"), "Docker image USER must be numeric UID[:GID] or root")
        uid = gid = 0
    require(uid < 2**32 - 1 and gid < 2**32 - 1, "Docker image USER is outside the UID/GID range")
    return uid, gid


def _validated_image(path, image_id):
    try:
        normalized, image, layers = _archive_image(path, image_id)
        container = image.get("config") or {}
        require(isinstance(container, dict), "Invalid Docker image configuration")
        user = container.get("User", "")
        require(isinstance(user, str), "Invalid Docker image USER")
        return normalized, _numeric_owner(user), layers
    except (BuildError, OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, tarfile.TarError):
        # Archives can embed private paths, image metadata or credentials. Do not
        # expose parser exceptions, even though this check precedes key creation.
        raise ConfigurationError("Invalid docker_archive; regenerate it with docker save and verify image_id") from None


def validated_image(path, image_id):
    normalized, owner, _ = _validated_image(path, image_id)
    return normalized, owner


def validate_archive(path, image_id):
    return validated_image(path, image_id)[0]


def _entry_layers(entry):
    layers = entry["Layers"]
    require(
        isinstance(layers, list)
        and all(
            isinstance(name, str) and name and not name.startswith("/") and ".." not in Path(name).parts
            for name in layers
        ),
        "Invalid Docker image layer list",
    )
    return tuple(layers)


def _resolved_archive_file(archive, name):
    """Resolve a regular member or a safe in-archive link without host extraction."""
    current = name
    seen = set()
    for _ in range(32):
        require(
            isinstance(current, str) and current and not posixpath.isabs(current) and ".." not in Path(current).parts,
            "Invalid Docker archive member",
        )
        require(current not in seen, "Docker archive link cycle")
        seen.add(current)
        member = archive.getmember(current)
        if member.isfile():
            return member
        require(member.issym() or member.islnk(), "Invalid Docker image layer")
        target = member.linkname
        require(isinstance(target, str) and target, "Invalid Docker archive link")
        if member.issym():
            target = posixpath.join(posixpath.dirname(current), target)
        current = posixpath.normpath(target)
    raise BuildError("Docker archive link nesting exceeds the supported limit")


def _archive_image(path, image_id):
    # Only inspect manifest/config members, never unpack layers on the build host.

    with tarfile.open(path, "r:*") as archive:
        member = archive.getmember("manifest.json")
        require(member.isfile() and member.size < 1024**2, "Invalid Docker save manifest")
        entries = json.load(archive.extractfile(member))
        require(isinstance(entries, list) and entries, "Empty Docker save archive")
        configurations = {}
        archive_layers = []
        archive_chains = set()
        for entry in entries:
            require(isinstance(entry, dict), "Invalid Docker save entry")
            filename = entry["Config"]
            require(
                isinstance(filename, str) and not filename.startswith("/") and ".." not in Path(filename).parts,
                "Invalid Docker config member",
            )
            member = archive.getmember(filename)
            require(member.isfile() and member.size < 16 * 1024**2, "Invalid Docker image configuration")
            data = archive.extractfile(member).read()
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            image = json.loads(data)
            layers = _entry_layers(entry)
            chain = []
            for layer in layers:
                _resolved_archive_file(archive, layer)
                chain.append(layer)
                identity = tuple(chain)
                if identity not in archive_chains:
                    archive_layers.append(layer)
                    archive_chains.add(identity)
            if digest in configurations:
                require(configurations[digest][1] == layers, "Ambiguous Docker image layers")
            configurations[digest] = (image, layers)
        if image_id in configurations:
            image, _ = configurations[image_id]
            return image_id, image, tuple(archive_layers)
        # Docker's containerd store reports an OCI manifest/index digest as Id;
        # the guest's classic store reports the image configuration digest.
        # Authenticate that graph before normalizing to the guest's identity.
        current = image_id
        for _ in range(8):
            algorithm, digest = current.split(":", 1)
            require(
                algorithm == "sha256" and is_sha256(digest),
                "Invalid OCI descriptor digest",
            )
            member = archive.getmember("blobs/sha256/" + digest)
            require(member.isfile() and member.size < 16 * 1024**2, "Invalid OCI manifest")
            data = archive.extractfile(member).read()
            require(hashlib.sha256(data).hexdigest() == digest, "OCI manifest digest mismatch")
            value = json.loads(data)
            if "manifests" in value:
                matches = [
                    item
                    for item in value["manifests"]
                    if item.get("platform", {}).get("os") == "linux"
                    and item.get("platform", {}).get("architecture") == "amd64"
                ]
                require(len(matches) == 1, "OCI index requires exactly one Linux amd64 image")
                current = matches[0]["digest"]
            else:
                image = value["config"]["digest"]
                require(image in configurations, "OCI image is absent from Docker save manifest")
                configuration, _ = configurations[image]
                return image, configuration, tuple(archive_layers)
        raise BuildError("OCI manifest nesting exceeds the supported limit")


def _allocated_file_bytes(size):
    return ((size + FILESYSTEM_BLOCK_BYTES - 1) // FILESYSTEM_BLOCK_BYTES) * FILESYSTEM_BLOCK_BYTES


def _stored_file_bytes(path, status):
    """Estimate target allocation from logical data extents, preserving sparse holes."""
    size = status.st_size
    if not size:
        return 0
    seek_data = getattr(os, "SEEK_DATA", None)
    seek_hole = getattr(os, "SEEK_HOLE", None)
    if seek_data is None or seek_hole is None:
        return _allocated_file_bytes(size)
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (status.st_dev, status.st_ino, size):
            return _allocated_file_bytes(max(size, opened.st_size))
        total = 0
        offset = 0
        while offset < size:
            try:
                data = os.lseek(descriptor, offset, seek_data)
            except OSError as error:
                if error.errno == errno.ENXIO:
                    return total
                return _allocated_file_bytes(size)
            hole = os.lseek(descriptor, data, seek_hole)
            if data < offset or data >= size or hole <= data:
                return _allocated_file_bytes(size)
            total += _allocated_file_bytes(min(hole, size) - data)
            offset = hole
        return total
    except OSError:
        # Filesystems may not implement SEEK_DATA/SEEK_HOLE. Counting the full
        # logical size is conservative for the uncompressed ext4 vault.
        return _allocated_file_bytes(size)
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _application_file_bytes(path):
    if not path:
        return 0
    identities = set()
    total = 0
    for item in (Path(path), *Path(path).rglob("*")):
        status = item.lstat()
        identity = (status.st_dev, status.st_ino)
        if item.is_file() and not item.is_symlink():
            if identity in identities:
                continue
            identities.add(identity)
            # copy_tree uses tar --sparse, so preserve logical holes while
            # ignoring compression savings from the source filesystem.
            total += _stored_file_bytes(item, status)
        else:
            total += FILESYSTEM_BLOCK_BYTES
    return total


def _estimated_layer_bytes(path, layers, capacity):
    """Estimate overlay2 allocation without extracting image content on the host."""
    total = 0
    complete = True
    with tarfile.open(path, "r:*") as archive:
        for name in layers:
            member = _resolved_archive_file(archive, name)
            try:
                with archive.extractfile(member) as stream, tarfile.open(fileobj=stream, mode="r|*") as layer:
                    for item in layer:
                        total += _allocated_file_bytes(item.size) if item.isfile() else FILESYSTEM_BLOCK_BYTES
                        if total > capacity:
                            return total, complete
            except tarfile.ReadError:
                # Older Python versions cannot inspect every compression format
                # Docker can import. The stored layer size remains a safe lower
                # bound, and the caller reports that the estimate is incomplete.
                total += member.size
                complete = False
    return total, complete


def _estimated_archive_extraction_bytes(path, capacity):
    """Estimate Docker's temporary extraction of the complete save archive."""
    total = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            total += _allocated_file_bytes(member.size) if member.isfile() else FILESYSTEM_BLOCK_BYTES
            if total > capacity:
                break
    return total


def validate_vault_capacity(app, layers):
    """Reject a vault that cannot hold its archive and estimated imported image."""
    raw_capacity = app["vault_drive_size"] * GIB
    usable_capacity = max(0, (raw_capacity - HEADER_BYTES) * VAULT_USABLE_PERCENT // 100)
    try:
        archive_bytes = Path(app["docker_archive"]).stat().st_size
        application_bytes = _application_file_bytes(app.get("application_files"))
        extracted_archive_bytes = _estimated_archive_extraction_bytes(app["docker_archive"], usable_capacity)
        layer_bytes, complete = _estimated_layer_bytes(app["docker_archive"], layers, usable_capacity)
    except (OSError, ValueError, EOFError, tarfile.TarError):
        raise ConfigurationError("Cannot estimate vault storage from docker_archive and application_files") from None
    content_bytes = archive_bytes + extracted_archive_bytes + application_bytes + layer_bytes
    safety_bytes = max(MINIMUM_VAULT_HEADROOM_BYTES, (content_bytes + 4) // 5)
    required_bytes = content_bytes + safety_bytes
    required_raw_bytes = HEADER_BYTES + (required_bytes * 100 + VAULT_USABLE_PERCENT - 1) // VAULT_USABLE_PERCENT
    required_gib = (required_raw_bytes + GIB - 1) // GIB
    require_config(
        usable_capacity >= required_bytes,
        f"vault_drive_size is {app['vault_drive_size']} GiB but the retained archive, temporary extraction, and "
        f"imported image require "
        f"at least {required_gib} GiB; increase it and add space for expected application runtime writes",
    )
    if not complete:
        print(
            "WARNING: Some Docker layers use an unsupported compression format; vault sizing used their stored size. "
            "Add extra capacity for their expanded size and application runtime writes.",
            flush=True,
        )


def delivery(directory, app, manifest, digest, internal, bundle, trusted_keys=()):
    # A delivery is self-contained for one platform/application. These bytes are
    # copied from the already-built generic bundle; Stage 2 never rebuilds them.
    bundle = Path(bundle)
    delivered_bundle = directory / "cvm_bundle"
    delivered_bundle.mkdir()
    for name in (*manifest["sha256"], "cvm_manifest.json", "resource_policy.rego"):
        require(
            Path(name).name == name and not (delivered_bundle / name).exists(),
            "Invalid delivery bundle member",
        )
        shutil.copy2(bundle / name, delivered_bundle / name)
    if (bundle / "approval.json").is_file():
        shutil.copy2(bundle / "approval.json", delivered_bundle / "approval.json")
    copied_manifest = verify_bundle(delivered_bundle)
    require(copied_manifest["build_id"] == manifest["build_id"], "Copied CVM bundle identity mismatch")
    if (delivered_bundle / "approval.json").is_file():
        verify_approval(delivered_bundle, trusted_keys)
    public = dict(
        internal,
        vault_bind=digest.hex(),
        resource=resource_path(manifest["build_id"], manifest["platform"], digest),
        allowed_ports=app["allowed_ports"],
    )
    # A plaintext content digest leaks equality and permits known-content probes.
    public.pop("content_sha256", None)
    if manifest.get("dev_mode"):
        del public["vault_bind"]
        del public["resource"]
    public["attestation_policy_id"] = manifest["attestation_policy_id"]
    write_json(directory / "vault_manifest.json", public, mode=0o644)
    # Only launcher/verification modules; no build or Trustee administration code.
    copy_modules(config.SOURCE, directory, HOST_MODULES)
    for script in ("launch_cvm.sh", "shutdown_cvm.sh"):
        template = (bundle / (script + ".tmpl")).read_text()
        for name, value in {
            "platform": manifest["platform"],
            "build_id": manifest["build_id"],
            "vault_bind": "dev-none" if manifest.get("dev_mode") else digest.hex(),
        }.items():
            template = template.replace("__" + name + "__", value)
        destination = directory / script
        destination.write_text(template)
        destination.chmod(0o755)
    (directory / "README.txt").write_text(
        f'CVM vault delivery for {manifest["platform"]}\nGeneric bundle: {manifest["build_id"]}\n\n'
        "This directory is a self-contained platform/application delivery.\n"
        "Run launch_cvm.sh by absolute path with sudo; this directory may be root-only.\n"
        "The complete generic CVM bundle is included in cvm_bundle; no separate download is required.\n"
        "The launcher verifies that included bundle and detects and binds the required NVIDIA GPUs.\n"
        "Repeat --gpu PCI_ADDRESS for explicit GPU placement. Stop shutdown_cvm.sh by absolute path with sudo.\n"
        "Shutdown requests an orderly guest power-off through QMP before terminating QEMU.\n"
        "One vault file may be attached to only one CVM. Copy only stopped, detached disks.\n"
        "Copies retain the same identity, key authorization and revocation scope.\n"
        "The guest independently validates the vault binding and current KBS authorization.\n"
        "Only vault.qcow2 is encrypted. user_config and user_data are clear, read-only guest inputs.\n"
        "applog is deliberately clear and writable for offline operator access.\n"
        "Do not place secrets in any sidecar; use /vault/application/data for confidential data and logs.\n"
        "Keep this directory and its cvm/ modules owned by root and not group- or world-writable.\n"
    )
    return public


def create_sidecars(directory, app):
    """Create the clear sidecars; input disks are immutable to the guest."""
    for name in ("applog", "user_config", "user_data"):
        sidecar(
            directory / f"{name}.qcow2",
            app[f"{name}_drive_size"],
            app.get(name),
            verify=config.public_sidecar if name in ("user_config", "user_data") else None,
            nfs_input=name == "user_data",
        )


@contextlib.contextmanager
def profile_from_image(image, *, approved=True, plain_http=False, trusted_keys=()):
    """Keep a retrieved generic CVM available until vault packaging finishes."""
    path = Path(image)
    if path.is_dir():
        yield load_profile_set(path / "profile_set.json", approved=approved, trusted_keys=trusted_keys)
        return
    with tempfile.TemporaryDirectory(prefix="cvm-image-") as temporary:
        directory, descriptor, metadata = materialize(image, Path(temporary) / "cvm", plain_http=plain_http)
        require(descriptor["artifactType"] == CVM_ARTIFACT_TYPE, "cvm_image must reference a generic CVM artifact")
        require(
            metadata.get("kind") == "cvm_bundle" and metadata.get("state") in ("finalized", "approved"),
            "cvm_image must be finalized before building a vault",
        )
        profiles = load_profile_set(directory / "profile_set.json", approved=approved, trusted_keys=trusted_keys)
        platform = metadata.get("platform")
        require(
            isinstance(platform, str)
            and set(profiles["bundles"]) == {platform}
            and metadata.get("profile_version") == profiles["profile_version"]
            and metadata.get("build_id") == profiles["bundles"][platform]["manifest"]["build_id"],
            "CVM artifact metadata does not match its profile set",
        )
        yield profiles


def requested_platforms(app, profiles):
    platforms = app.get("platforms", list(profiles["bundles"]))
    require(
        isinstance(platforms, list)
        and platforms
        and all(isinstance(platform, str) and platform in profiles["bundles"] for platform in platforms)
        and len(set(platforms)) == len(platforms),
        "Invalid requested platform set",
    )
    return platforms


def build_dev(app, profiles, output):
    require("trustee" not in app, "Dev vaults must not receive Trustee administration credentials")
    platforms = requested_platforms(app, profiles)
    output = Path(output or Path("target") / ("vault_" + app["deployment_id"])).resolve()
    require(not output.exists(), "Output already exists")
    output.mkdir(parents=True)
    copies = []
    for platform in platforms:
        manifest = profiles["bundles"][platform]["manifest"]
        require(manifest.get("dev_mode") is True, "Cannot build a plaintext vault for a production bundle")
        directory = output / platform
        directory.mkdir()
        create_sidecars(directory, app)
        image = directory / "vault.qcow2"
        create_image(image, app["vault_drive_size"] * 1024**3)
        with nbd(image) as device:
            run(["mkfs.ext4", "-q", "-F", device])
            with mounted(device) as root:
                populate(root, app)
                internal = {
                    "schema_version": 2,
                    "deployment_id": app["deployment_id"],
                    "dev_mode": True,
                    "cvm_build_id": manifest["build_id"],
                    "platform": platform,
                    "profile_version": profiles["profile_version"],
                    "storage_profile": "plain-ext4-dev",
                    "content_sha256": content_digest(root),
                }
                write_json(root / "vault_manifest.json", internal)
            copies.append(
                delivery(directory, app, manifest, bytes(32), internal, profiles["bundles"][platform]["directory"])
            )
    write_json(
        output / "vault_set.json", {"schema_version": 2, "deployment_id": app["deployment_id"], "copies": copies}
    )
    package_deliveries(output, app["deployment_id"], copies)
    return output


def build(path, output=None, candidate=False, dev=False, plain_http=False, project_config=None):
    section = "vault_build.yml"
    try:
        app = config.application(path)
        if dev:
            require_config(
                project_config is None, "--dev must not receive --project-config or Trustee administration credentials"
            )
        else:
            section = "cvm_project.yml"
            project = config.project(path, project_config)
            # Reject stale credentials before image retrieval and vault sealing.
            # api() reads the file again at publication, so operators may still
            # rotate a valid short-lived token while a build is running.
            trustee_client.admin_token(project["trustee"])
            app["trustee"] = project["trustee"]
            app["approval_keys"] = project["approval"]["public_keys"]
    except ConfigurationError:
        raise
    except (BuildError, OSError, ValueError, KeyError, TypeError):
        # Lower-level file/service validators may interpolate confidential paths
        # or input values. Only explicitly classified schema diagnostics escape.
        raise ConfigurationError(f"Invalid {section} inputs; check referenced files and field types") from None
    app["deployment_id"] = uuid.uuid4().hex
    print(f"Deployment ID: {app['deployment_id']}", flush=True)
    with profile_from_image(
        app["cvm_image"],
        approved=not (candidate or dev),
        plain_http=plain_http,
        trusted_keys=tuple(app.get("approval_keys", ())),
    ) as profiles:
        return build_with_profile(app, profiles, output, candidate, dev)


def build_with_profile(app, profiles, output=None, candidate=False, dev=False):
    require(
        app["requires_gpu"] == (profiles["contract"]["gpu"] == "nvidia_cc"),
        "Application GPU capability must match the profile",
    )
    require(
        set(profiles["contract"]["bootstrap_egress"]) <= set(app["allowed_out_ports"]),
        "Application must preserve bootstrap egress",
    )
    app["image_id"], owner, layers = _validated_image(app["docker_archive"], app["image_id"])
    validate_vault_capacity(app, layers)
    app["_application_owner"] = _numeric_owner(app["container"].get("user")) if app["container"].get("user") else owner
    if dev:
        require(
            not candidate and profiles["profile_version"].startswith("dev-"), "Dev mode requires its own dev- profile"
        )
        return build_dev(app, profiles, output)
    platforms = requested_platforms(app, profiles)
    require("trustee" in app, "Trustee administration configuration is required")
    trusted_keys = tuple(app.get("approval_keys", ()))
    protect_process()
    output = Path(output or Path("target") / ("vault_" + app["deployment_id"])).resolve()
    require(not output.exists(), "Output already exists; choose a new deployment output")
    output.mkdir(parents=True, mode=0o700)
    published = []
    # Keep the first authenticated mapping/mount open as the source until all
    # platforms are copied. Its key FD closes after open; the kernel holds the key.
    with contextlib.ExitStack() as source_stack:
        source_root = None
        expected_content = None
        for platform in platforms:
            manifest = profiles["bundles"][platform]["manifest"]
            directory = output / platform
            directory.mkdir(mode=0o700)
            resource = None
            activated = False
            try:
                secret = os.urandom(64)
                stack = source_stack if source_root is None else contextlib.ExitStack()
                try:
                    image = directory / "vault.qcow2"
                    create_image(image, app["vault_drive_size"] * 1024**3)
                    device = stack.enter_context(nbd(image))
                    with memory_file(secret) as key:
                        format_vault(device, key)
                        mapper = stack.enter_context(opened_vault(device, key))
                    run(["mkfs.ext4", "-q", "-F", mapper])
                    root = stack.enter_context(mounted(mapper))
                    create_sidecars(directory, app)
                    if source_root is None:
                        populate(root, app)
                        source_root = root
                        expected_content = content_digest(root)
                    else:
                        copy_tree(source_root, root)
                    require(content_digest(root) == expected_content, "Platform copies differ in application content")
                    internal = {
                        "schema_version": 2,
                        "deployment_id": app["deployment_id"],
                        "platform": platform,
                        "cvm_build_id": manifest["build_id"],
                        "profile_version": profiles["profile_version"],
                        "storage_profile": STORAGE_PROFILE,
                        "vault_header_bytes": HEADER_BYTES,
                        "luks_uuid": run(["cryptsetup", "luksUUID", device]).decode().strip(),
                        "content_sha256": expected_content,
                        "candidate": candidate,
                    }
                    write_json(root / "vault_manifest.json", internal)
                    run(["sync", "-f", root])
                    header = snapshot_header(device)
                    digest = binding(header)
                    # Read every authenticated sector before publishing this key.
                    scan(mapper)
                    public = delivery(
                        directory,
                        app,
                        manifest,
                        digest,
                        internal,
                        profiles["bundles"][platform]["directory"],
                        trusted_keys,
                    )
                    resource = public["resource"]
                    require(snapshot_header(device) == header, "Header changed during construction")
                    # Native Trustee uploads may overwrite. Publish only this fresh
                    # identity and retain uncertain outcomes for operator recovery.
                    write_json(directory / "provisioning.json", {"resource": resource, "state": "uploading"})
                    trustee_client.upload_resource(app["trustee"], resource, secret)
                    activated = True
                    write_json(directory / "provisioning.json", {"resource": resource, "state": "active"})
                    published.append(public)
                    print(f"Sealed {platform}: {directory}", flush=True)
                finally:
                    if stack is not source_stack:
                        stack.close()
            except Exception:
                # Retain the identity and encrypted artifact for an admin to
                # resolve an uncertain upload;
                # never silently delete a potentially active delivery.
                write_json(
                    output / "build_failure.json",
                    {
                        "deployment_id": app["deployment_id"],
                        "platform": platform,
                        "resource": resource,
                        "upload_acknowledged": activated,
                        "completed_platforms": [p["platform"] for p in published],
                    },
                )
                raise
    write_json(
        output / "vault_set.json",
        {"schema_version": 2, "deployment_id": app["deployment_id"], "copies": published},
        mode=0o644,
    )
    package_deliveries(output, app["deployment_id"], published)
    return output
