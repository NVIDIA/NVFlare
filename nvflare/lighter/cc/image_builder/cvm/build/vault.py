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
import hashlib
import json
import os
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
from .storage import content_digest, copy_tree, create_image, format_vault, mounted, nbd, opened_vault, sidecar


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
    for service in app["services"]:
        shutil.copyfile(service, root / "services" / Path(service).name)


def validate_archive(path, image_id):
    try:
        return _archive_image_id(path, image_id)
    except (BuildError, OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, tarfile.TarError):
        # Archives can embed private paths, image metadata or credentials. Do not
        # expose parser exceptions, even though this check precedes key creation.
        raise ConfigurationError("Invalid docker_archive; regenerate it with docker save and verify image_id") from None


def _archive_image_id(path, image_id):
    # Only inspect manifest/config members, never unpack layers on the build host.

    with tarfile.open(path, "r:*") as archive:
        member = archive.getmember("manifest.json")
        require(member.isfile() and member.size < 1024**2, "Invalid Docker save manifest")
        entries = json.load(archive.extractfile(member))
        require(isinstance(entries, list) and entries, "Empty Docker save archive")
        configurations = set()
        for entry in entries:
            filename = entry["Config"]
            require(
                isinstance(filename, str) and not filename.startswith("/") and ".." not in Path(filename).parts,
                "Invalid Docker config member",
            )
            member = archive.getmember(filename)
            require(member.isfile() and member.size < 16 * 1024**2, "Invalid Docker image configuration")
            data = archive.extractfile(member).read()
            configurations.add("sha256:" + hashlib.sha256(data).hexdigest())
        if image_id in configurations:
            return image_id
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
                return image
        raise BuildError("OCI manifest nesting exceeds the supported limit")


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
        "Run: sudo ./launch_cvm.sh\n"
        "The complete generic CVM bundle is included in cvm_bundle; no separate download is required.\n"
        "The launcher verifies that included bundle and detects and binds the required NVIDIA GPUs.\n"
        "Repeat --gpu PCI_ADDRESS for explicit GPU placement. Stop with: sudo ./shutdown_cvm.sh\n"
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
    if candidate:
        require(
            profiles["profile_version"].startswith("test-"), "Candidate vaults require a separate test- profile version"
        )
    require(
        app["requires_gpu"] == (profiles["contract"]["gpu"] == "nvidia_cc"),
        "Application GPU capability must match the profile",
    )
    require(
        set(profiles["contract"]["bootstrap_egress"]) <= set(app["allowed_out_ports"]),
        "Application must preserve bootstrap egress",
    )
    app["image_id"] = validate_archive(app["docker_archive"], app["image_id"])
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
    # First copy remains mounted as the authenticated source for later copies.
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
