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

"""Create, validate, publish and pull OCI artifacts."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from ..common.contracts import PLATFORMS
from ..common.errors import BuildError, require
from ..common.io import canonical, digest_file, read_json, write_json
from ..common.linux import lock, run

OCI_LAYOUT_VERSION = "1.0.0"


INDEX_MEDIA_TYPE = "application/vnd.oci.image.index.v1+json"


MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"


CVM_ARTIFACT_TYPE = "application/vnd.nvidia.cvm.bundle.v1"


DELIVERY_ARTIFACT_TYPE = "application/vnd.nvidia.cvm.delivery.v1"


CVM_CONFIG_MEDIA_TYPE = "application/vnd.nvidia.cvm.bundle.config.v1+json"


DELIVERY_CONFIG_MEDIA_TYPE = "application/vnd.nvidia.cvm.delivery.config.v1+json"


CVM_LAYER_MEDIA_TYPE = "application/vnd.nvidia.cvm.bundle.layer.v1.tar+gzip"


VAULT_LAYER_MEDIA_TYPE = "application/vnd.nvidia.cvm.vault.layer.v1.tar+gzip"


REF_NAME = "artifact"


SUPPORTED_ARTIFACT_TYPES = {CVM_ARTIFACT_TYPE, DELIVERY_ARTIFACT_TYPE}


def _descriptor(media_type, digest, size, **extra):
    result = {"mediaType": media_type, "digest": "sha256:" + digest, "size": size}
    result.update(extra)
    return result


def _blob(layout, data, media_type, **extra):
    digest = hashlib.sha256(data).hexdigest()
    destination = layout / "blobs" / "sha256" / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return _descriptor(media_type, digest, len(data), **extra)


def _safe_name(name):
    value = PurePosixPath(name)
    require(name and not value.is_absolute() and ".." not in value.parts, "Unsafe OCI artifact member")
    return value.as_posix()


def _tar_info(archive, source, name):
    info = archive.gettarinfo(str(source), arcname=_safe_name(name))
    require(info.isdir() or info.isfile(), "OCI artifacts support only regular files and directories")
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.mode &= 0o777
    info.pax_headers = {}
    return info


def _add_path(archive, source, name):
    source = Path(source)
    require(source.exists(), f"Missing OCI artifact member: {source.name}")
    if "__pycache__" in source.parts or source.name.endswith(".pyc"):
        return
    info = _tar_info(archive, source, name)
    if info.isdir():
        archive.addfile(info)
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            _add_path(archive, child, PurePosixPath(name, child.name).as_posix())
    else:
        with source.open("rb") as content:
            archive.addfile(info, content)


def _layer(layout, spec):
    temporary = layout / "layer.partial"
    with temporary.open("xb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for source, name in spec["members"]:
                    _add_path(archive, source, name)
        output.flush()
        os.fsync(output.fileno())
    digest = digest_file(temporary)
    destination = layout / "blobs" / "sha256" / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    return _descriptor(
        spec["media_type"],
        digest,
        destination.stat().st_size,
        annotations={"org.opencontainers.image.title": spec["title"]},
    )


def _archive_layout(layout, destination):
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    with temporary.open("xb") as output:
        os.fchmod(output.fileno(), 0o600)
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for source in sorted(layout.rglob("*"), key=lambda item: item.relative_to(layout).as_posix()):
                name = source.relative_to(layout).as_posix()
                info = _tar_info(archive, source, name)
                if info.isdir():
                    archive.addfile(info)
                else:
                    with source.open("rb") as content:
                        archive.addfile(info, content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)


def create(destination, artifact_type, config_media_type, config, layers, annotations=None):
    """Write a deterministic OCI image-layout tar and return its manifest descriptor."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".oci-", dir=destination.parent) as temporary:
        layout = Path(temporary)
        (layout / "oci-layout").write_bytes(canonical({"imageLayoutVersion": OCI_LAYOUT_VERSION}) + b"\n")
        config_descriptor = _blob(layout, canonical(config), config_media_type)
        layer_descriptors = [_layer(layout, layer) for layer in layers]
        manifest = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "artifactType": artifact_type,
            "config": config_descriptor,
            "layers": layer_descriptors,
            "annotations": annotations or {},
        }
        manifest_descriptor = _blob(
            layout,
            canonical(manifest),
            MANIFEST_MEDIA_TYPE,
            artifactType=artifact_type,
            annotations={"org.opencontainers.image.ref.name": REF_NAME},
        )
        index = {"schemaVersion": 2, "mediaType": INDEX_MEDIA_TYPE, "manifests": [manifest_descriptor]}
        (layout / "index.json").write_bytes(canonical(index) + b"\n")
        _archive_layout(layout, destination)
    return manifest_descriptor


def _copy_member(archive, member, destination):
    source = archive.extractfile(member)
    require(source is not None, "Unreadable OCI layout member")
    digest = hashlib.sha256()
    with destination.open("xb") as output:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            output.write(block)
    return digest.hexdigest()


def _archive_to_layout(source, layout):
    layout.mkdir()
    seen = set()
    with tarfile.open(source, "r:") as archive:
        for member in archive:
            name = _safe_name(member.name)
            require(name not in seen, "Duplicate OCI layout member")
            seen.add(name)
            allowed = name in {"oci-layout", "index.json"} or (
                len(PurePosixPath(name).parts) == 3
                and PurePosixPath(name).parts[:2] == ("blobs", "sha256")
                and len(PurePosixPath(name).name) == 64
                and all(character in "0123456789abcdef" for character in PurePosixPath(name).name)
            )
            if member.isdir():
                require(name in {"blobs", "blobs/sha256"}, "Unexpected OCI layout directory")
                (layout / name).mkdir(parents=True, exist_ok=True)
                continue
            require(allowed and member.isfile(), "Unexpected OCI layout member")
            if name in {"oci-layout", "index.json"}:
                require(member.size <= 1024 * 1024, "Oversized OCI metadata")
            destination = layout / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            actual = _copy_member(archive, member, destination)
            if name.startswith("blobs/sha256/"):
                require(actual == PurePosixPath(name).name, "OCI blob digest mismatch")


def _json(path, limit=1024 * 1024):
    require(path.is_file() and path.stat().st_size <= limit, "Invalid OCI metadata")
    try:
        return json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeError):
        raise BuildError("Invalid OCI JSON metadata") from None


def _validated_blob(layout, descriptor, expected_media_type=None, limit=None):
    require(isinstance(descriptor, dict), "Invalid OCI descriptor")
    digest = descriptor.get("digest", "")
    require(
        digest.startswith("sha256:")
        and len(digest) == 71
        and all(character in "0123456789abcdef" for character in digest[7:]),
        "Invalid OCI descriptor digest",
    )
    if expected_media_type:
        require(descriptor.get("mediaType") == expected_media_type, "Unexpected OCI media type")
    path = layout / "blobs" / "sha256" / digest[7:]
    require(path.is_file() and path.stat().st_size == descriptor.get("size"), "OCI descriptor size mismatch")
    if limit is not None:
        require(path.stat().st_size <= limit, "Oversized OCI metadata blob")
    require(digest_file(path) == digest[7:], "OCI descriptor digest mismatch")
    return path


def inspect_layout(layout):
    layout = Path(layout)
    require(_json(layout / "oci-layout") == {"imageLayoutVersion": OCI_LAYOUT_VERSION}, "Unsupported OCI layout")
    index = _json(layout / "index.json")
    require(index.get("schemaVersion") == 2 and index.get("mediaType") == INDEX_MEDIA_TYPE, "Invalid OCI index")
    manifests = index.get("manifests")
    require(isinstance(manifests, list) and len(manifests) == 1, "OCI artifact must contain one manifest")
    manifest_path = _validated_blob(layout, manifests[0], MANIFEST_MEDIA_TYPE, 1024 * 1024)
    manifest = _json(manifest_path)
    artifact_type = manifest.get("artifactType")
    require(
        manifest.get("schemaVersion") == 2
        and manifest.get("mediaType") == MANIFEST_MEDIA_TYPE
        and artifact_type in SUPPORTED_ARTIFACT_TYPES,
        "Unsupported CVM OCI artifact",
    )
    expected_config = CVM_CONFIG_MEDIA_TYPE if artifact_type == CVM_ARTIFACT_TYPE else DELIVERY_CONFIG_MEDIA_TYPE
    config_path = _validated_blob(layout, manifest.get("config"), expected_config, 1024 * 1024)
    config = _json(config_path)
    require(isinstance(config, dict), "Invalid OCI artifact config")
    materialized_name = config.get("materialized_name")
    require(
        isinstance(materialized_name, str)
        and materialized_name not in {"", ".", ".."}
        and PurePosixPath(materialized_name).name == materialized_name,
        "Invalid OCI materialized directory name",
    )
    layers = manifest.get("layers")
    require(isinstance(layers, list) and layers, "OCI artifact has no layers")
    expected_sequence = (
        [CVM_LAYER_MEDIA_TYPE] if artifact_type == CVM_ARTIFACT_TYPE else [CVM_LAYER_MEDIA_TYPE, VAULT_LAYER_MEDIA_TYPE]
    )
    require(
        [layer.get("mediaType") for layer in layers] == expected_sequence,
        "Unexpected CVM OCI layer sequence",
    )
    require(
        config.get("kind") == ("cvm_bundle" if artifact_type == CVM_ARTIFACT_TYPE else "cvm_delivery"),
        "OCI artifact config does not match its type",
    )
    launch_directory = config.get("launch_directory")
    require(
        launch_directory is None
        or (
            artifact_type == DELIVERY_ARTIFACT_TYPE
            and isinstance(launch_directory, str)
            and PurePosixPath(launch_directory).name == launch_directory
            and launch_directory not in {"", ".", ".."}
        ),
        "Invalid OCI launch directory",
    )
    for layer in layers:
        _validated_blob(layout, layer, layer["mediaType"])
    descriptor = dict(manifests[0])
    descriptor["artifactType"] = artifact_type
    return descriptor, manifest, config


def _extract_layer(layer_path, output, written):
    with tarfile.open(layer_path, "r:gz") as archive:
        for member in archive:
            name = _safe_name(member.name)
            require(name not in written, "OCI layers contain a duplicate path")
            written.add(name)
            destination = output / name
            require(destination == output / PurePosixPath(name), "Unsafe OCI layer path")
            require(member.isdir() or member.isfile(), "OCI layers support only regular files and directories")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(member.mode & 0o777)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                require(source is not None, "Unreadable OCI layer member")
                with destination.open("xb") as stream:
                    shutil.copyfileobj(source, stream, 1024 * 1024)
                destination.chmod(member.mode & 0o777)


def materialize_layout(layout, output):
    """Verify an OCI layout and atomically materialize its runtime directory."""
    descriptor, manifest, config = inspect_layout(layout)
    output = Path(output).resolve()
    require(not output.exists(), "Materialization output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    temporary.chmod(0o700)
    try:
        written = set()
        for layer in manifest["layers"]:
            path = Path(layout) / "blobs" / "sha256" / layer["digest"][7:]
            _extract_layer(path, temporary, written)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return descriptor, config


def _merge_cvm(incoming, output, config):
    require(config.get("state") in {"finalized", "approved"}, "Only a finalized CVM artifact can be merged")
    platform = config.get("platform")
    require(isinstance(platform, str) and platform in PLATFORMS, "Invalid CVM platform")
    incoming_profile = read_json(incoming / "profile_set.json")
    require(
        incoming_profile.get("profile_version") == config.get("profile_version")
        and set(incoming_profile.get("bundles", {})) == {platform},
        "CVM artifact profile does not match its config",
    )
    entry = incoming_profile["bundles"][platform]
    incoming_bundle = incoming / platform
    require(
        digest_file(incoming_bundle / "cvm_manifest.json") == entry.get("manifest_sha256"),
        "CVM profile manifest digest mismatch",
    )
    record_path = output / "profile_set.json"
    with lock(output.parent / ("." + output.name + ".merge.lock")):
        current = read_json(record_path)
        require(
            current.get("profile_version") == incoming_profile["profile_version"]
            and current.get("contract") == incoming_profile.get("contract"),
            "Cannot merge CVM artifacts from different profiles",
        )
        destination = output / platform
        existing = current.get("bundles", {}).get(platform)
        if existing is not None:
            require(existing == entry and destination.is_dir(), "CVM platform already has different content")
            require(
                digest_file(destination / "cvm_manifest.json") == entry["manifest_sha256"],
                "Existing CVM platform manifest changed",
            )
            return
        if destination.exists():
            require(
                destination.is_dir() and digest_file(destination / "cvm_manifest.json") == entry["manifest_sha256"],
                "Untracked CVM platform directory conflicts with the artifact",
            )
        else:
            os.replace(incoming_bundle, destination)
        current["bundles"][platform] = entry
        write_json(record_path, current, mode=0o644)


def materialize(source, output=None, merge=False, plain_http=False):
    """Materialize a local OCI-layout tar or an immutable registry reference."""
    source_path = Path(source).expanduser()
    with tempfile.TemporaryDirectory(prefix="cvm-oci-") as temporary:
        layout = Path(temporary) / "layout"
        if source_path.is_file():
            _archive_to_layout(source_path, layout)
        else:
            reference = str(source)
            if reference.startswith("oci://"):
                reference = reference[6:]
            require("@sha256:" in reference, "Registry delivery must use an immutable digest reference")
            command = ["oras", "cp", "--to-oci-layout"]
            if plain_http:
                command.append("--from-plain-http")
            run([*command, reference, f"{layout}:{REF_NAME}"], timeout=None)
        _, _, config = inspect_layout(layout)
        destination = output or config["materialized_name"]
        require(destination, "OCI artifact config lacks a materialized directory name")
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with lock(destination.parent / ("." + destination.name + ".materialize.lock")):
            if merge and destination.exists():
                with tempfile.TemporaryDirectory(prefix="cvm-merge-", dir=destination.parent) as merge_root:
                    incoming = Path(merge_root) / "incoming"
                    descriptor, config = materialize_layout(layout, incoming)
                    require(descriptor["artifactType"] == CVM_ARTIFACT_TYPE, "Only CVM bundle artifacts can be merged")
                    _merge_cvm(incoming, destination, config)
            else:
                descriptor, config = materialize_layout(layout, destination)
    return destination, descriptor, config


def publish(source, destination, plain_http=False):
    """Copy a verified OCI-layout tar into a registry using ORAS."""
    source = Path(source).expanduser().resolve()
    require(source.is_file(), "OCI artifact tar does not exist")
    with tempfile.TemporaryDirectory(prefix="cvm-oci-") as temporary:
        layout = Path(temporary) / "layout"
        _archive_to_layout(source, layout)
        descriptor, _, _ = inspect_layout(layout)
        reference = str(destination)
        if reference.startswith("oci://"):
            reference = reference[6:]
        command = ["oras", "cp", "--from-oci-layout"]
        if plain_http:
            command.append("--to-plain-http")
        run([*command, f"{layout}:{REF_NAME}", reference], timeout=None)
    repository = reference.split("@", 1)[0]
    last_slash = repository.rfind("/")
    if ":" in repository[last_slash + 1 :]:
        repository = repository[: repository.rfind(":")]
    return repository + "@" + descriptor["digest"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    pull = actions.add_parser("pull", help="materialize a local OCI tar or immutable registry artifact")
    pull.add_argument("source")
    pull.add_argument("--output")
    pull.add_argument("--merge", action="store_true", help="add a finalized CVM platform to an existing profile")
    pull.add_argument("--plain-http", action="store_true", help="allow an unencrypted test registry connection")
    push = actions.add_parser("publish", help="copy a local OCI-layout tar to a registry")
    push.add_argument("source")
    push.add_argument("destination")
    push.add_argument("--plain-http", action="store_true", help="allow an unencrypted test registry connection")
    args = parser.parse_args()
    try:
        if args.action == "pull":
            output, descriptor, config = materialize(args.source, args.output, args.merge, args.plain_http)
            print(f"Materialized {descriptor['artifactType']} {descriptor['digest']} at {output}")
            if config.get("launch_directory"):
                print(f"Launch: cd {output / config['launch_directory']} && sudo ./launch_cvm.sh")
        else:
            print(f"Published: {publish(args.source, args.destination, args.plain_http)}")
    except (BuildError, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"OCI operation failed: {exc}\n")


if __name__ == "__main__":
    main()
