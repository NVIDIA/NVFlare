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

"""Translate completed NVFlare kits into CVM Builder application vaults.

The included builder runs as a separate, privileged process. Its inputs, outputs and recovery
records are retained separately from participant deliveries.
"""

import copy
import hashlib
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import yaml

from nvflare.lighter.constants import CtxKey, ParticipantType, PropKey, ProvFileName
from nvflare.lighter.utils import verify_folder_signature

PLATFORMS = {"amd_sev_snp", "intel_tdx"}
DELIVERY_TYPE = "application/vnd.nvidia.cvm.delivery.v1"
SIZES = {"vault_drive_size": 8, "applog_drive_size": 1, "user_config_drive_size": 1, "user_data_drive_size": 1}
APPLICATION_SETTINGS = {
    "cvm_image",
    "docker_archive",
    "platforms",
    "requires_gpu",
    "allowed_ports",
    "allowed_out_ports",
    "allowed_in_cidrs",
    "allowed_out_cidrs",
    "user_config",
    "user_data",
    "hosts_entries",
    "tee_device",
    "host_bin",
    "workspace_uid",
    "workspace_gid",
    *SIZES,
}


def default_builder_dir():
    """The builder shipped inside this NVFlare installation."""
    return Path(__file__).resolve().parent / "image_builder"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ports(values):
    _require(isinstance(values, list), "Ports must be a list")
    _require(all(type(p) is int and 0 < p < 65536 for p in values), "Ports must be integers from 1 to 65535")
    return set(values)


def _read_json(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    _require(isinstance(value, dict), f"Expected a JSON object in {path}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_private(path, content):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)


def _tar_json(archive, name, limit=16 * 1024 * 1024):
    _require(
        isinstance(name, str) and not name.startswith("/") and ".." not in Path(name).parts,
        "Invalid archive metadata path",
    )
    matches = [member for member in archive.getmembers() if member.name == name]
    _require(
        len(matches) == 1 and matches[0].isfile() and matches[0].size <= limit,
        f"Missing, duplicate or invalid archive metadata: {name}",
    )
    raw = archive.extractfile(matches[0]).read()
    return json.loads(raw), "sha256:" + hashlib.sha256(raw).hexdigest()


def docker_image_id(path):
    """Read Docker save metadata without extracting files or using a Docker daemon."""
    try:
        with tarfile.open(path, "r:*") as archive:
            entries, _ = _tar_json(archive, "manifest.json", 1024 * 1024)
            _require(isinstance(entries, list) and entries, "Empty Docker save archive")
            images = set()
            for entry in entries:
                _require(isinstance(entry, dict), "Invalid Docker save manifest entry")
                config, digest = _tar_json(archive, entry.get("Config"))
                _require(
                    isinstance(config, dict) and config.get("os") == "linux" and config.get("architecture") == "amd64",
                    "Docker archive must contain a Linux amd64 image",
                )
                images.add(digest)
            _require(len(images) == 1, "Docker archive must contain exactly one image; save each image separately")
            return images.pop()
    except (OSError, tarfile.TarError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid docker_archive {path}: {exc}") from exc


def _delivery_config(archive_path, manifest_digest):
    with tarfile.open(archive_path, "r:*") as archive:
        index, _ = _tar_json(archive, "index.json", 1024 * 1024)
        descriptors = index.get("manifests", [])
        _require(
            len(descriptors) == 1 and descriptors[0].get("digest") == manifest_digest,
            "OCI index does not match delivery inventory",
        )
        manifest, digest = _tar_json(archive, "blobs/sha256/" + manifest_digest[7:])
        _require(
            digest == manifest_digest and manifest.get("artifactType") == DELIVERY_TYPE, "Invalid OCI delivery manifest"
        )
        config_digest = manifest.get("config", {}).get("digest", "")
        _require(re.fullmatch(r"sha256:[a-f0-9]{64}", config_digest), "Invalid OCI configuration digest")
        config, digest = _tar_json(archive, "blobs/sha256/" + config_digest[7:])
        _require(digest == config_digest and isinstance(config, dict), "Invalid OCI delivery configuration")
        return config


def _check_tree(root, public=False):
    for item in [root, *root.rglob("*")]:
        mode = item.lstat().st_mode
        _require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f"Unsupported file or symbolic link in input tree: {item}")
        if public and item.is_file():
            name = item.name.lower()
            _require(
                not name.endswith((".key", ".p12", ".pfx", ".jks"))
                and name not in ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"),
                "Private keys must be in the encrypted application workspace, not public inputs",
            )
            with item.open("rb") as stream:
                tail = b""
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    data = tail + block
                    _require(
                        not re.search(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", data),
                        "Private keys must be in the encrypted application workspace, not public inputs",
                    )
                    tail = data[-128:]


def invoke_vault_builder(builder_dir, config_file, output_dir, log_file, project_config):
    """Run once on a configured Linux worker, retaining all evidence on failure."""
    _require(sys.platform == "linux", "CVM vault construction requires a configured trusted Linux worker")
    if output_dir.exists():
        raise FileExistsError(f"Vault output already exists: {output_dir}")
    argv = [
        str(builder_dir / "cvmctl"),
        "vault",
        str(config_file),
        "--project-config",
        str(project_config),
        "--output",
        str(output_dir),
    ]
    if os.geteuid() != 0:
        argv = ["sudo", "-n", *argv]
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as log:
            result = subprocess.run(argv, cwd=builder_dir, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Vault builder exited with status {result.returncode}")
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Vault build failed; preserve {output_dir} and inspect {log_file}. "
            "Resolve possible key activation before deliberately starting a new build."
        ) from exc


def collect_artifacts(output_dir, platforms=None):
    """Read only public delivery metadata and verify archive bytes before handoff."""
    output_dir = Path(output_dir).resolve(strict=True)
    vault_set = _read_json(output_dir / "vault_set.json")
    index = _read_json(output_dir / "oci_artifacts.json")
    copies = vault_set.get("copies")
    deployment = vault_set.get("deployment_id", "")
    _require(vault_set.get("schema_version") == 2 and isinstance(copies, list) and copies, "Invalid vault_set.json")
    _require(
        isinstance(deployment, str) and re.fullmatch(r"[a-f0-9]{32}", deployment), "Invalid generated deployment ID"
    )
    _require(all(isinstance(c, dict) for c in copies), "Invalid vault copies")
    selected = [c.get("platform") for c in copies]
    _require(
        len(set(selected)) == len(copies)
        and set(selected) <= PLATFORMS
        and (platforms is None or set(selected) == set(platforms)),
        "Vault results do not match the requested platforms",
    )
    inventory = index.get("artifacts")
    _require(
        index.get("schema_version") == 1 and isinstance(inventory, dict) and len(inventory) == len(copies),
        "OCI results do not match the requested platforms",
    )
    artifacts = []
    seen = set()
    for name, entry in inventory.items():
        _require(
            isinstance(name, str) and Path(name).name == name and name not in (".", ".."), "Invalid artifact filename"
        )
        archive = output_dir / name
        _require(isinstance(entry, dict), "Invalid OCI inventory entry")
        _require(archive.is_file() and not archive.is_symlink(), f"Missing OCI artifact: {archive}")
        _require(entry.get("artifact_type") == DELIVERY_TYPE, "Expected a complete CVM delivery OCI artifact")
        _require(re.fullmatch(r"sha256:[a-f0-9]{64}", entry.get("manifest_digest", "")), "Invalid OCI digest")
        _require(_sha256(archive) == entry.get("archive_sha256"), f"OCI archive checksum mismatch: {archive}")
        identity = _delivery_config(archive, entry["manifest_digest"])
        platform = identity.get("platform")
        _require(platform in selected and platform not in seen, "Unexpected OCI delivery platform")
        seen.add(platform)
        item = next(c for c in copies if c["platform"] == platform)
        _require(
            item.get("deployment_id") == deployment
            and identity.get("deployment_id") == deployment
            and identity.get("cvm_build_id") == item.get("cvm_build_id"),
            "Unexpected vault identity",
        )
        build_id = item.get("cvm_build_id", "")
        resource = item.get("resource", "")
        _require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", build_id), "Invalid CVM build ID")
        _require(
            resource.startswith(f"keys/{build_id}/") and len(resource.split("/")) == 3 and resource.split("/")[-1],
            "Invalid vault resource identity",
        )
        artifacts.append(
            {
                "platform": platform,
                "path": str(archive),
                "archive_sha256": entry["archive_sha256"],
                "manifest_digest": entry["manifest_digest"],
                "cvm_build_id": build_id,
                "resource": resource,
            }
        )
    return {"deployment_id": deployment, "artifacts": artifacts}


class VaultAdapter:
    def __init__(self, settings, project_file, workspace_root, project):
        _require(isinstance(settings, dict) and settings, "cvm_vault must be a non-empty mapping")
        _require("cvm_profile" not in settings, "Use cvm_image with a pulled directory or registry reference")
        _require(
            not set(settings)
            - APPLICATION_SETTINGS
            - {
                "cvm_builder_dir",
                "output_root",
                "project_config",
                "participants",
                "participant_overrides",
            },
            "Unknown cvm_vault setting",
        )
        self.project_dir = Path(project_file).expanduser().resolve().parent
        self.project = project
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        project_workspace = self.workspace_root / project.name
        self.previous_production_dirs = {path.resolve() for path in project_workspace.glob("prod_*")}
        if "cvm_builder_dir" in settings:
            self.builder_dir = self._path(settings["cvm_builder_dir"], directory=True)
        else:
            self.builder_dir = default_builder_dir()
            _require(self.builder_dir.is_dir(), "The installed NVFlare package does not include CVM Builder")
        wrapper = self.builder_dir / "cvmctl"
        _require(wrapper.is_file() and os.access(wrapper, os.X_OK), "cvm_builder_dir must contain executable cvmctl")
        self.output_root = self._path(settings["output_root"], must_exist=False) if "output_root" in settings else None
        if self.output_root is not None:
            _require(
                not self.output_root.is_relative_to(self.workspace_root),
                "Explicit cvm_vault.output_root must be outside the provisioning workspace",
            )
        self.staging_root = self.output_root or project_workspace / ".cvm-vault-builds"
        _require(
            "project_config" not in settings
            or isinstance(settings["project_config"], str)
            and settings["project_config"],
            "project_config must be a non-empty path when supplied",
        )
        self.project_config = self._project_config(settings.get("project_config"))
        self.archive_ids = {}
        names = settings.get("participants")
        _require(
            isinstance(names, list)
            and names
            and all(isinstance(n, str) for n in names)
            and len(set(names)) == len(names),
            "cvm_vault.participants must explicitly list unique server/client names",
        )
        eligible = {
            p.name: p
            for p in project.get_all_participants()
            if p.type in (ParticipantType.SERVER, ParticipantType.CLIENT)
        }
        _require(set(names) <= set(eligible), "cvm_vault participants must be existing clients or servers")
        overrides = settings.get("participant_overrides", {})
        _require(
            isinstance(overrides, dict) and set(overrides) <= set(names), "Overrides must name selected participants"
        )
        self.plans = []
        for name in names:
            override = overrides.get(name, {})
            _require(
                not isinstance(override, dict) or "cvm_profile" not in override,
                "Use cvm_image with a pulled directory or registry reference in participant overrides",
            )
            _require(
                isinstance(override, dict) and not set(override) - APPLICATION_SETTINGS, "Invalid participant override"
            )
            values = {key: copy.deepcopy(value) for key, value in settings.items() if key in APPLICATION_SETTINGS}
            values.update(copy.deepcopy(override))
            # This identifies caller-side staging only. Deployment identity comes
            # exclusively from the builder's successful output metadata.
            run_name = uuid.uuid4().hex
            app, profile, image = self._application(values)
            inputs = self.staging_root / (run_name + "-inputs")
            output = self.output_root / run_name if self.output_root else None
            self.plans.append(
                {
                    "participant": eligible[name],
                    "app": app,
                    "profile": profile,
                    "image": image,
                    "inputs": inputs,
                    "output": output,
                    "uid": values.get("workspace_uid"),
                    "gid": values.get("workspace_gid"),
                }
            )
        # SignatureBuilder owns signing, before finalization and vault staging.
        for plan in self.plans:
            plan["participant"].set_prop(PropKey.CVM_VAULT, True)

    def _path(self, value, directory=False, must_exist=True):
        _require(isinstance(value, str) and value, "Expected a non-empty cvm_vault path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_dir / path
        # Refuse source-root links as well as links inside copied trees.
        _require(not path.is_symlink(), f"Symbolic link input is not supported: {path}")
        path = path.resolve()
        if must_exist:
            _require(path.is_dir() if directory else path.is_file(), f"Missing cvm_vault input: {path}")
        return path

    def _project_config(self, explicit):
        if explicit is not None:
            path = self._path(explicit)
        else:
            path = next(
                (
                    parent / "cvm_project.yml"
                    for parent in (self.project_dir, *self.project_dir.parents)
                    if (parent / "cvm_project.yml").exists() or (parent / "cvm_project.yml").is_symlink()
                ),
                None,
            )
            _require(
                path is not None, "No cvm_project.yml found beside project.yml or its ancestors; set project_config"
            )
            _require(path.is_file(), f"Invalid project configuration: {path}")
        path = path.resolve()
        with path.open() as stream:
            config = yaml.safe_load(stream)
        _require(
            isinstance(config, dict) and set(config) == {"trustee", "approval"},
            "cvm_project.yml must contain only trustee and approval",
        )
        approval = config["approval"]
        _require(
            isinstance(approval, dict) and set(approval) == {"public_keys"},
            "Project approval requires public_keys",
        )
        keys = approval["public_keys"]
        _require(
            isinstance(keys, list) and keys and all(isinstance(key, str) and key for key in keys),
            "approval.public_keys must list at least one acceptance public key path",
        )
        for key in keys:
            credential = Path(key)
            if not credential.is_absolute():
                credential = path.parent / credential
            _require(credential.is_file(), f"Missing approval public key: {credential}")
        service = config["trustee"]
        _require(
            isinstance(service, dict) and set(service) == {"url", "ca", "admin_token_file"},
            "Project trustee requires url, ca and admin_token_file",
        )
        endpoint = service["url"]
        _require(isinstance(endpoint, str), "Project trustee requires an HTTPS endpoint")
        parsed = urlparse(endpoint)
        _require(
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() for c in endpoint),
            "Project trustee requires an HTTPS endpoint without credentials, query or fragment",
        )
        for key in ("ca", "admin_token_file"):
            value = service[key]
            _require(isinstance(value, str) and value, f"Missing project trustee {key}")
            credential = Path(value)
            if not credential.is_absolute():
                credential = path.parent / credential
            _require(credential.is_file(), f"Missing project trustee credential: {credential}")
        return path.resolve()

    def _application(self, values):
        app = {key: values[key] for key in ("requires_gpu", "hosts_entries") if key in values}
        image = self._image_source(values.get("cvm_image"))
        app["cvm_image"] = str(image)
        app["docker_archive"] = str(self._path(values.get("docker_archive")))
        archive = app["docker_archive"]
        if archive not in self.archive_ids:
            self.archive_ids[archive] = docker_image_id(archive)
        app["image_id"] = self.archive_ids[archive]
        if "platforms" in values:
            app["platforms"] = values["platforms"]
            _require(
                isinstance(app["platforms"], list)
                and app["platforms"]
                and all(isinstance(p, str) for p in app["platforms"])
                and len(set(app["platforms"])) == len(app["platforms"])
                and set(app["platforms"]) <= PLATFORMS,
                "Requested platforms must be unique supported CPU platforms",
            )
        app.setdefault("requires_gpu", False)
        _require(type(app["requires_gpu"]) is bool, "requires_gpu must be boolean")
        app["allowed_ports"] = sorted(_ports(values.get("allowed_ports", [])))
        app["allowed_out_ports"] = sorted({443} | _ports(values.get("allowed_out_ports", [])))
        for key in ("allowed_in_cidrs", "allowed_out_cidrs"):
            if key in values:
                _require(
                    isinstance(values[key], list) and all(isinstance(item, str) for item in values[key]),
                    f"{key} must be a list of CIDR strings",
                )
                for item in values[key]:
                    ipaddress.ip_network(item, strict=True)
                app[key] = list(values[key])
        for key, default in SIZES.items():
            app[key] = values.get(key, default)
            _require(type(app[key]) is int and app[key] > 0, f"{key} must be a positive integer GiB size")
        for key in ("workspace_uid", "workspace_gid"):
            _require(
                key not in values or (type(values[key]) is int and values[key] >= 0),
                f"{key} must be a nonnegative integer",
            )
        for key in ("user_config", "user_data"):
            if key in values:
                path = self._path(values[key], directory=True)
                _check_tree(path, public=True)
                _require(
                    not path.is_relative_to(self.workspace_root),
                    "Public inputs cannot contain provisioning credentials",
                )
                app[key] = str(path)
        hosts = app.setdefault("hosts_entries", {})
        _require(isinstance(hosts, dict), "hosts_entries must be a mapping")
        for name, address in hosts.items():
            _require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", name), "Invalid host name")
            ipaddress.ip_address(address)
        tee_device = values.get("tee_device", False)
        _require(type(tee_device) is bool, "tee_device must be boolean")
        # NVFlare's confidential-computing authorizers default to /host/bin tools,
        # so the mount stays on for kits unless a site turns it off explicitly.
        host_bin = values.get("host_bin", True)
        _require(type(host_bin) is bool, "host_bin must be boolean")
        app["container"] = {
            "entrypoint": ["/bin/bash"],
            "command": ["/vault/application/workspace/startup/sub_start.sh", "--verify", "--foreground"],
            "env": {"NVFL_WORKSPACE": "/vault/application/runtime"},
            "volumes": [],
            "ports": [],
            "tee_device": tee_device,
            "host_bin": host_bin,
        }
        app["services"] = []
        # The builder retrieves and validates registry images itself.
        # Local metadata also permits early platform and network checks.
        profile = self._apply_profile(app, image) if isinstance(image, Path) else None
        return app, profile, image

    def _image_source(self, value):
        _require(isinstance(value, str) and value, "cvm_image must be a pulled directory or registry reference")
        local = Path(value).expanduser()
        if not local.is_absolute():
            local = self.project_dir / local
        if "://" not in value and local.exists():
            _require(local.is_dir(), "cvm_image must name a pulled directory containing profile_set.json")
            return self._path(value, directory=True)
        registry = (
            "://" in value
            or "@sha256:" in value
            or (not value.startswith(("/", "./", "../", "~")) and re.match(r"(?:localhost|[^/]*[.:][^/]*)/", value))
        )
        if not registry:
            return self._path(value, directory=True)
        reference = value
        if "://" in reference:
            scheme, reference = reference.split("://", 1)
            _require(scheme in ("oci", "https"), "cvm_image registry URLs must use oci:// or https://")
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._:/-]*)@sha256:([a-f0-9]{64})", reference)
        _require(match, "cvm_image registry references must use registry/repository@sha256:<64 lowercase hex digits>")
        repository = match.group(1)
        parsed = urlparse("https://" + repository)
        _require(
            parsed.hostname
            and parsed.path.strip("/")
            and all(part not in ("", ".", "..") for part in parsed.path.split("/")[1:])
            and (parsed.port is None or 0 < parsed.port < 65536),
            "Invalid cvm_image registry repository",
        )
        return value

    @staticmethod
    def _apply_profile(app, image):
        profile_path = image / "profile_set.json"
        _require(profile_path.is_file(), f"cvm_image must contain profile_set.json and approved CVM bundles: {image}")
        profile = _read_json(profile_path)
        _require(
            profile.get("schema_version") == 2 and isinstance(profile.get("contract"), dict),
            "Invalid CVM profile set",
        )
        bundles = profile.get("bundles")
        _require(isinstance(bundles, dict) and bundles and set(bundles) <= PLATFORMS, "Invalid CVM platform bundles")
        for platform, bundle in bundles.items():
            _require(
                isinstance(bundle, dict)
                and isinstance(bundle.get("build_id"), str)
                and re.fullmatch(r"[a-f0-9]{64}", bundle.get("manifest_sha256", "")),
                "Invalid CVM bundle metadata",
            )
            directory = image / platform
            for name in ("cvm_manifest.json", "approval.json", "resource_policy.rego"):
                _require((directory / name).is_file(), f"Missing approved CVM bundle input: {directory / name}")
            manifest_path = directory / "cvm_manifest.json"
            _require(_sha256(manifest_path) == bundle["manifest_sha256"], "CVM manifest checksum mismatch")
            manifest = _read_json(manifest_path)
            _require(
                manifest.get("build_id") == bundle["build_id"]
                and manifest.get("platform") == platform
                and manifest.get("profile_version") == profile.get("profile_version")
                and isinstance(profile.get("profile_version"), str)
                and manifest.get("contract") == profile["contract"],
                "CVM manifest does not match profile set",
            )
        _require(
            set(app.get("platforms", bundles)) <= set(bundles), "Requested platforms must be present in the CVM image"
        )
        gpu = profile["contract"].get("gpu")
        _require(gpu in ("none", "nvidia_cc"), "Invalid CVM GPU contract")
        _require(app["requires_gpu"] == (gpu == "nvidia_cc"), "requires_gpu must match the CVM profile")
        app["allowed_out_ports"] = sorted(
            _ports(app["allowed_out_ports"]) | _ports(profile["contract"].get("bootstrap_egress"))
        )
        return profile

    def _network(self, plan, ctx):
        participant = plan["participant"]
        app = plan["app"]
        server_ports = {ctx[CtxKey.FED_LEARN_PORT], ctx[CtxKey.ADMIN_PORT]}
        outgoing = _ports(app["allowed_out_ports"]) | _ports(list(server_ports))
        incoming = _ports(app["allowed_ports"])
        kit = Path(ctx[CtxKey.CURRENT_PROD_DIR]) / participant.name / "startup"
        config = _read_json(kit / f"fed_{participant.type}.json")
        for server in config.get("servers", []):
            endpoint = urlparse("//" + server["service"]["target"])
            _require(endpoint.hostname and endpoint.port, "Signed server endpoint must have a host and fixed port")
            outgoing |= _ports([endpoint.port])
            if participant.type == ParticipantType.CLIENT:
                host = app["hosts_entries"].get(endpoint.hostname, endpoint.hostname)
                _require(host.lower() != "localhost", "CVM clients require a reachable server address, not localhost")
                try:
                    address = ipaddress.ip_address(host)
                except ValueError:
                    pass  # A DNS name is resolved by the guest.
                else:
                    _require(
                        not address.is_loopback and not address.is_unspecified,
                        "CVM server address is not remotely reachable",
                    )
            else:
                incoming |= _ports([endpoint.port, server["admin_port"]])
        if participant.type == ParticipantType.SERVER:
            incoming |= server_ports
        else:
            connection = participant.get_connect_to()
            if connection and connection.port:
                outgoing |= _ports([connection.port])
            if connection and connection.name and connection.name != self.project.get_server().name:
                relays = {p.name: p for p in self.project.get_all_participants() if p.type == ParticipantType.RELAY}
                _require(connection.name in relays, "Unknown client relay")
                listener = relays[connection.name].get_listening_host()
                _require(listener and listener.port, "A CVM client requires a fixed relay listening port")
                outgoing |= _ports([listener.port])
        listener = participant.get_listening_host()
        if listener:
            _require(listener.port, "CVM listeners require a fixed port")
            incoming |= _ports([listener.port])
        app["allowed_ports"] = sorted(incoming)
        app["allowed_out_ports"] = sorted(outgoing)
        app["container"]["ports"] = [{"host": port, "container": port} for port in sorted(incoming)]

    def build(self, ctx):
        prod = ctx.get(CtxKey.CURRENT_PROD_DIR)
        _require(
            ctx.get(CtxKey.PROVISION_SUCCESS) is True and prod and Path(prod).is_dir(),
            "No successfully finalized production directory",
        )
        prod = Path(prod).resolve()
        _require(
            prod.parent == self.workspace_root / self.project.name and prod not in self.previous_production_dirs,
            "Vault construction requires a new production directory from this provisioning run",
        )
        # Validate every completed kit before any privileged build starts.
        for plan in self.plans:
            source = prod / plan["participant"].name
            _require(source.is_dir() and source.parent == prod, "Missing participant workspace")
            _check_tree(source)
            for name in ("sub_start.sh", "rootCA.pem"):
                _require((source / "startup" / name).is_file(), f"Missing startup/{name}")
            _require(
                verify_folder_signature(
                    str(source),
                    str(source / "startup/rootCA.pem"),
                    single_signer=True,
                    signature_file=ProvFileName.SIGNATURE_JSON,
                ),
                f"Finalized workspace for {plan['participant'].name} has missing or invalid signatures",
            )
            self._network(plan, ctx)
            _require(
                not plan["inputs"].exists() and (plan["output"] is None or not plan["output"].exists()),
                "Vault build staging already exists",
            )
        self.staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require(
            not self.staging_root.is_symlink() and stat.S_IMODE(self.staging_root.stat().st_mode) & 0o077 == 0,
            "Vault staging/output_root must be private (mode 0700)",
        )
        results = []
        for plan in self.plans:
            inputs = plan["inputs"]
            output = plan["output"] or prod / plan["participant"].name
            inputs.mkdir(mode=0o700)
            application = inputs / "application"
            application.mkdir(mode=0o700)
            source = prod / plan["participant"].name
            destination = application / "workspace"
            shutil.copytree(source, destination, copy_function=shutil.copy2)
            _require(
                verify_folder_signature(
                    str(destination),
                    str(destination / "startup/rootCA.pem"),
                    single_signer=True,
                    signature_file=ProvFileName.SIGNATURE_JSON,
                ),
                "Staged workspace signature verification failed; no vault was built",
            )
            for original in [source, *source.rglob("*")]:
                copied = destination / original.relative_to(source)
                info = original.stat()
                uid = info.st_uid if plan["uid"] is None else plan["uid"]
                gid = info.st_gid if plan["gid"] is None else plan["gid"]
                if (copied.stat().st_uid, copied.stat().st_gid) != (uid, gid):
                    os.chown(copied, uid, gid)
            # Keep runtime files outside the signed source kit. sub_start.sh
            # refreshes a verified working copy on each container start.
            runtime = application / "runtime"
            runtime.mkdir(mode=0o700)
            owner = destination.stat()
            if (runtime.stat().st_uid, runtime.stat().st_gid) != (owner.st_uid, owner.st_gid):
                os.chown(runtime, owner.st_uid, owner.st_gid)
            # The application parent must be traversable by the image's runtime UID.
            # The enclosing input directory remains private to the build operator.
            application.chmod(0o755)
            app = plan["app"]
            app["application_files"] = str(application)
            config_file = inputs / "vault_build.yml"
            _write_private(config_file, yaml.safe_dump(app, sort_keys=False))
            log_file = inputs / "build.log"
            if self.output_root is None:
                # Match the previous packager's participant delivery location.
                # Preserve the original signed kit instead of deleting it.
                source.rename(inputs / "startup-kit")
            invoke_vault_builder(self.builder_dir, config_file, output, log_file, self.project_config)
            platforms = app.get("platforms")
            if platforms is None and plan["profile"] is not None:
                platforms = list(plan["profile"]["bundles"])
            try:
                if os.geteuid() == 0 or os.access(output, os.R_OK | os.X_OK):
                    metadata = collect_artifacts(output, platforms)
                else:
                    # A sudo worker creates a root-only output tree. Read metadata
                    # under the same worker boundary without relaxing permissions.
                    completed = subprocess.run(
                        [
                            "sudo",
                            "-n",
                            sys.executable,
                            "-m",
                            __name__,
                            str(output),
                            *(platforms or []),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    metadata = json.loads(completed.stdout)
                artifacts = metadata["artifacts"]
                if plan["profile"] is not None:
                    expected = plan["profile"]["bundles"]
                    _require(
                        all(a["cvm_build_id"] == expected[a["platform"]]["build_id"] for a in artifacts),
                        "Vault delivery changed the selected CVM build ID",
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot collect vault delivery metadata. Preserve {inputs} and {output}; "
                    "keys may already be active. Resolve their state before rebuilding."
                ) from exc
            result = {
                "participant": plan["participant"].name,
                "deployment_id": metadata["deployment_id"],
                "output_dir": str(output),
                "artifacts": artifacts,
            }
            _write_private(inputs / "result.json", json.dumps(result, indent=2) + "\n")
            results.append(result)
            ctx.info(f"CVM vault delivery for {result['participant']}: {output}")
        return results


if __name__ == "__main__":
    # Internal metadata collector for the preconfigured privileged worker.
    print(json.dumps(collect_artifacts(Path(sys.argv[1]), sys.argv[2:] or None)))
