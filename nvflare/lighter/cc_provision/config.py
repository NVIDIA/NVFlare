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

"""Strict unified confidential-computing configuration validation."""

import ipaddress
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping
from urllib.parse import urlsplit

from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer
from nvflare.lighter import utils
from nvflare.lighter.cc_provision.deployment import (
    CPUTEE,
    GPUTEE,
    CCDeploymentMode,
    ResolvedAttestationService,
    WorkloadSource,
    WorkloadSourceType,
)

SCHEMA_VERSION = 1
PROJECT_FIELDS = {"schema_version", "attestation_services", "container_registries", "approval", "build_tools"}
PARTICIPANT_FIELDS = {
    "schema_version",
    "cc_deployment_mode",
    "cpu_tee",
    "gpu_tee",
    "attestation",
    "class_allow_list",
    "workload",
    *(mode.value for mode in CCDeploymentMode),
}
LEGACY_FIELDS = {
    "compute_env",
    "cc_cpu_mechanism",
    "cc_gpu",
    "cc_issuers",
    "cc_attestation",
    "role",
    "image_build",
    "platforms",
    "requires_gpu",
}
CLASS_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
REPOSITORY = re.compile(r"[a-z0-9]+([._-][a-z0-9]+)*(/[a-z0-9]+([._-][a-z0-9]+)*)*")
DNS_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact(value, allowed: Iterable[str], name, required=()):
    value = _mapping(value, name)
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown:
        raise ValueError(f"Unsupported {name} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing {name} fields: {', '.join(sorted(missing))}")
    return value


def _schema(value, name):
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{name}.schema_version must be {SCHEMA_VERSION}")


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value, name, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside the supported range")
    return value


def _https_endpoint(value, name):
    _nonempty_string(value, name)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in value)
    ):
        raise ValueError(f"{name} must be an HTTPS endpoint without credentials, query, or fragment")


def _path(config_path: Path, value, name, *, required=True, directory=False, executable=False):
    _nonempty_string(value, name)
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    if result.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {result}")
    result = result.resolve()
    if required and not (result.is_dir() if directory else result.is_file()):
        kind = "directory" if directory else "file"
        raise ValueError(f"{name} does not name an existing {kind}: {result}")
    if executable and (not result.is_file() or not result.stat().st_mode & 0o111):
        raise ValueError(f"{name} must name an executable file: {result}")
    return result


def _immutable(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    return value


def load_project_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    value = utils.load_yaml(str(config_path))
    config = _exact(value, PROJECT_FIELDS, "cc_project", required=("schema_version", "attestation_services"))
    _schema(config, "cc_project")

    services = _mapping(config["attestation_services"], "attestation_services")
    if not services:
        raise ValueError("attestation_services must not be empty")
    for name, service in services.items():
        _nonempty_string(name, "attestation service name")
        _validate_service(name, service, config_path)

    registries = _mapping(config.get("container_registries", {}), "container_registries")
    for name, registry in registries.items():
        _nonempty_string(name, "container registry name")
        registry = _exact(
            registry,
            {"endpoint", "ca_cert_file", "publisher_username_file", "publisher_password_file"},
            f"container_registries.{name}",
            required=("endpoint", "ca_cert_file", "publisher_username_file", "publisher_password_file"),
        )
        endpoint = _nonempty_string(registry["endpoint"], f"container_registries.{name}.endpoint")
        parsed = urlsplit("https://" + endpoint)
        try:
            port = parsed.port
        except ValueError:
            port = -1
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or any(char.isspace() for char in endpoint)
            or port is not None
            and not 0 < port < 65536
        ):
            raise ValueError(f"container_registries.{name}.endpoint must be a registry host[:port]")
        for field in ("ca_cert_file", "publisher_username_file", "publisher_password_file"):
            registry[field] = str(_path(config_path, registry[field], f"container_registries.{name}.{field}"))

    if "approval" in config:
        approval = _exact(config["approval"], {"public_key_files"}, "approval", required=("public_key_files",))
        keys = approval["public_key_files"]
        if not isinstance(keys, list) or not keys:
            raise ValueError("approval.public_key_files must be a non-empty list")
        approval["public_key_files"] = [
            str(_path(config_path, value, "approval.public_key_files entry")) for value in keys
        ]

    tools = _mapping(config.get("build_tools", {}), "build_tools")
    if set(tools) - {CCDeploymentMode.BARE_METAL_CVM.value, CCDeploymentMode.COCO.value}:
        raise ValueError("Unsupported build_tools deployment mode")
    if CCDeploymentMode.BARE_METAL_CVM.value in tools:
        settings = _exact(
            tools[CCDeploymentMode.BARE_METAL_CVM.value],
            {"cvm_builder_dir", "output_root"},
            "build_tools.bare_metal_cvm",
            required=("cvm_builder_dir",),
        )
        builder = _path(
            config_path,
            settings["cvm_builder_dir"],
            "build_tools.bare_metal_cvm.cvm_builder_dir",
            directory=True,
        )
        settings["cvm_builder_dir"] = str(builder)
        _path(
            config_path,
            str(builder / "cvmctl"),
            "build_tools.bare_metal_cvm.cvm_builder_dir/cvmctl",
            executable=True,
        )
        if "output_root" in settings:
            settings["output_root"] = str(
                _path(
                    config_path,
                    settings["output_root"],
                    "build_tools.bare_metal_cvm.output_root",
                    required=False,
                )
            )
    if CCDeploymentMode.COCO.value in tools:
        settings = _exact(
            tools[CCDeploymentMode.COCO.value],
            {"build_command", "build_timeout_seconds"},
            "build_tools.coco",
            required=("build_command",),
        )
        settings["build_command"] = str(
            _path(config_path, settings["build_command"], "build_tools.coco.build_command", executable=True)
        )
        if "build_timeout_seconds" in settings:
            _positive_int(settings["build_timeout_seconds"], "build_tools.coco.build_timeout_seconds")
    config["_config_path"] = config_path
    return config


def _validate_service(name, value, config_path):
    service = _mapping(value, f"attestation_services.{name}")
    service_type = service.get("type")
    if service_type == "trustee":
        allowed = {
            "type",
            "kbs_endpoint",
            "ca_cert_file",
            "admin_token_file",
            "attestation_token_endpoint",
            "attestation_signing_public_key_file",
            "token_expiration_seconds",
            "check_frequency_seconds",
            "registration_token_timeout_seconds",
            "refresh_token_timeout_seconds",
            "get_token_request_timeout_seconds",
            "retry",
            "proof_iat_leeway_seconds",
            "workload_constraints",
        }
        required = allowed - {"retry", "proof_iat_leeway_seconds", "workload_constraints"}
        _exact(service, allowed, f"attestation_services.{name}", required=required)
        _https_endpoint(service["kbs_endpoint"], f"attestation_services.{name}.kbs_endpoint")
        endpoint = urlsplit(service["attestation_token_endpoint"])
        if (
            endpoint.scheme != "http"
            or endpoint.hostname != "127.0.0.1"
            or endpoint.path != "/aa/token"
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                f"attestation_services.{name}.attestation_token_endpoint must be the guest-local "
                "http://127.0.0.1:<port>/aa/token endpoint"
            )
        for field in ("ca_cert_file", "admin_token_file", "attestation_signing_public_key_file"):
            service[field] = str(_path(config_path, service[field], f"attestation_services.{name}.{field}"))
        age = _positive_int(
            service["token_expiration_seconds"], f"attestation_services.{name}.token_expiration_seconds"
        )
        frequency = _positive_int(
            service["check_frequency_seconds"], f"attestation_services.{name}.check_frequency_seconds"
        )
        if age > 300 or frequency >= age:
            raise ValueError("Trustee requires 0 < check_frequency_seconds < token_expiration_seconds <= 300")
        for field in (
            "registration_token_timeout_seconds",
            "refresh_token_timeout_seconds",
            "get_token_request_timeout_seconds",
        ):
            _positive_int(service[field], f"attestation_services.{name}.{field}")
        retry_fields = {
            "max_attempts",
            "initial_delay_seconds",
            "max_delay_seconds",
            "backoff_multiplier",
            "jitter_ratio",
        }
        retry = _exact(
            service.get("retry", {}),
            retry_fields,
            f"attestation_services.{name}.retry",
            required=retry_fields if "retry" in service else (),
        )
        if retry:
            _positive_int(retry.get("max_attempts"), f"attestation_services.{name}.retry.max_attempts")
            initial = _number(
                retry.get("initial_delay_seconds"),
                f"attestation_services.{name}.retry.initial_delay_seconds",
                minimum=0.000001,
            )
            maximum = _number(
                retry.get("max_delay_seconds"),
                f"attestation_services.{name}.retry.max_delay_seconds",
                minimum=initial,
            )
            _number(
                retry.get("backoff_multiplier"),
                f"attestation_services.{name}.retry.backoff_multiplier",
                minimum=1,
            )
            _number(retry.get("jitter_ratio"), f"attestation_services.{name}.retry.jitter_ratio", minimum=0, maximum=1)
        constraints = service.get("workload_constraints")
        if isinstance(constraints, dict) and any(
            isinstance(pins, dict) and "gpu_required" in pins for pins in constraints.values()
        ):
            raise ValueError(
                f"attestation_services.{name}.workload_constraints must not set gpu_required; "
                "it is derived from each participant's gpu_tee"
            )
        # Reuse the runtime verifier's strict key and policy validation so the
        # project fails before it emits any signed startup kits.
        CoCoAuthorizer(
            trustee_public_key=_path(
                config_path,
                service["attestation_signing_public_key_file"],
                f"attestation_services.{name}.attestation_signing_public_key_file",
            ).read_text(),
            audience="nvflare-config-validation",
            token_url=service["attestation_token_endpoint"],
            max_token_age_seconds=age,
            **(
                {"proof_iat_leeway_seconds": service["proof_iat_leeway_seconds"]}
                if "proof_iat_leeway_seconds" in service
                else {}
            ),
            **({"workload_constraints": service["workload_constraints"]} if "workload_constraints" in service else {}),
            **({"retry_max_attempts": retry["max_attempts"]} if retry else {}),
            **({"retry_initial_delay": retry["initial_delay_seconds"]} if retry else {}),
            **({"retry_max_delay": retry["max_delay_seconds"]} if retry else {}),
            **({"retry_backoff_multiplier": retry["backoff_multiplier"]} if retry else {}),
            **({"retry_jitter_ratio": retry["jitter_ratio"]} if retry else {}),
        )
    elif service_type == "azure_maa":
        _exact(
            service,
            {"type", "endpoint", "token_expiration_seconds", "check_frequency_seconds"},
            f"attestation_services.{name}",
            required=("type", "endpoint", "token_expiration_seconds", "check_frequency_seconds"),
        )
        _https_endpoint(service["endpoint"], f"attestation_services.{name}.endpoint")
        if urlsplit(service["endpoint"]).path not in ("", "/"):
            raise ValueError(f"attestation_services.{name}.endpoint must not contain a path")
        age = _positive_int(
            service["token_expiration_seconds"], f"attestation_services.{name}.token_expiration_seconds"
        )
        frequency = _positive_int(
            service["check_frequency_seconds"], f"attestation_services.{name}.check_frequency_seconds"
        )
        if frequency >= age:
            raise ValueError("azure_maa requires check_frequency_seconds < token_expiration_seconds")
    else:
        raise ValueError(f"attestation_services.{name}.type must be trustee or azure_maa")


def load_participant_config(config_path: Path, project_config: Mapping[str, Any]):
    config_path = config_path.expanduser().resolve()
    value = utils.load_yaml(str(config_path))
    config = _exact(
        value,
        PARTICIPANT_FIELDS,
        "cc_config",
        required=("schema_version", "cc_deployment_mode", "cpu_tee", "gpu_tee", "attestation", "workload"),
    )
    legacy = set(config) & LEGACY_FIELDS
    if legacy:
        raise ValueError(f"Legacy CC fields are not supported: {', '.join(sorted(legacy))}")
    _schema(config, "cc_config")
    try:
        mode = CCDeploymentMode(config["cc_deployment_mode"])
    except (ValueError, TypeError):
        raise ValueError("cc_deployment_mode must be bare_metal_cvm, coco, or azure_cc") from None
    try:
        cpu = CPUTEE(config["cpu_tee"])
    except (ValueError, TypeError):
        raise ValueError("cpu_tee must be intel_tdx or amd_sev_snp") from None
    try:
        gpu = GPUTEE(config["gpu_tee"])
    except (ValueError, TypeError):
        raise ValueError("gpu_tee must be none or nvidia_cc") from None

    attestation = _exact(config["attestation"], {"service"}, "attestation", required=("service",))
    service_name = _nonempty_string(attestation["service"], "attestation.service")
    services = project_config["attestation_services"]
    if service_name not in services:
        raise ValueError(f"Unknown attestation.service: {service_name}")
    service = services[service_name]
    expected_type = "azure_maa" if mode is CCDeploymentMode.AZURE_CC else "trustee"
    if service["type"] != expected_type:
        raise ValueError(f"{mode.value} requires an attestation service of type {expected_type}")

    allow_list = config.get("class_allow_list", [])
    if not isinstance(allow_list, list) or any(
        not isinstance(item, str) or not CLASS_PATH.fullmatch(item) for item in allow_list
    ):
        raise ValueError(
            "class_allow_list entries must be fully qualified class names; wildcards and prefixes are forbidden"
        )
    if len(set(allow_list)) != len(allow_list):
        raise ValueError("class_allow_list entries must be unique")

    workload = _exact(config["workload"], {"source"}, "workload", required=("source",))
    source = _mapping(workload["source"], "workload.source")
    try:
        source_type = WorkloadSourceType(source.get("type"))
    except (ValueError, TypeError):
        raise ValueError("workload.source.type must be docker_archive, docker_build, or external") from None
    expected_source = {
        CCDeploymentMode.BARE_METAL_CVM: WorkloadSourceType.DOCKER_ARCHIVE,
        CCDeploymentMode.COCO: WorkloadSourceType.DOCKER_BUILD,
        CCDeploymentMode.AZURE_CC: WorkloadSourceType.EXTERNAL,
    }[mode]
    if source_type is not expected_source:
        raise ValueError(f"{mode.value} requires workload.source.type: {expected_source.value}")
    source_values = _validate_source(source_type, source, config_path)

    mode_blocks = set(config) & {item.value for item in CCDeploymentMode}
    if mode_blocks != {mode.value}:
        raise ValueError(f"cc_config must contain exactly the selected {mode.value} block")
    mode_config = _validate_mode(mode, config[mode.value], cpu, gpu, config_path, project_config)
    resolved_service = ResolvedAttestationService(
        name=service_name,
        service_type=service["type"],
        config_path=Path(project_config["_config_path"]),
        values=_immutable(service),
    )
    return {
        "raw": config,
        "config_path": config_path,
        "mode": mode,
        "cpu_tee": cpu,
        "gpu_tee": gpu,
        "attestation_service": resolved_service,
        "class_allow_list": tuple(allow_list),
        "workload_source": WorkloadSource(source_type=source_type, values=_immutable(source_values)),
        "mode_config": _immutable(mode_config),
    }


def _validate_source(source_type, source, config_path):
    if source_type is WorkloadSourceType.DOCKER_ARCHIVE:
        _exact(source, {"type", "path"}, "workload.source", required=("type", "path"))
        return {"path": _path(config_path, source["path"], "workload.source.path")}
    if source_type is WorkloadSourceType.DOCKER_BUILD:
        _exact(source, {"type", "context", "dockerfile"}, "workload.source", required=("type", "context", "dockerfile"))
        context = _path(config_path, source["context"], "workload.source.context", directory=True)
        dockerfile = Path(_nonempty_string(source["dockerfile"], "workload.source.dockerfile"))
        if dockerfile.is_absolute() or ".." in dockerfile.parts:
            raise ValueError("workload.source.dockerfile must be relative to workload.source.context")
        resolved = (context / dockerfile).resolve()
        if not resolved.is_relative_to(context) or not resolved.is_file():
            raise ValueError("workload.source.dockerfile must name a file inside workload.source.context")
        return {"context": context, "dockerfile": dockerfile}
    _exact(source, {"type"}, "workload.source", required=("type",))
    return {}


def _validate_mode(mode, value, cpu, gpu, config_path, project_config):
    if mode is CCDeploymentMode.BARE_METAL_CVM:
        settings = _exact(
            value,
            {"cvm_image", "storage", "network", "user_config", "user_data", "hosts_entries"},
            "bare_metal_cvm",
            required=("cvm_image", "storage", "network"),
        )
        _nonempty_string(settings["cvm_image"], "bare_metal_cvm.cvm_image")
        storage = _exact(
            settings["storage"],
            {"vault_size_gib", "applog_size_gib", "user_config_size_gib", "user_data_size_gib"},
            "bare_metal_cvm.storage",
        )
        for field, default in (
            ("vault_size_gib", 8),
            ("applog_size_gib", 1),
            ("user_config_size_gib", 1),
            ("user_data_size_gib", 1),
        ):
            storage.setdefault(field, default)
            _positive_int(storage[field], f"bare_metal_cvm.storage.{field}")
        network = _exact(
            settings["network"],
            {"allowed_in_ports", "allowed_out_ports", "allowed_in_cidrs", "allowed_out_cidrs"},
            "bare_metal_cvm.network",
        )
        for field in ("allowed_in_ports", "allowed_out_ports"):
            values = network.setdefault(field, [])
            if not isinstance(values, list) or any(type(port) is not int or not 0 < port < 65536 for port in values):
                raise ValueError(f"bare_metal_cvm.network.{field} must contain ports from 1 to 65535")
        for field in ("allowed_in_cidrs", "allowed_out_cidrs"):
            values = network.setdefault(field, [])
            if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"bare_metal_cvm.network.{field} must contain CIDR strings")
            for item in values:
                try:
                    ipaddress.ip_network(item, strict=True)
                except ValueError:
                    raise ValueError(f"bare_metal_cvm.network.{field} contains an invalid CIDR: {item}") from None
        for field in ("user_config", "user_data"):
            if field in settings:
                settings[field] = _path(config_path, settings[field], f"bare_metal_cvm.{field}", directory=True)
        if "hosts_entries" in settings and not isinstance(settings["hosts_entries"], dict):
            raise ValueError("bare_metal_cvm.hosts_entries must be a mapping")
        tools = project_config.get("build_tools", {}).get(mode.value)
        if not tools:
            raise ValueError("bare_metal_cvm requires build_tools.bare_metal_cvm in cc_project.yml")
        if not project_config.get("approval"):
            raise ValueError("bare_metal_cvm requires approval.public_key_files in cc_project.yml")
        settings["storage"] = storage
        settings["network"] = network
        return settings
    if mode is CCDeploymentMode.COCO:
        settings = _exact(
            value,
            {"release_name", "registry", "registry_repository", "platform_config_file"},
            "coco",
            required=("release_name", "registry", "registry_repository", "platform_config_file"),
        )
        release = _nonempty_string(settings["release_name"], "coco.release_name")
        if len(release) > 63 or not DNS_LABEL.fullmatch(release):
            raise ValueError("coco.release_name must be a lowercase DNS label")
        registry = _nonempty_string(settings["registry"], "coco.registry")
        if registry not in project_config.get("container_registries", {}):
            raise ValueError(f"Unknown coco.registry: {registry}")
        repository = _nonempty_string(settings["registry_repository"], "coco.registry_repository")
        if not REPOSITORY.fullmatch(repository):
            raise ValueError("Invalid coco.registry_repository")
        settings["platform_config_file"] = _path(
            config_path, settings["platform_config_file"], "coco.platform_config_file"
        )
        if settings["platform_config_file"].name != "platform.env":
            raise ValueError("coco.platform_config_file must point to platform.env")
        if not project_config.get("build_tools", {}).get(mode.value):
            raise ValueError("coco requires build_tools.coco in cc_project.yml")
        return settings
    settings = _exact(value, {"deployment_target"}, "azure_cc", required=("deployment_target",))
    if settings["deployment_target"] not in ("confidential_vm", "confidential_container"):
        raise ValueError("azure_cc.deployment_target must be confidential_vm or confidential_container")
    if cpu is not CPUTEE.AMD_SEV_SNP or gpu is not GPUTEE.NONE:
        raise ValueError("azure_cc currently supports only cpu_tee: amd_sev_snp with gpu_tee: none")
    return settings
