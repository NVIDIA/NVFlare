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

import argparse
import json
from datetime import datetime, timedelta

import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.tool.deploy.deploy_commands import (
    GPU_RESOURCE_CONSUMER,
    GPU_RESOURCE_MANAGER,
    HELM_RELEASE_NAME_MAX_LENGTH,
    PROCESS_CLIENT_LAUNCHER,
    _k8s_release_name,
    prepare_deployment,
)


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2))


def _write_cert(path, common_name="site-1", org="nvidia"):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
        ]
    )
    now = datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _make_client_kit(tmp_path, name="site-1"):
    kit = tmp_path / name
    startup = kit / "startup"
    local = kit / "local"
    startup.mkdir(parents=True)
    local.mkdir()
    _write_json(
        startup / "fed_client.json",
        {
            "format_version": 2,
            "servers": [{"name": "project", "service": {"scheme": "tcp"}, "identity": "server"}],
            "client": {
                "ssl_private_key": "client.key",
                "ssl_cert": "client.crt",
                "ssl_root_cert": "rootCA.pem",
                "fqsn": name,
                "is_leaf": True,
                "connection_security": "mtls",
            },
        },
    )
    _write_cert(startup / "client.crt", common_name=name)
    (startup / "client.key").write_text("key")
    (startup / "rootCA.pem").write_text("ca")
    (startup / "sub_start.sh").write_text(
        "python3 -m nvflare.private.fed.app.client.client_train --set uid=site-1 org=nvidia config_folder=config\n"
    )
    (startup / "start.sh").write_text("#!/usr/bin/env bash\n./sub_start.sh &\n")
    (startup / "stop_fl.sh").write_text("#!/usr/bin/env bash\ntouch ../shutdown.fl\n")
    (startup / "docker.sh").write_text("#!/usr/bin/env bash\ndocker run legacy-image\n")
    _write_json(
        local / "resources.json.default",
        {
            "format_version": 2,
            "components": [
                {"id": "resource_manager", "path": "old.ResourceManager", "args": {}},
                {"id": "resource_consumer", "path": "old.ResourceConsumer", "args": {}},
                {"id": "process_launcher", "path": "old.ProcessLauncher", "args": {}},
            ],
        },
    )
    _write_json(
        local / "resources.json",
        {
            "format_version": 2,
            "components": [
                {"id": "process_launcher", "path": "stale.ProcessLauncher", "args": {}},
            ],
        },
    )
    _write_json(
        local / "comm_config.json",
        {
            "allow_adhoc_conns": False,
            "backbone_conn_gen": 2,
            "internal": {
                "scheme": "tcp",
                "resources": {"host": "localhost", "port": 8102, "connection_security": "clear"},
            },
        },
    )
    return kit


def _make_server_kit(tmp_path, fed_learn_port=8002, admin_port=8003, name="server"):
    kit = tmp_path / name
    startup = kit / "startup"
    local = kit / "local"
    startup.mkdir(parents=True)
    local.mkdir()
    _write_json(
        startup / "fed_server.json",
        {
            "format_version": 2,
            "servers": [
                {
                    "identity": name,
                    "service": {"scheme": "tcp", "target": f"{name}:{fed_learn_port}"},
                    "admin_port": admin_port,
                }
            ],
        },
    )
    _write_cert(startup / "server.crt", common_name=name)
    (startup / "server.key").write_text("key")
    (startup / "rootCA.pem").write_text("ca")
    (startup / "start.sh").write_text("#!/usr/bin/env bash\n./sub_start.sh &\n")
    (startup / "sub_start.sh").write_text(
        "python3 -m nvflare.private.fed.app.server.server_train --set org=nvidia config_folder=config\n"
    )
    _write_json(
        local / "resources.json.default",
        {
            "format_version": 2,
            "components": [
                {"id": "process_launcher", "path": "old.ProcessLauncher", "args": {}},
            ],
        },
    )
    _write_json(
        local / "comm_config.json",
        {
            "allow_adhoc_conns": False,
            "backbone_conn_gen": 2,
            "internal": {
                "scheme": "tcp",
                "resources": {"host": "localhost", "port": 8102, "connection_security": "clear"},
            },
        },
    )
    return kit


def _run_prepare(kit, output, config):
    config_path = output.parent / f"{output.name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    prepare_deployment(argparse.Namespace(kit=str(kit), output=str(output), config=str(config_path)))


def _component(resources, component_id):
    return next(c for c in resources["components"] if c["id"] == component_id)


def _add_server_storage(resources_path, snapshot_persistor=None):
    resources = json.loads(resources_path.read_text())
    resources["snapshot_persistor"] = snapshot_persistor or {
        "path": "nvflare.app_common.state_persistors.storage_state_persistor.StorageStatePersistor",
        "args": {
            "uri_root": "/",
            "storage": {
                "path": "nvflare.app_common.storages.filesystem_storage.FilesystemStorage",
                "args": {"root_dir": "/tmp/nvflare/snapshot-storage", "uri_root": "/"},
            },
        },
    }
    resources["components"].append(
        {
            "id": "job_manager",
            "path": "nvflare.apis.impl.job_def_manager.SimpleJobDefManager",
            "args": {"uri_root": "/tmp/nvflare/jobs-storage", "job_store_id": "job_store"},
        }
    )
    _write_json(resources_path, resources)


def test_prepare_docker_client_copies_and_patches_runtime_files(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-docker"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "docker",
            "parent": {"docker_image": "repo/nvflare:dev", "network": "nvflare-test"},
            "job_launcher": {
                "default_python_path": "/usr/bin/python",
                "default_job_env": {"NCCL_P2P_DISABLE": "1"},
                "default_job_container_kwargs": {"shm_size": "8g"},
            },
        },
    )
    capsys.readouterr()

    assert not (kit / "startup" / "start_docker.sh").exists()
    assert (kit / "startup" / "start.sh").exists()
    assert (kit / "startup" / "stop_fl.sh").exists()
    assert (kit / "startup" / "docker.sh").exists()
    assert (output / "startup" / "start_docker.sh").exists()
    assert not (output / "startup" / "start.sh").exists()
    assert not (output / "startup" / "sub_start.sh").exists()
    assert not (output / "startup" / "stop_fl.sh").exists()
    assert not (output / "startup" / "docker.sh").exists()
    script = (output / "startup" / "start_docker.sh").read_text()
    assert "repo/nvflare:dev" in script
    assert 'NETWORK_NAME="nvflare-test"' in script
    assert "--network-alias" not in script
    assert "/var/tmp/nvflare/workspace/startup/sub_start.sh" not in script
    assert "/usr/local/bin/python3" in script
    assert "nvflare.private.fed.app.client.client_train" in script
    assert "-m \\\n    /var/tmp/nvflare/workspace" in script
    assert "fed_client.json" in script
    assert "uid=site-1" in script
    assert "org=nvidia" in script

    resources = json.loads((output / "local" / "resources.json.default").read_text())
    assert (kit / "local" / "resources.json").exists()
    assert not (output / "local" / "resources.json").exists()
    component_ids = [c["id"] for c in resources["components"]]
    assert "process_launcher" not in component_ids
    assert "resource_consumer" not in component_ids
    assert _component(resources, "resource_manager")["path"].endswith("PassthroughResourceManager")
    launcher = _component(resources, "docker_launcher")
    assert launcher["path"] == "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher"
    assert launcher["args"]["network"] == "nvflare-test"
    assert launcher["args"]["default_python_path"] == "/usr/bin/python"
    assert launcher["args"]["default_job_env"] == {"NCCL_P2P_DISABLE": "1"}
    assert launcher["args"]["default_job_container_kwargs"] == {"shm_size": "8g"}

    comm_config = json.loads((output / "local" / "comm_config.json").read_text())
    assert comm_config["internal"]["resources"]["host"] == "0.0.0.0"
    assert (output / "local" / "study_data.yaml").exists()


@pytest.mark.parametrize(
    "admin_port, expected_admin_publish_count",
    [
        (8002, 0),
        (8003, 1),
    ],
)
def test_prepare_docker_server_publishes_admin_port_only_when_distinct(
    tmp_path, capsys, admin_port, expected_admin_publish_count
):
    kit = _make_server_kit(tmp_path, fed_learn_port=8002, admin_port=admin_port)
    output = tmp_path / "server-docker"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "docker",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    script = (output / "startup" / "start_docker.sh").read_text()
    assert script.count("-p 8002:8002") == 1
    assert script.count("-p 8003:8003") == expected_admin_publish_count


def test_prepare_docker_server_adds_logical_server_network_alias(tmp_path, capsys):
    kit = _make_server_kit(tmp_path, name="abc.aws.com")
    output = tmp_path / "server-docker"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "docker",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    script = (output / "startup" / "start_docker.sh").read_text()
    assert "--name abc.aws.com" in script
    assert "--network-alias server" in script


def test_prepare_docker_server_relocates_storage_to_mounted_workspace(tmp_path, capsys):
    kit = _make_server_kit(tmp_path)
    _add_server_storage(kit / "local" / "resources.json.default")
    output = tmp_path / "server-docker"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "docker",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    captured = capsys.readouterr()

    assert "snapshot_persistor is present" not in captured.out + captured.err
    resources = json.loads((output / "local" / "resources.json.default").read_text())
    assert (
        resources["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"]
        == "/var/tmp/nvflare/workspace/snapshot-storage"
    )
    assert _component(resources, "job_manager")["args"]["uri_root"] == "/var/tmp/nvflare/workspace/jobs-storage"


@pytest.mark.parametrize(
    "admin_port, expected_admin_port",
    [
        (8002, None),
        (8003, 8003),
    ],
)
def test_prepare_k8s_server_exposes_admin_port_only_when_distinct(tmp_path, capsys, admin_port, expected_admin_port):
    kit = _make_server_kit(tmp_path, fed_learn_port=8002, admin_port=admin_port)
    output = tmp_path / "server-k8s"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "k8s",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    values = yaml.safe_load((output / "helm_chart" / "values.yaml").read_text())
    assert values["fedLearnPort"] == 8002
    assert values["adminPort"] == expected_admin_port

    tcp_services = (output / "helm_chart" / "templates" / "server-tcp-services.yaml").read_text()
    assert ".Values.fedLearnPort" in tcp_services
    assert ".Values.adminPort" in tcp_services


@pytest.mark.parametrize("runtime", ["docker", "k8s"])
def test_prepare_server_without_snapshot_persistor_is_silent(tmp_path, capsys, runtime):
    kit = _make_server_kit(tmp_path)
    output = tmp_path / f"server-{runtime}"

    _run_prepare(
        kit,
        output,
        {
            "runtime": runtime,
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    captured = capsys.readouterr()

    assert "snapshot_persistor is present" not in captured.out + captured.err


@pytest.mark.parametrize("runtime", ["docker", "k8s"])
def test_prepare_server_warns_when_snapshot_persistor_shape_is_unexpected(tmp_path, capsys, runtime):
    kit = _make_server_kit(tmp_path)
    _add_server_storage(
        kit / "local" / "resources.json.default",
        snapshot_persistor={
            "path": "custom.SnapshotPersistor",
            "args": {
                "storage": {
                    "path": "custom.Storage",
                }
            },
        },
    )
    output = tmp_path / f"server-{runtime}"

    _run_prepare(
        kit,
        output,
        {
            "runtime": runtime,
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    captured = capsys.readouterr()

    combined_output = captured.out + captured.err
    assert "snapshot_persistor is present" in combined_output
    assert "snapshot_persistor.args.storage.args.root_dir" in combined_output
    resources = json.loads((output / "local" / "resources.json.default").read_text())
    assert "args" not in resources["snapshot_persistor"]["args"]["storage"]
    assert _component(resources, "job_manager")["args"]["uri_root"] == "/var/tmp/nvflare/workspace/jobs-storage"


def test_prepare_docker_reads_org_from_cert_without_sub_start(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    (kit / "startup" / "sub_start.sh").unlink()
    output = tmp_path / "site-1-docker"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "docker",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    script = (output / "startup" / "start_docker.sh").read_text()
    assert "org=nvidia" in script
    assert not (output / "startup" / "sub_start.sh").exists()


def test_prepare_docker_creates_comm_config_when_missing(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    (kit / "local" / "comm_config.json").unlink()
    output = tmp_path / "site-1-docker"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "docker",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    comm_config = json.loads((output / "local" / "comm_config.json").read_text())
    assert comm_config["backbone_conn_gen"] == 2
    assert comm_config["internal"]["scheme"] == "tcp"
    assert comm_config["internal"]["resources"] == {
        "host": "0.0.0.0",
        "connection_security": "clear",
    }


def test_prepare_uses_conventional_config_and_output_paths(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = kit / "prepared" / "docker"
    (kit / "config.yaml").write_text(
        yaml.safe_dump({"runtime": "docker", "parent": {"docker_image": "repo/nvflare:dev"}}, sort_keys=False)
    )

    prepare_deployment(argparse.Namespace(kit=str(kit), kit_flag=None, output=None, config=None))
    capsys.readouterr()

    assert output.exists()
    assert (output / "startup" / "start_docker.sh").exists()

    (output / "stale.txt").write_text("stale")
    prepare_deployment(argparse.Namespace(kit=str(kit), kit_flag=None, output=None, config=None))
    capsys.readouterr()

    assert not (output / "stale.txt").exists()
    assert not (output / "prepared").exists()
    assert (output / "startup" / "start_docker.sh").exists()


def test_prepare_default_output_is_runtime_specific(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    config_path = kit / "config.yaml"

    config_path.write_text(
        yaml.safe_dump({"runtime": "docker", "parent": {"docker_image": "repo/nvflare:dev"}}, sort_keys=False)
    )
    prepare_deployment(argparse.Namespace(kit=str(kit), kit_flag=None, output=None, config=None))
    capsys.readouterr()

    config_path.write_text(
        yaml.safe_dump({"runtime": "k8s", "parent": {"docker_image": "repo/nvflare:dev"}}, sort_keys=False)
    )
    prepare_deployment(argparse.Namespace(kit=str(kit), kit_flag=None, output=None, config=None))
    capsys.readouterr()

    assert (kit / "prepared" / "docker" / "startup" / "start_docker.sh").exists()
    assert (kit / "prepared" / "k8s" / "helm_chart" / "values.yaml").exists()
    assert not (kit / "prepared" / "k8s" / "prepared").exists()


def test_prepare_k8s_client_writes_chart_and_launcher_config(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-k8s"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "k8s",
            "namespace": "flare",
            "parent": {
                "docker_image": "repo/nvflare:dev",
                "parent_port": 9102,
                "workspace_pvc": "nvflws.team.example.com",
                "workspace_mount_path": "/workspace",
                "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
                "pod_security_context": {"runAsUser": 1000},
            },
            "job_launcher": {
                "config_file_path": None,
                "pending_timeout": 7,
                "default_python_path": "/usr/bin/python",
                "job_pod_security_context": {"runAsNonRoot": True},
            },
        },
    )
    capsys.readouterr()

    resources = json.loads((output / "local" / "resources.json.default").read_text())
    assert not (output / "local" / "resources.json").exists()
    launcher = _component(resources, "k8s_launcher")
    assert launcher["path"] == "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
    assert launcher["args"]["namespace"] == "flare"
    assert launcher["args"]["study_data_pvc_file_path"] == "/workspace/local/study_data.yaml"
    assert launcher["args"]["default_python_path"] == "/usr/bin/python"
    assert launcher["args"]["pending_timeout"] == 7
    assert launcher["args"]["security_context"] == {"runAsNonRoot": True}

    comm_config = json.loads((output / "local" / "comm_config.json").read_text())
    assert comm_config["internal"]["resources"] == {
        "host": "site-1",
        "port": 9102,
        "connection_security": "clear",
    }

    values = yaml.safe_load((output / "helm_chart" / "values.yaml").read_text())
    assert not (output / "startup" / "start.sh").exists()
    assert not (output / "startup" / "sub_start.sh").exists()
    assert not (output / "startup" / "stop_fl.sh").exists()
    assert not (output / "startup" / "docker.sh").exists()
    assert not (output / "startup" / "start_docker.sh").exists()
    assert values["name"] == "site-1"
    assert values["siteName"] == "site-1"
    assert values["serviceName"] == "site-1"
    assert values["image"] == {"repository": "repo/nvflare", "tag": "dev", "pullPolicy": "Always"}
    assert values["persistence"]["workspace"]["claimName"] == "nvflws.team.example.com"
    assert values["persistence"]["workspace"]["volumeName"] == "workspace"
    assert values["persistence"]["workspace"]["mountPath"] == "/workspace"
    assert values["port"] == 9102
    assert values["securityContext"] == {"runAsUser": 1000}
    assert values["resources"] == {"requests": {"cpu": "1", "memory": "2Gi"}}
    assert (output / "helm_chart" / "templates" / "client-deployment.yaml").exists()


@pytest.mark.parametrize("namespace", ["nvflare", "1abc", "2026-prod"])
def test_prepare_k8s_accepts_valid_namespace(tmp_path, capsys, namespace):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / f"prepared-{namespace}"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "k8s",
            "namespace": namespace,
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    resources = json.loads((output / "local" / "resources.json.default").read_text())
    launcher = _component(resources, "k8s_launcher")
    assert launcher["args"]["namespace"] == namespace


@pytest.mark.parametrize(
    "namespace",
    ["MyNamespace", "namespace_", "-bad", "bad-", "bad.name", "nvflare\n", "", "a" * 64, None, 7],
)
def test_prepare_k8s_rejects_invalid_namespace(tmp_path, capsys, namespace):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "prepared"

    with pytest.raises(SystemExit):
        _run_prepare(
            kit,
            output,
            {
                "runtime": "k8s",
                "namespace": namespace,
                "parent": {"docker_image": "repo/nvflare:dev"},
            },
        )

    err = capsys.readouterr().err
    assert "INVALID_CONFIG" in err
    assert "k8s config.namespace" in err
    assert not output.exists()


def test_prepare_warns_when_replacing_custom_resource_and_launcher_config(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    resources_path = kit / "local" / "resources.json.default"
    resources = json.loads(resources_path.read_text())
    resources["components"] = [
        {
            "id": "resource_manager",
            "path": "nvflare.app_common.resource_managers.list_resource_manager.ListResourceManager",
            "args": {"resources": [{"gpu": 1}]},
        },
        {
            "id": "resource_consumer",
            "path": "custom.AuditResourceConsumer",
            "args": {},
        },
        {
            "id": "process_launcher",
            "path": "custom.ProcessLauncher",
            "args": {},
        },
        {
            "id": "k8s_launcher",
            "path": "custom.K8sLauncher",
            "args": {"timeout": 99},
        },
    ]
    _write_json(resources_path, resources)
    output = tmp_path / "site-1-k8s"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "k8s",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    captured = capsys.readouterr()

    combined_output = captured.out + captured.err
    assert "replaces component 'resource_manager'" in combined_output
    assert "removes component 'resource_consumer'" in combined_output
    assert "replaces component 'process_launcher'" in combined_output
    assert "replaces component 'k8s_launcher'" in combined_output


def test_prepare_does_not_warn_for_default_components_with_empty_args(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    resources_path = kit / "local" / "resources.json.default"
    resources = json.loads(resources_path.read_text())
    resources["components"] = [
        {"id": "resource_manager", "path": GPU_RESOURCE_MANAGER, "args": {}},
        {"id": "resource_consumer", "path": GPU_RESOURCE_CONSUMER, "args": {}},
        {"id": "process_launcher", "path": PROCESS_CLIENT_LAUNCHER, "args": {}},
    ]
    _write_json(resources_path, resources)
    output = tmp_path / "site-1-k8s"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "k8s",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    captured = capsys.readouterr()

    assert "Warning:" not in captured.out + captured.err


def test_prepare_k8s_client_sanitizes_service_name_without_changing_site_identity(tmp_path, capsys):
    kit = _make_client_kit(tmp_path, name="Site_1")
    output = tmp_path / "site-1-k8s"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "k8s",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    comm_config = json.loads((output / "local" / "comm_config.json").read_text())
    values = yaml.safe_load((output / "helm_chart" / "values.yaml").read_text())
    assert values["name"] == "Site_1"
    assert values["siteName"] == "Site_1"
    assert values["serviceName"] == comm_config["internal"]["resources"]["host"]
    assert values["serviceName"] != "Site_1"
    assert values["serviceName"].startswith("site-1-")
    assert "_" not in values["serviceName"]


def test_prepare_k8s_creates_comm_config_when_missing(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    (kit / "local" / "comm_config.json").unlink()
    output = tmp_path / "site-1-k8s"

    _run_prepare(
        kit,
        output,
        {
            "runtime": "k8s",
            "parent": {"docker_image": "repo/nvflare:dev"},
        },
    )
    capsys.readouterr()

    comm_config = json.loads((output / "local" / "comm_config.json").read_text())
    assert comm_config["backbone_conn_gen"] == 2
    assert comm_config["internal"]["scheme"] == "tcp"
    assert comm_config["internal"]["resources"] == {
        "host": "site-1",
        "port": 8102,
        "connection_security": "clear",
    }


def test_k8s_release_name_is_safe_for_helm():
    safe_names = [
        _k8s_release_name("1-my-site"),
        _k8s_release_name("a" * 80),
        _k8s_release_name("Site_1"),
    ]

    for name in safe_names:
        assert len(name) <= HELM_RELEASE_NAME_MAX_LENGTH
        assert name[0].isalpha()
        assert name[-1].isalnum()
        assert name == name.lower()
        assert all(c.isalnum() or c == "-" for c in name)

    assert _k8s_release_name("site-1") == "site-1"
    assert _k8s_release_name("1-my-site").startswith("site-1-my-site-")
    assert _k8s_release_name("a" * 80) != _k8s_release_name("a" * 79 + "b")


def test_prepare_rejects_admin_kit_without_writing_output(tmp_path, capsys):
    kit = tmp_path / "admin"
    startup = kit / "startup"
    local = kit / "local"
    startup.mkdir(parents=True)
    local.mkdir()
    _write_json(startup / "fed_admin.json", {"admin": {"name": "admin@nvidia.com"}})
    (startup / "client.crt").write_text("crt")
    (startup / "client.key").write_text("key")
    (startup / "rootCA.pem").write_text("ca")
    _write_json(local / "resources.json.default", {"components": []})

    output = tmp_path / "admin-docker"
    config_path = tmp_path / "docker.yaml"
    config_path.write_text(yaml.safe_dump({"runtime": "docker", "parent": {"docker_image": "repo/nvflare:dev"}}))

    with pytest.raises(SystemExit):
        prepare_deployment(argparse.Namespace(kit=str(kit), output=str(output), config=str(config_path)))

    assert "UNSUPPORTED_KIT" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "runtime": "docker",
                "parent": {"docker_image": "repo/nvflare:dev"},
                "job_launcher": {"default_python_path": 7},
            },
            "job_launcher.default_python_path",
        ),
        (
            {"runtime": "k8s", "parent": {"docker_image": "repo/nvflare:dev", "workspace_pvc": 7}},
            "parent.workspace_pvc",
        ),
        (
            {"runtime": "k8s", "parent": {"docker_image": "repo/nvflare:dev", "workspace_mount_path": []}},
            "parent.workspace_mount_path",
        ),
        (
            {
                "runtime": "k8s",
                "parent": {"docker_image": "repo/nvflare:dev"},
                "job_launcher": {"config_file_path": 7},
            },
            "job_launcher.config_file_path",
        ),
        (
            {
                "runtime": "k8s",
                "parent": {"docker_image": "repo/nvflare:dev"},
                "job_launcher": {"default_python_path": False},
            },
            "job_launcher.default_python_path",
        ),
    ],
)
def test_prepare_rejects_non_string_optional_config_values(tmp_path, capsys, config, expected):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "prepared"

    with pytest.raises(SystemExit):
        _run_prepare(kit, output, config)

    err = capsys.readouterr().err
    assert "INVALID_CONFIG" in err
    assert expected in err
    assert not output.exists()
