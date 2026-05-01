#!/usr/bin/env python3
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

"""Multicloud NVFlare deploy/destroy tool.

Usage:
    python deploy.py up     # deploy server + clients
    python deploy.py down   # tear everything down
    python deploy.py status # show current state
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from clouds import CLOUD_ORDER, get_provider

DRY_RUN = False

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
WORK_DIR = TOOL_DIR / ".work"
PROJECT_TEMPLATE = TOOL_DIR / "project.yml"
DEFAULT_KUBECONFIG_DIR = REPO_ROOT / ".tmp" / "kubeconfigs"
K8S_RUNTIME = "k8s"
VALID_ROLES = {"server", "client"}


# ---------------------------------------------------------------------------
# Participant config
# ---------------------------------------------------------------------------
@dataclass
class Participant:
    name: str
    namespace: str
    kubeconfig: str
    role: str  # "server" or "client"
    cloud: str
    prepare: dict
    helm_overrides: list = field(default_factory=list)
    pvc_config: dict = field(default_factory=dict)
    pod_annotations: dict = field(default_factory=dict)
    study_data: dict = field(default_factory=dict)
    # Kubernetes imagePullPolicy override. `None` keeps whatever the helm chart
    # defaults to (typically IfNotPresent). Set to "Always" on mutable dev tags.
    pull_policy: str | None = None


@dataclass
class DeployConfig:
    name: str
    participants: list[Participant]
    server_cloud: str
    gcp_project: str | None
    gcp_region: str | None
    aws_region: str | None
    aws_eks_cluster_name: str | None
    azure_resource_group: str | None
    azure_location: str | None


def _normalize_study_data(study_data: dict | None) -> dict:
    result = {}
    for study, datasets in (study_data or {}).items():
        result[study] = {}
        for dataset, cfg in datasets.items():
            pvc = cfg.get("pvc")
            mode = cfg.get("mode")
            if not pvc or not mode:
                raise ValueError(f"study_data entry '{study}/{dataset}' must define pvc and mode.")
            result[study][dataset] = {"source": pvc, "mode": mode}
    return result


def _kubeconfig_path(cloud: str) -> Path:
    return (DEFAULT_KUBECONFIG_DIR / f"{cloud}.yaml").resolve()


def load_config(config_path: Path) -> DeployConfig:
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text())
    name = raw.get("name") or config_path.stem
    clouds = raw.get("clouds") or {}
    if not clouds:
        raise ValueError(f"{config_path}: missing 'clouds' section")
    raw_participants = raw.get("participants") or []
    if not raw_participants:
        raise ValueError(f"{config_path}: missing 'participants' section")

    cloud_derived: dict[str, dict] = {}
    for cloud_name, cloud_cfg in clouds.items():
        kc_path = _kubeconfig_path(cloud_name)
        if kc_path.exists():
            cloud_derived[cloud_name] = get_provider(cloud_name).parse_kubeconfig(kc_path)

    participants: list[Participant] = []
    servers: list[str] = []
    for entry in raw_participants:
        for key in ("name", "cloud", "namespace", "role"):
            if key not in entry:
                raise ValueError(f"{config_path}: participant missing '{key}': {entry}")
        cloud = entry["cloud"]
        if cloud not in clouds:
            raise ValueError(f"{config_path}: participant {entry['name']} references unknown cloud '{cloud}'")
        role = entry["role"]
        if role not in VALID_ROLES:
            raise ValueError(f"{config_path}: participant {entry['name']} has invalid role '{role}'")
        if role == "server":
            servers.append(entry["name"])

        cloud_cfg = clouds[cloud] or {}
        merged = {**cloud_cfg, **entry}
        prepare = copy.deepcopy(entry.get("prepare") or cloud_cfg.get("prepare") or {})
        if not prepare:
            raise ValueError(f"{config_path}: participant {entry['name']} has no prepare config")
        if prepare.get("runtime") != K8S_RUNTIME:
            raise ValueError(f"{config_path}: participant {entry['name']} prepare.runtime must be '{K8S_RUNTIME}'")
        kc_path = _kubeconfig_path(cloud)
        pvc_config = merged.get("pvc_config") or {}
        study_data = _normalize_study_data(merged.get("study_data"))

        participants.append(
            Participant(
                name=merged["name"],
                namespace=merged["namespace"],
                kubeconfig=str(kc_path),
                role=role,
                cloud=cloud,
                prepare=prepare,
                helm_overrides=list(merged.get("helm_overrides") or []),
                pvc_config=pvc_config,
                pod_annotations=merged.get("pod_annotations") or {},
                study_data=study_data,
                pull_policy=merged.get("pull_policy"),
            )
        )

    if len(servers) != 1:
        raise ValueError(f"{config_path}: expected exactly one server participant, found {len(servers)}: {servers}")
    server_cloud = next(p.cloud for p in participants if p.role == "server")

    gcp = cloud_derived.get("gcp", {})
    aws = cloud_derived.get("aws", {})
    azure = clouds.get("azure") or {}
    return DeployConfig(
        name=name,
        participants=participants,
        server_cloud=server_cloud,
        gcp_project=gcp.get("project"),
        gcp_region=gcp.get("region"),
        aws_region=aws.get("region"),
        aws_eks_cluster_name=aws.get("eks_cluster_name"),
        azure_resource_group=azure.get("resource_group"),
        azure_location=azure.get("location"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class FakeProc:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _dry_run_stdout(cmd: list[str]) -> str:
    if cmd[:4] == ["gcloud", "config", "get-value", "project"]:
        return "<gcp-project>\n"
    if cmd[:4] == ["gcloud", "compute", "addresses", "describe"]:
        return "<server-ip>\n"
    if cmd[:3] == ["aws", "ec2", "describe-addresses"]:
        return json.dumps([{"PublicIp": "<server-ip>", "AllocationId": "<alloc-id>"}]) + "\n"
    if cmd[:4] == ["aws", "configure", "get", "region"]:
        return "<aws-region>\n"
    if cmd[:3] == ["aws", "ec2", "allocate-address"]:
        return json.dumps({"PublicIp": "<server-ip>", "AllocationId": "<alloc-id>"}) + "\n"
    if cmd[:3] == ["aws", "eks", "describe-cluster"]:
        return "<vpc-id>\n"
    if cmd[:3] == ["aws", "ec2", "describe-subnets"]:
        return "<subnet-id>\n"
    if cmd[:4] == ["az", "network", "public-ip", "show"]:
        return "<server-ip>\n"
    return ""


def _mask(text: str) -> str:
    return text.replace(str(REPO_ROOT), "<repo>")


def _print_cmd(cmd: list[str], tag: str = "$"):
    if DRY_RUN:
        print(f"  {tag} {_mask(shlex.join(cmd))}")
    else:
        print(f"  {tag} {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")


def run(cmd: list[str], check=True, capture=False, **kwargs) -> subprocess.CompletedProcess:
    _print_cmd(cmd)
    if DRY_RUN:
        stdin = kwargs.get("input")
        if stdin:
            for line in stdin.splitlines():
                print(f"    | {_mask(line)}")
        return FakeProc(0, stdout=_dry_run_stdout(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, **kwargs)


def run_quiet(cmd: list[str]) -> subprocess.CompletedProcess:
    if DRY_RUN:
        _print_cmd(cmd, tag="?")
        # Auth checks pass; existence checks miss so the "create from scratch" path runs.
        if (
            (cmd[0] == "gcloud" and "auth" in cmd)
            or (cmd[0] == "aws" and "sts" in cmd)
            or (cmd[0] == "az" and "account" in cmd)
        ):
            return FakeProc(0)
        return FakeProc(1)
    return subprocess.run(cmd, capture_output=True, text=True)


def kubectl(kubeconfig: str, *args) -> subprocess.CompletedProcess:
    return run(["kubectl", "--kubeconfig", kubeconfig] + list(args))


def helm(kubeconfig: str, *args) -> subprocess.CompletedProcess:
    return run(["helm", "--kubeconfig", kubeconfig] + list(args))


def check_auth(clouds_used: set):
    print("Checking cloud auth ...")
    for cloud in CLOUD_ORDER:
        if cloud not in clouds_used:
            continue
        provider = get_provider(cloud)
        r = run_quiet(provider.auth_check_cmd)
        if r.returncode != 0:
            sys.exit(provider.auth_failed_message)
    print("  Auth OK")


def check_auth_for(cloud: str):
    """Re-check auth before cloud-specific operations."""
    provider = get_provider(cloud)
    r = run_quiet(provider.auth_check_cmd)
    if r.returncode != 0:
        sys.exit(provider.auth_expired_message)


def _safe_resource_name(name: str) -> str:
    base = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not base:
        base = "cluster"
    full = f"nvflare-{base}"
    if len(full) <= 63:
        return full
    digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:8]
    return f"{full[:54].rstrip('-')}-{digest}"


def _participant_state(p: Participant) -> dict:
    return {"kubeconfig": p.kubeconfig, "namespace": p.namespace, "cloud": p.cloud, "role": p.role}


def deployment_state(config: DeployConfig, *, gcp_project: str | None = None) -> dict:
    ip_name = _safe_resource_name(config.name)
    return {
        "ip_name": ip_name,
        "gcp_project": gcp_project or config.gcp_project,
        "gcp_region": config.gcp_region or "us-central1",
        "aws_region": config.aws_region,
        "server_cloud": config.server_cloud,
        "azure_resource_group": config.azure_resource_group,
        "azure_location": config.azure_location,
        "azure_pip_name": ip_name if config.server_cloud == "azure" else None,
        "participants": {p.name: _participant_state(p) for p in config.participants},
    }


def namespace_exists(kubeconfig: str, ns: str) -> bool:
    return run_quiet(["kubectl", "--kubeconfig", kubeconfig, "get", "ns", ns]).returncode == 0


def helm_release_exists(kubeconfig: str, name: str, ns: str) -> bool:
    return run_quiet(["helm", "--kubeconfig", kubeconfig, "status", name, "-n", ns]).returncode == 0


def nvflare_cmd() -> str:
    if DRY_RUN:
        return "nvflare"
    nvflare = REPO_ROOT / ".venv" / "bin" / "nvflare"
    return str(nvflare) if nvflare.exists() else "nvflare"


# ---------------------------------------------------------------------------
# Provision
# ---------------------------------------------------------------------------
def provision(server_ip: str, config: DeployConfig) -> Path:
    print("Provisioning ...")
    if DRY_RUN:
        project_file = WORK_DIR / "project.yml"
        provision_dir = WORK_DIR / "provision"
        run(["nvflare", "provision", "-p", str(project_file), "-w", str(provision_dir)])
        return Path("<prod_dir>")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    project_text = PROJECT_TEMPLATE.read_text()
    project_text = project_text.replace("__SERVER_IP__", server_ip)
    project = yaml.safe_load(project_text)

    template_server = next(e for e in project["participants"] if e.get("type") == "server")
    template_admin = next(e for e in project["participants"] if e.get("type") == "admin")

    new_participants = []
    for p in config.participants:
        if p.role == "server":
            new_participants.append({**template_server, "name": p.name})
        else:
            new_participants.append({"name": p.name, "type": "client", "org": "nvidia"})
    new_participants.append(template_admin)
    project["participants"] = new_participants

    project_file = WORK_DIR / "project.yml"
    project_file.write_text(yaml.dump(project, default_flow_style=False, sort_keys=False))

    provision_dir = WORK_DIR / "provision"
    if provision_dir.exists():
        shutil.rmtree(provision_dir)

    run([nvflare_cmd(), "provision", "-p", str(project_file), "-w", str(provision_dir)])

    prod_dirs = sorted(provision_dir.glob("*/prod_*"))
    if not prod_dirs:
        sys.exit("Provisioning produced no output")
    return prod_dirs[-1]


def _write_yaml_file(path: Path, data: dict):
    payload = yaml.safe_dump(data, sort_keys=False)
    if DRY_RUN:
        print(f"  Would write {_mask(str(path))}:")
        for line in payload.splitlines():
            print(f"    | {line}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def prepare_runtime_kits(prod_dir: Path, participants: list[Participant]) -> dict[str, Path]:
    print("Preparing K8s runtime kits ...")
    config_dir = WORK_DIR / "prepare-configs"
    if not DRY_RUN and config_dir.exists():
        shutil.rmtree(config_dir)

    kit_dirs = {}
    for p in participants:
        kit_dir = prod_dir / p.name
        prepare_config = copy.deepcopy(p.prepare)
        prepare_config["namespace"] = p.namespace
        if p.study_data:
            _write_yaml_file(kit_dir / "local" / "study_data.yaml", p.study_data)

        config_path = config_dir / f"{p.name}.yaml"
        output_dir = kit_dir / "prepared" / K8S_RUNTIME
        _write_yaml_file(config_path, prepare_config)
        run(
            [
                nvflare_cmd(),
                "deploy",
                "prepare",
                "--kit",
                str(kit_dir),
                "--output",
                str(output_dir),
                "--config",
                str(config_path),
            ]
        )
        kit_dirs[p.name] = output_dir
    return kit_dirs


# ---------------------------------------------------------------------------
# Deploy one participant
# ---------------------------------------------------------------------------
def deploy_participant(
    p: Participant,
    kit_dir: Path,
    server_ip: str | None = None,
    state: dict | None = None,
):
    chart_dir = kit_dir / "helm_chart"

    print(f"\n{'=' * 60}")
    print(f"Deploying {p.name} → {p.namespace}")
    print(f"{'=' * 60}")

    check_auth_for(p.cloud)

    # 1. Namespace
    if not namespace_exists(p.kubeconfig, p.namespace):
        print(f"  Creating namespace {p.namespace}")
        kubectl(p.kubeconfig, "create", "ns", p.namespace)
    else:
        print(f"  Namespace {p.namespace} exists")

    # 2. PVCs
    print("  Applying PVCs ...")
    for pvc_name, cfg in p.pvc_config.items():
        pvc_yaml = json.dumps(
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": pvc_name, "namespace": p.namespace},
                "spec": {
                    "storageClassName": cfg["sc"],
                    "accessModes": [cfg["access"]],
                    "resources": {"requests": {"storage": cfg["size"]}},
                },
            }
        )
        run(["kubectl", "--kubeconfig", p.kubeconfig, "-n", p.namespace, "apply", "-f", "-"], input=pvc_yaml)

    # 3. Remove any running participant so the fresh kit is the only source of truth.
    if helm_release_exists(p.kubeconfig, p.name, p.namespace):
        print(f"  Removing existing Helm release {p.name} ...")
        helm(p.kubeconfig, "uninstall", p.name, "-n", p.namespace, "--wait", "--timeout=600s")

    # 4. Stage kit files
    print("  Staging kit files via temp pod ...")
    pod_name = f"kit-copy-{p.name}"

    parent_prepare = p.prepare.get("parent") or {}
    sec_ctx = {}
    security_context = parent_prepare.get("pod_security_context")
    if security_context:
        sec_ctx = {"securityContext": security_context}
    workspace_pvc = parent_prepare.get("workspace_pvc", "nvflws")

    copy_image = "busybox:1.36"
    pod_spec = {
        "spec": {
            **sec_ctx,
            "volumes": [{"name": "ws", "persistentVolumeClaim": {"claimName": workspace_pvc}}],
            "containers": [
                {
                    "name": "copy",
                    "image": copy_image,
                    "command": ["sleep", "600"],
                    "volumeMounts": [{"name": "ws", "mountPath": "/ws"}],
                    "resources": {
                        "requests": {"cpu": "10m", "memory": "64Mi"},
                        "limits": {"cpu": "100m", "memory": "128Mi"},
                    },
                }
            ],
        }
    }

    kubectl(p.kubeconfig, "-n", p.namespace, "delete", "pod", pod_name, "--ignore-not-found", "--timeout=60s")
    kubectl(
        p.kubeconfig,
        "-n",
        p.namespace,
        "run",
        pod_name,
        f"--image={copy_image}",
        "--restart=Never",
        f"--overrides={json.dumps(pod_spec)}",
    )

    print("  Waiting for staging pod ...")
    kubectl(p.kubeconfig, "-n", p.namespace, "wait", "--for=condition=Ready", f"pod/{pod_name}", "--timeout=600s")

    kubectl(p.kubeconfig, "-n", p.namespace, "exec", pod_name, "--", "rm", "-rf", "/ws/startup", "/ws/local")
    kubectl(p.kubeconfig, "-n", p.namespace, "cp", str(kit_dir / "startup"), f"{pod_name}:/ws/startup")
    kubectl(p.kubeconfig, "-n", p.namespace, "cp", str(kit_dir / "local"), f"{pod_name}:/ws/local")
    kubectl(p.kubeconfig, "-n", p.namespace, "delete", "pod", pod_name, "--timeout=60s")

    # 5. Helm install
    print(f"  Helm installing {p.name} ...")
    helm_args = ["upgrade", "--install", p.name, str(chart_dir), "-n", p.namespace]
    helm_args += p.helm_overrides

    if p.role == "server" and server_ip:
        helm_args += ["--set", "service.type=LoadBalancer"]
        helm_args += get_provider(p.cloud).server_service_helm_args(server_ip=server_ip, state=state or {})

    for k, v in p.pod_annotations.items():
        escaped = k.replace(".", r"\.").replace("/", r"\/")
        escaped_v = str(v).replace("\\", r"\\").replace(",", r"\,").replace("=", r"\=")
        helm_args += ["--set-string", f"podAnnotations.{escaped}={escaped_v}"]

    if p.pull_policy:
        helm_args += ["--set", f"image.pullPolicy={p.pull_policy}"]

    helm(p.kubeconfig, *helm_args)

    # 6. Wait for server deployment
    if p.role == "server":
        print("  Waiting for server to be ready ...")
        kubectl(
            p.kubeconfig,
            "-n",
            p.namespace,
            "wait",
            "--for=condition=available",
            f"deployment/{p.name}",
            "--timeout=600s",
        )


def deploy_participants(participants: list[Participant], kit_dirs: dict[str, Path], server_ip: str, state: dict):
    servers = [p for p in participants if p.role == "server"]
    clients = [p for p in participants if p.role != "server"]

    for server in servers:
        deploy_participant(server, kit_dirs[server.name], server_ip=server_ip, state=state)

    if DRY_RUN or len(clients) <= 1:
        for client in clients:
            deploy_participant(client, kit_dirs[client.name], server_ip=server_ip, state=state)
        return

    print(f"\nDeploying {len(clients)} clients in parallel ...")
    errors = []
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        future_to_participant = {
            executor.submit(deploy_participant, client, kit_dirs[client.name], server_ip=server_ip, state=state): client
            for client in clients
        }
        for future in as_completed(future_to_participant):
            participant = future_to_participant[future]
            try:
                future.result()
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                errors.append((participant.name, e))

    if errors:
        for name, error in errors:
            print(f"  {name} failed: {error}")
        raise RuntimeError("Client deploy failed for: " + ", ".join(name for name, _ in errors))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_up(args):
    config = load_config(Path(args.config))
    participants = config.participants
    clouds_used = {p.cloud for p in participants}
    check_auth(clouds_used)

    server_provider = get_provider(config.server_cloud)
    server_provider.validate_server_config(config)

    gcp_project = None
    if "gcp" in clouds_used:
        gcp_project = (
            config.gcp_project or run(["gcloud", "config", "get-value", "project"], capture=True).stdout.strip()
        )
    gcp_region = config.gcp_region or "us-central1"
    aws_region = config.aws_region

    state = deployment_state(config, gcp_project=gcp_project)
    server_ip, ip_name = server_provider.reserve_ip(
        run=run,
        ip_tag=state["ip_name"],
        state=state,
        gcp_project=gcp_project,
        gcp_region=gcp_region,
        aws_region=aws_region,
        azure_resource_group=config.azure_resource_group,
        azure_location=config.azure_location,
    )

    server_provider.prepare_server_state(run=run, state=state, config=config, ip_name=ip_name, aws_region=aws_region)
    state["server_ip"] = server_ip

    prod_dir = provision(server_ip, config)
    kit_dirs = prepare_runtime_kits(prod_dir, participants)

    deploy_participants(participants, kit_dirs, server_ip=server_ip, state=state)

    print(f"\n{'=' * 60}")
    print("Deployment complete.")
    print(f"  Server IP:   {server_ip}")
    print(f"  Admin kit:   {prod_dir / 'admin@nvidia.com'}")
    print(f"{'=' * 60}")


def teardown_participant(name: str, info: dict) -> bool:
    kc, ns, cloud = info["kubeconfig"], info["namespace"], info.get("cloud", "")
    print(f"Tearing down {name} ({ns}) ...")
    try:
        check_auth_for(cloud)
    except SystemExit:
        print(f"  Auth failed for {kc} — skipping")
        return False

    run(["helm", "--kubeconfig", kc, "uninstall", name, "-n", ns], check=False)
    r = run(["kubectl", "--kubeconfig", kc, "delete", "ns", ns, "--ignore-not-found", "--timeout=120s"], check=False)
    if r.returncode != 0:
        return False

    return True


def teardown_participants(items: list[tuple[str, dict]], parallel: bool = True) -> bool:
    if not items:
        return True

    if DRY_RUN or not parallel or len(items) <= 1:
        ok = True
        for name, info in items:
            ok = teardown_participant(name, info) and ok
        return ok

    errors = []
    with ThreadPoolExecutor(max_workers=len(items)) as executor:
        future_to_name = {executor.submit(teardown_participant, name, info): name for name, info in items}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                if not future.result():
                    errors.append(name)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                print(f"  {name} failed: {e}")
                errors.append(name)

    return not errors


def cmd_down(args):
    config = load_config(Path(args.config))
    state = deployment_state(config)
    participants = state.get("participants", {})
    participant_items = list(participants.items())
    client_items = [(name, info) for name, info in participant_items if info["role"] != "server"]
    server_items = [(name, info) for name, info in participant_items if info["role"] == "server"]
    clients_ok = teardown_participants(client_items)
    server_ok = teardown_participants(server_items)

    if not clients_ok or not server_ok:
        print("Partial teardown. Re-run down with the same config after fixing the failure.")
        sys.exit(1)

    get_provider(state.get("server_cloud", "gcp")).release_ip(run=run, ip_name=state["ip_name"], state=state)
    print("Destroyed.")


def cmd_status(args):
    config = load_config(Path(args.config))
    state = deployment_state(config)

    print(f"IP name:   {state.get('ip_name', 'N/A')}")
    for name, info in state.get("participants", {}).items():
        kc, ns = info["kubeconfig"], info["namespace"]
        print(f"\n{name} ({ns}):")
        r = run_quiet(["kubectl", "--kubeconfig", kc, "-n", ns, "get", "pods", "--no-headers"])
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().split("\n"):
                print(f"  {line}")
        else:
            print("  No pods")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    default_config = str(TOOL_DIR / "all-clouds.yaml")
    parser = argparse.ArgumentParser(description="Multicloud NVFlare deploy tool")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--config", default=default_config, help="Path to deploy config YAML")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("up", help="Deploy server + clients")

    sub.add_parser("down", help="Tear down everything")
    sub.add_parser("status", help="Show deployment status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    global DRY_RUN
    DRY_RUN = args.dry_run

    {"up": cmd_up, "down": cmd_down, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
