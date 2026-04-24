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
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DRY_RUN = False

LAUNCHER_FOR_ROLE = {
    "server": "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher",
    "client": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
}

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
WORK_DIR = TOOL_DIR / ".work"
STATE_FILE = WORK_DIR / "state.json"
PROJECT_TEMPLATE = TOOL_DIR / "project.yml"


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
    launcher_class: str
    image: str
    helm_overrides: list = field(default_factory=list)
    security_context: dict | None = None
    pending_timeout: int | None = None
    pvc_config: dict = field(default_factory=dict)
    # Kubernetes imagePullPolicy override. `None` keeps whatever the helm chart
    # defaults to (typically IfNotPresent). Set to "Always" on mutable dev tags.
    pull_policy: str | None = None


@dataclass
class DeployConfig:
    participants: list[Participant]
    server_cloud: str
    gcp_project: str | None
    gcp_region: str | None
    aws_region: str | None
    aws_eks_cluster_name: str | None


def _parse_kubeconfig(kc_path: Path, cloud: str) -> dict:
    """Extract cluster-identifying fields (region, project, cluster name) from a kubeconfig.

    Relies on the default context-name formats produced by:
      * GCP: ``gcloud container clusters get-credentials`` → ``gke_<project>_<region>_<cluster>``
      * AWS: ``aws eks update-kubeconfig`` → ``arn:aws:eks:<region>:<account>:cluster/<name>``
    """
    data = yaml.safe_load(kc_path.read_text())
    current_ctx = data.get("current-context")
    if not current_ctx:
        raise ValueError(f"{kc_path}: no current-context")
    ctx = next((c for c in data.get("contexts", []) if c.get("name") == current_ctx), None)
    if not ctx:
        raise ValueError(f"{kc_path}: current-context {current_ctx!r} not found in contexts")
    cluster_name = ctx["context"]["cluster"]

    if cloud == "gcp":
        parts = cluster_name.split("_")
        if len(parts) < 4 or parts[0] != "gke":
            raise ValueError(
                f"{kc_path}: GKE context cluster {cluster_name!r} not in 'gke_<project>_<region>_<cluster>' form"
            )
        return {"project": parts[1], "region": parts[2]}
    if cloud == "aws":
        if not cluster_name.startswith("arn:aws:eks:"):
            raise ValueError(f"{kc_path}: EKS context cluster {cluster_name!r} is not an ARN")
        parts = cluster_name.split(":")
        if len(parts) < 6:
            raise ValueError(f"{kc_path}: malformed EKS ARN {cluster_name!r}")
        return {"region": parts[3], "eks_cluster_name": parts[5].split("/", 1)[1]}
    # No autoderive for other clouds (e.g. azure); not needed for current operations.
    return {}


def load_config(config_path: Path) -> DeployConfig:
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text())
    clouds = raw.get("clouds") or {}
    if not clouds:
        raise ValueError(f"{config_path}: missing 'clouds' section")
    raw_participants = raw.get("participants") or []
    if not raw_participants:
        raise ValueError(f"{config_path}: missing 'participants' section")

    cloud_derived: dict[str, dict] = {}
    for cloud_name, cloud_cfg in clouds.items():
        kc_rel = (cloud_cfg or {}).get("kubeconfig")
        if not kc_rel:
            continue
        kc_path = (config_path.parent / kc_rel).resolve()
        if kc_path.exists():
            cloud_derived[cloud_name] = _parse_kubeconfig(kc_path, cloud_name)

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
        if role not in LAUNCHER_FOR_ROLE:
            raise ValueError(f"{config_path}: participant {entry['name']} has invalid role '{role}'")
        if role == "server":
            servers.append(entry["name"])

        merged = {**clouds[cloud], **entry}
        kubeconfig = merged.get("kubeconfig")
        if not kubeconfig:
            raise ValueError(f"{config_path}: participant {entry['name']} has no kubeconfig (set on cloud or entry)")
        image = merged.get("image")
        if not image:
            raise ValueError(f"{config_path}: participant {entry['name']} has no image (set on cloud or entry)")
        kc_path = (config_path.parent / kubeconfig).resolve()

        participants.append(
            Participant(
                name=merged["name"],
                namespace=merged["namespace"],
                kubeconfig=str(kc_path),
                role=role,
                cloud=cloud,
                launcher_class=LAUNCHER_FOR_ROLE[role],
                image=image,
                helm_overrides=list(merged.get("helm_overrides") or []),
                security_context=merged.get("security_context"),
                pending_timeout=merged.get("pending_timeout"),
                pvc_config=merged.get("pvc_config") or {},
                pull_policy=merged.get("pull_policy"),
            )
        )

    if len(servers) != 1:
        raise ValueError(f"{config_path}: expected exactly one server participant, found {len(servers)}: {servers}")
    server_cloud = next(p.cloud for p in participants if p.role == "server")

    gcp = cloud_derived.get("gcp", {})
    aws = cloud_derived.get("aws", {})
    return DeployConfig(
        participants=participants,
        server_cloud=server_cloud,
        gcp_project=gcp.get("project"),
        gcp_region=gcp.get("region"),
        aws_region=aws.get("region"),
        aws_eks_cluster_name=aws.get("eks_cluster_name"),
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
    if cmd[:3] == ["aws", "ec2", "allocate-address"]:
        return json.dumps({"PublicIp": "<server-ip>", "AllocationId": "<alloc-id>"}) + "\n"
    if cmd[:3] == ["aws", "eks", "describe-cluster"]:
        return "<vpc-id>\n"
    if cmd[:3] == ["aws", "ec2", "describe-subnets"]:
        return "<subnet-id>\n"
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
        if (cmd[0] == "gcloud" and "auth" in cmd) or (cmd[0] == "aws" and "sts" in cmd):
            return FakeProc(0)
        return FakeProc(1)
    return subprocess.run(cmd, capture_output=True, text=True)


def kubectl(kubeconfig: str, *args) -> subprocess.CompletedProcess:
    return run(["kubectl", "--kubeconfig", kubeconfig] + list(args))


def kubectl_check(kubeconfig: str, *args) -> bool:
    r = run_quiet(["kubectl", "--kubeconfig", kubeconfig] + list(args))
    return r.returncode == 0


def helm(kubeconfig: str, *args) -> subprocess.CompletedProcess:
    return run(["helm", "--kubeconfig", kubeconfig] + list(args))


def check_auth(clouds_used: set):
    print("Checking cloud auth ...")
    if "gcp" in clouds_used:
        r = run_quiet(["gcloud", "auth", "print-access-token"])
        if r.returncode != 0:
            sys.exit("GCP auth failed. Run: gcloud auth login")
    if "aws" in clouds_used:
        r = run_quiet(["aws", "sts", "get-caller-identity"])
        if r.returncode != 0:
            sys.exit("AWS auth failed. Run: aws sso login")
    print("  Auth OK")


def check_auth_for(cloud: str):
    """Re-check auth before cloud-specific operations."""
    if cloud == "aws":
        r = run_quiet(["aws", "sts", "get-caller-identity"])
        if r.returncode != 0:
            sys.exit("AWS session expired. Run: aws sso login")
    elif cloud == "gcp":
        r = run_quiet(["gcloud", "auth", "print-access-token"])
        if r.returncode != 0:
            sys.exit("GCP auth expired. Run: gcloud auth login")


def load_state() -> dict:
    if DRY_RUN:
        return {}
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _synthetic_state(config: DeployConfig) -> dict:
    ip_name = "<alloc-id>" if config.server_cloud == "aws" else "nvflare-DRYRUN-0"
    state = {
        "server_ip": "<server-ip>",
        "ip_name": ip_name,
        "gcp_project": "<gcp-project>",
        "gcp_region": config.gcp_region or "us-central1",
        "aws_region": config.aws_region,
        "server_cloud": config.server_cloud,
        "participants": {
            p.name: {"kubeconfig": p.kubeconfig, "namespace": p.namespace, "cloud": p.cloud}
            for p in config.participants
        },
    }
    if config.server_cloud == "aws":
        state["aws_eip_allocation_id"] = "<alloc-id>"
        state["aws_nlb_subnet_id"] = "<subnet-id>"
    return state


def save_state(state: dict):
    if DRY_RUN:
        return
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def namespace_exists(kubeconfig: str, ns: str) -> bool:
    return kubectl_check(kubeconfig, "get", "ns", ns)


def helm_release_exists(kubeconfig: str, name: str, ns: str) -> bool:
    r = run_quiet(["helm", "--kubeconfig", kubeconfig, "status", name, "-n", ns])
    return r.returncode == 0


def pod_exists(kubeconfig: str, ns: str, name: str) -> bool:
    return kubectl_check(kubeconfig, "-n", ns, "get", "pod", name)


# ---------------------------------------------------------------------------
# Reserve / release static IP
# ---------------------------------------------------------------------------
def _ip_tag() -> str:
    if DRY_RUN:
        return "nvflare-DRYRUN-0"
    return f"nvflare-{os.environ.get('USER', 'dev')}-{int(time.time())}"


def reserve_ip(
    server_cloud: str,
    *,
    gcp_project: str | None = None,
    gcp_region: str | None = None,
    aws_region: str | None = None,
) -> tuple[str, str]:
    if server_cloud == "gcp":
        return _reserve_ip_gcp(gcp_project, gcp_region)
    if server_cloud == "aws":
        return _reserve_ip_aws(aws_region)
    if server_cloud == "azure":
        raise NotImplementedError("Azure server IP reservation not yet supported")
    raise ValueError(f"unknown server cloud: {server_cloud}")


def _reserve_ip_gcp(project: str, region: str) -> tuple[str, str]:
    ip_name = _ip_tag()
    print(f"Reserving static IP {ip_name} ...")
    run(["gcloud", "compute", "addresses", "create", ip_name, f"--region={region}", f"--project={project}", "--quiet"])
    r = run(
        [
            "gcloud",
            "compute",
            "addresses",
            "describe",
            ip_name,
            f"--region={region}",
            f"--project={project}",
            "--format=value(address)",
        ],
        capture=True,
    )
    ip = r.stdout.strip()
    print(f"  Reserved: {ip} ({ip_name})")
    return ip, ip_name


def _reserve_ip_aws(region: str) -> tuple[str, str]:
    tag = _ip_tag()
    print(f"Allocating Elastic IP {tag} ...")
    r = run(
        [
            "aws",
            "ec2",
            "allocate-address",
            "--domain",
            "vpc",
            "--region",
            region,
            "--tag-specifications",
            f"ResourceType=elastic-ip,Tags=[{{Key=Name,Value={tag}}}]",
            "--output",
            "json",
        ],
        capture=True,
    )
    resp = json.loads(r.stdout) if r.stdout.strip() else {}
    ip = resp.get("PublicIp", "")
    alloc_id = resp.get("AllocationId", "")
    if not ip or not alloc_id:
        raise RuntimeError(f"allocate-address returned unexpected response: {r.stdout!r}")
    print(f"  Reserved: {ip} ({alloc_id})")
    return ip, alloc_id


def _discover_aws_public_subnet(cluster_name: str, region: str) -> str:
    print(f"Discovering public subnet for EKS cluster {cluster_name} ...")
    r = run(
        [
            "aws",
            "eks",
            "describe-cluster",
            "--name",
            cluster_name,
            "--region",
            region,
            "--query",
            "cluster.resourcesVpcConfig.vpcId",
            "--output",
            "text",
        ],
        capture=True,
    )
    vpc_id = r.stdout.strip()
    if not vpc_id:
        raise RuntimeError(f"could not resolve VPC id for EKS cluster {cluster_name}")
    r = run(
        [
            "aws",
            "ec2",
            "describe-subnets",
            "--filters",
            f"Name=vpc-id,Values={vpc_id}",
            "Name=tag:kubernetes.io/role/elb,Values=1",
            "--region",
            region,
            "--query",
            "Subnets[0].SubnetId",
            "--output",
            "text",
        ],
        capture=True,
    )
    subnet_id = r.stdout.strip()
    if not subnet_id or subnet_id == "None":
        raise RuntimeError(f"no public subnet (tag kubernetes.io/role/elb=1) in VPC {vpc_id}")
    print(f"  Using subnet: {subnet_id}")
    return subnet_id


def release_ip(
    server_cloud: str,
    ip_name: str,
    *,
    gcp_project: str | None = None,
    gcp_region: str | None = None,
    aws_region: str | None = None,
):
    if not ip_name:
        return
    if server_cloud == "gcp":
        print(f"Releasing IP {ip_name} ...")
        run(
            [
                "gcloud",
                "compute",
                "addresses",
                "delete",
                ip_name,
                f"--region={gcp_region}",
                f"--project={gcp_project}",
                "--quiet",
            ],
            check=False,
        )
        return
    if server_cloud == "aws":
        print(f"Releasing Elastic IP {ip_name} ...")
        run(
            ["aws", "ec2", "release-address", "--allocation-id", ip_name, "--region", aws_region],
            check=False,
        )
        return
    if server_cloud == "azure":
        raise NotImplementedError("Azure server IP release not yet supported")
    raise ValueError(f"unknown server cloud: {server_cloud}")


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

    server = next(p for p in config.participants if p.role == "server")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    project_text = PROJECT_TEMPLATE.read_text()
    project_text = project_text.replace("__SERVER_IP__", server_ip)
    project_text = project_text.replace("__DOCKER_IMAGE__", server.image)
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

    nvflare = REPO_ROOT / ".venv" / "bin" / "nvflare"
    nvflare_cmd = str(nvflare) if nvflare.exists() else "nvflare"

    provision_dir = WORK_DIR / "provision"
    if provision_dir.exists():
        shutil.rmtree(provision_dir)

    run([nvflare_cmd, "provision", "-p", str(project_file), "-w", str(provision_dir)])

    prod_dirs = sorted(provision_dir.glob("*/prod_*"))
    if not prod_dirs:
        sys.exit("Provisioning produced no output")
    return prod_dirs[-1]


# ---------------------------------------------------------------------------
# Post-process resources.json for K8sJobLauncher
# ---------------------------------------------------------------------------
def patch_resources_json(
    kit_dir: Path,
    namespace: str,
    launcher_class: str,
    security_context: dict | None = None,
    pending_timeout: int | None = None,
):
    src = kit_dir / "local" / "resources.json.default"
    dst = kit_dir / "local" / "resources.json"
    if DRY_RUN:
        preview = {"namespace": namespace, "launcher_class": launcher_class}
        if security_context:
            preview["security_context"] = security_context
        if pending_timeout is not None:
            preview["pending_timeout"] = pending_timeout
        print(f"  Would patch {dst}: {json.dumps(preview)}")
        print(f"  Would write {kit_dir / 'local' / 'study_data_pvc.yaml'}: default: nvfldata")
        return
    r = json.loads(src.read_text())
    replaced = False
    for i, c in enumerate(r["components"]):
        if "process_launcher" in c.get("id", "") or "ProcessJobLauncher" in c.get("path", ""):
            args = {
                "config_file_path": None,
                "study_data_pvc_file_path": "/var/tmp/nvflare/workspace/local/study_data_pvc.yaml",
                "namespace": namespace,
                "python_path": "/usr/local/bin/python3",
            }
            if security_context:
                args["security_context"] = security_context
            if pending_timeout is not None:
                args["pending_timeout"] = pending_timeout
            r["components"][i] = {"id": "k8s_launcher", "path": launcher_class, "args": args}
            replaced = True
    if not replaced:
        raise RuntimeError(f"No ProcessJobLauncher component found in {src}; cannot inject K8sJobLauncher.")
    dst.write_text(json.dumps(r, indent=4))

    local_dir = kit_dir / "local"
    local_dir.mkdir(exist_ok=True)
    (local_dir / "study_data_pvc.yaml").write_text("default: nvfldata\n")


# ---------------------------------------------------------------------------
# Deploy one participant
# ---------------------------------------------------------------------------
def deploy_participant(
    p: Participant,
    prod_dir: Path,
    server_ip: str | None = None,
    aws_server_alloc_id: str | None = None,
    aws_server_subnet: str | None = None,
):
    kit_dir = prod_dir / p.name
    chart_dir = kit_dir / "helm_chart"

    print(f"\n{'=' * 60}")
    print(f"Deploying {p.name} → {p.namespace}")
    print(f"{'=' * 60}")

    check_auth_for(p.cloud)

    # 1. Namespace (idempotent)
    if not namespace_exists(p.kubeconfig, p.namespace):
        print(f"  Creating namespace {p.namespace}")
        kubectl(p.kubeconfig, "create", "ns", p.namespace)
    else:
        print(f"  Namespace {p.namespace} exists")

    # 2. PVCs (idempotent via apply)
    print("  Creating PVCs ...")
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
        r = run_quiet(["kubectl", "--kubeconfig", p.kubeconfig, "-n", p.namespace, "get", "pvc", pvc_name])
        if r.returncode != 0:
            run(["kubectl", "--kubeconfig", p.kubeconfig, "-n", p.namespace, "apply", "-f", "-"], input=pvc_yaml)
        else:
            print(f"    PVC {pvc_name} exists")

    # 3. Stage kit files (skip if helm release already exists)
    if helm_release_exists(p.kubeconfig, p.name, p.namespace):
        print(f"  Helm release {p.name} already deployed — skipping kit staging")
    else:
        print("  Staging kit files via temp pod ...")
        pod_name = f"kit-copy-{p.name}"

        # Build pod spec
        sec_ctx = {}
        if p.security_context:
            sec_ctx = {"securityContext": p.security_context}

        pod_spec = {
            "spec": {
                **sec_ctx,
                "volumes": [{"name": "ws", "persistentVolumeClaim": {"claimName": "nvflws"}}],
                "containers": [
                    {
                        "name": "copy",
                        "image": "busybox",
                        "command": ["sleep", "600"],
                        "volumeMounts": [{"name": "ws", "mountPath": "/ws"}],
                    }
                ],
            }
        }

        if pod_exists(p.kubeconfig, p.namespace, pod_name):
            kubectl(p.kubeconfig, "-n", p.namespace, "delete", "pod", pod_name, "--timeout=60s")

        kubectl(
            p.kubeconfig,
            "-n",
            p.namespace,
            "run",
            pod_name,
            "--image=busybox",
            "--restart=Never",
            f"--overrides={json.dumps(pod_spec)}",
        )

        print("  Waiting for PVCs to bind ...")
        kubectl(p.kubeconfig, "-n", p.namespace, "wait", "--for=condition=Ready", f"pod/{pod_name}", "--timeout=600s")

        kubectl(p.kubeconfig, "-n", p.namespace, "cp", str(kit_dir / "startup"), f"{pod_name}:/ws/startup")
        kubectl(p.kubeconfig, "-n", p.namespace, "cp", str(kit_dir / "local"), f"{pod_name}:/ws/local")
        kubectl(p.kubeconfig, "-n", p.namespace, "delete", "pod", pod_name, "--timeout=60s")

        # 4. Helm install
        print(f"  Helm installing {p.name} ...")
        helm_args = ["install", p.name, str(chart_dir), "-n", p.namespace]
        helm_args += p.helm_overrides

        if p.role == "server" and server_ip:
            helm_args += ["--set", "service.type=LoadBalancer"]
            if p.cloud == "aws":
                if not aws_server_alloc_id or not aws_server_subnet:
                    raise RuntimeError("AWS server requires EIP allocation id and NLB subnet id")
                annotations = {
                    "service.beta.kubernetes.io/aws-load-balancer-type": "external",
                    "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "ip",
                    "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
                    "service.beta.kubernetes.io/aws-load-balancer-eip-allocations": aws_server_alloc_id,
                    "service.beta.kubernetes.io/aws-load-balancer-subnets": aws_server_subnet,
                    # Single-AZ NLB (one subnet annotation) needs cross-zone to reach pods in other AZs.
                    "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled": "true",
                }
                for k, v in annotations.items():
                    escaped = k.replace(".", r"\.")
                    helm_args += ["--set-string", f"service.annotations.{escaped}={v}"]
            else:
                helm_args += ["--set", f"service.loadBalancerIP={server_ip}"]

        if p.security_context:
            for k, v in _flatten_set("securityContext", p.security_context):
                helm_args += ["--set", f"{k}={v}"]

        if ":" in p.image:
            repo, tag = p.image.rsplit(":", 1)
            helm_args += ["--set", f"image.repository={repo}", "--set", f"image.tag={tag}"]
        else:
            helm_args += ["--set", f"image.repository={p.image}"]

        if p.pull_policy:
            helm_args += ["--set", f"image.pullPolicy={p.pull_policy}"]

        helm(p.kubeconfig, *helm_args)

    # 5. Wait for server deployment
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


def _flatten_set(prefix: str, d: dict) -> list[tuple[str, str]]:
    result = []
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            result.extend(_flatten_set(key, v))
        elif isinstance(v, bool):
            # helm --set treats only lowercase true/false as booleans
            result.append((key, "true" if v else "false"))
        else:
            result.append((key, str(v)))
    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_up(args):
    config = load_config(Path(args.config))
    participants = config.participants
    clouds_used = {p.cloud for p in participants}
    check_auth(clouds_used)

    if config.server_cloud == "aws" and not config.aws_eks_cluster_name:
        sys.exit("clouds.aws.eks_cluster_name is required when the server is in AWS")

    gcp_project = None
    if "gcp" in clouds_used:
        gcp_project = (
            config.gcp_project or run(["gcloud", "config", "get-value", "project"], capture=True).stdout.strip()
        )
    gcp_region = config.gcp_region or "us-central1"
    aws_region = config.aws_region

    state = load_state()

    # Reserve IP (skip if already in state)
    if "server_ip" in state and state["server_ip"]:
        server_ip = state["server_ip"]
        ip_name = state.get("ip_name", "")
        print(f"Reusing IP from state: {server_ip}")
    else:
        server_ip, ip_name = reserve_ip(
            config.server_cloud,
            gcp_project=gcp_project,
            gcp_region=gcp_region,
            aws_region=aws_region,
        )

    # Discover AWS NLB subnet when server is in AWS
    if config.server_cloud == "aws":
        nlb_subnet = state.get("aws_nlb_subnet_id") or _discover_aws_public_subnet(
            config.aws_eks_cluster_name, aws_region
        )
        state["aws_nlb_subnet_id"] = nlb_subnet
        state["aws_eip_allocation_id"] = ip_name

    # Save state
    state.update(
        {
            "server_ip": server_ip,
            "ip_name": ip_name,
            "gcp_project": gcp_project,
            "gcp_region": gcp_region,
            "aws_region": aws_region,
            "server_cloud": config.server_cloud,
            "participants": {
                p.name: {"kubeconfig": p.kubeconfig, "namespace": p.namespace, "cloud": p.cloud} for p in participants
            },
        }
    )
    save_state(state)

    prod_dir = provision(server_ip, config)

    # Post-process resources.json
    print("Post-processing resources.json for K8sJobLauncher ...")
    for p in participants:
        patch_resources_json(
            prod_dir / p.name,
            p.namespace,
            p.launcher_class,
            p.security_context,
            p.pending_timeout,
        )

    # Deploy each participant
    aws_alloc_id = state.get("aws_eip_allocation_id")
    aws_subnet = state.get("aws_nlb_subnet_id")
    for p in participants:
        deploy_participant(
            p,
            prod_dir,
            server_ip=server_ip,
            aws_server_alloc_id=aws_alloc_id,
            aws_server_subnet=aws_subnet,
        )

    print(f"\n{'=' * 60}")
    print("Deployment complete.")
    print(f"  Server IP:   {server_ip}")
    print(f"  Admin kit:   {prod_dir / 'admin@nvidia.com'}")
    print(f"{'=' * 60}")


def cmd_down(args):
    state = load_state()
    if not state and DRY_RUN:
        state = _synthetic_state(load_config(Path(args.config)))
    if not state:
        sys.exit("No state file found. Nothing to destroy.")

    participants = state.get("participants", {})
    fail = False

    for name, info in participants.items():
        kc, ns, cloud = info["kubeconfig"], info["namespace"], info.get("cloud", "")
        print(f"Tearing down {name} ({ns}) ...")
        try:
            check_auth_for(cloud)
        except SystemExit:
            print(f"  Auth failed for {kc} — skipping")
            fail = True
            continue

        run(["helm", "--kubeconfig", kc, "uninstall", name, "-n", ns], check=False)
        r = run(
            ["kubectl", "--kubeconfig", kc, "delete", "ns", ns, "--ignore-not-found", "--timeout=120s"], check=False
        )
        if r.returncode != 0:
            fail = True

    if fail:
        print("Partial teardown. State preserved for retry.")
        sys.exit(1)

    ip_name = state.get("ip_name")
    if ip_name:
        release_ip(
            state.get("server_cloud", "gcp"),
            ip_name,
            gcp_project=state.get("gcp_project", ""),
            gcp_region=state.get("gcp_region", "us-central1"),
            aws_region=state.get("aws_region"),
        )

    STATE_FILE.unlink(missing_ok=True)
    print("Destroyed.")


def cmd_status(args):
    state = load_state()
    if not state and DRY_RUN:
        state = _synthetic_state(load_config(Path(args.config)))
    if not state:
        print("No deployment state found.")
        return

    print(f"Server IP: {state.get('server_ip', 'N/A')}")
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
    default_config = str(TOOL_DIR / "gcp-server.yaml")
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
