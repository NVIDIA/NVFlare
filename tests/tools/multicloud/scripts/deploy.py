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
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOOL_DIR = SCRIPT_DIR.parent
REPO_ROOT = TOOL_DIR.parents[2]
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
    launcher_class: str
    helm_overrides: list = field(default_factory=list)
    security_context: dict | None = None
    pvc_config: dict = field(default_factory=dict)


def default_participants() -> list[Participant]:
    gcp_kc = str(REPO_ROOT / ".tmp" / "kubeconfigs" / "gcp.yaml")
    aws_kc = str(REPO_ROOT / ".tmp" / "kubeconfigs" / "aws.yaml")
    return [
        Participant(
            name="gcp-server",
            namespace="nvflare-server",
            kubeconfig=gcp_kc,
            role="server",
            launcher_class="nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher",
            helm_overrides=["--set", "hostPortEnabled=false", "--set", "tcpConfigMapEnabled=false"],
            pvc_config={
                "nvfletc": {"sc": "standard-rwo", "access": "ReadWriteOnce", "size": "1Gi"},
                "nvflws": {"sc": "filestore-rwx", "access": "ReadWriteMany", "size": "1Ti"},
                "nvfldata": {"sc": "standard-rwo", "access": "ReadWriteOnce", "size": "1Gi"},
            },
        ),
        Participant(
            name="gcp-client-1",
            namespace="nvflare-client-1",
            kubeconfig=gcp_kc,
            role="client",
            launcher_class="nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
            helm_overrides=["--set", "hostPortEnabled=false", "--set", "tcpConfigMapEnabled=false"],
            pvc_config={
                "nvfletc": {"sc": "standard-rwo", "access": "ReadWriteOnce", "size": "1Gi"},
                "nvflws": {"sc": "filestore-rwx", "access": "ReadWriteMany", "size": "1Ti"},
                "nvfldata": {"sc": "standard-rwo", "access": "ReadWriteOnce", "size": "1Gi"},
            },
        ),
        Participant(
            name="aws-client-2",
            namespace="nvflare-client-2",
            kubeconfig=aws_kc,
            role="client",
            launcher_class="nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
            security_context={"seLinuxOptions": {"type": "spc_t"}},
            pvc_config={
                "nvfletc": {"sc": "auto-ebs-sc", "access": "ReadWriteOnce", "size": "1Gi"},
                "nvflws": {"sc": "efs-sc", "access": "ReadWriteMany", "size": "10Gi"},
                "nvfldata": {"sc": "auto-ebs-sc", "access": "ReadWriteOnce", "size": "1Gi"},
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: list[str], check=True, capture=False, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, **kwargs)


def run_quiet(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def kubectl(kubeconfig: str, *args) -> subprocess.CompletedProcess:
    return run(["kubectl", "--kubeconfig", kubeconfig] + list(args))


def kubectl_check(kubeconfig: str, *args) -> bool:
    r = run_quiet(["kubectl", "--kubeconfig", kubeconfig] + list(args))
    return r.returncode == 0


def helm(kubeconfig: str, *args) -> subprocess.CompletedProcess:
    return run(["helm", "--kubeconfig", kubeconfig] + list(args))


def check_auth():
    print("Checking cloud auth ...")
    r = run_quiet(["gcloud", "auth", "print-access-token"])
    if r.returncode != 0:
        sys.exit("GCP auth failed. Run: gcloud auth login")
    r = run_quiet(["aws", "sts", "get-caller-identity"])
    if r.returncode != 0:
        sys.exit("AWS auth failed. Run: aws sso login")
    print("  Auth OK")


def check_auth_for(kubeconfig: str):
    """Re-check auth before cloud-specific operations."""
    if "aws" in kubeconfig.lower():
        r = run_quiet(["aws", "sts", "get-caller-identity"])
        if r.returncode != 0:
            sys.exit("AWS session expired. Run: aws sso login")
    else:
        r = run_quiet(["gcloud", "auth", "print-access-token"])
        if r.returncode != 0:
            sys.exit("GCP auth expired. Run: gcloud auth login")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
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
def reserve_ip(project: str, region: str, provided_ip: str | None) -> tuple[str, str]:
    if provided_ip:
        print(f"Using provided IP: {provided_ip}")
        return provided_ip, ""
    ip_name = f"nvflare-{os.environ.get('USER', 'dev')}-{int(time.time())}"
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


def release_ip(ip_name: str, project: str, region: str):
    if not ip_name:
        return
    print(f"Releasing IP {ip_name} ...")
    run(
        ["gcloud", "compute", "addresses", "delete", ip_name, f"--region={region}", f"--project={project}", "--quiet"],
        check=False,
    )


# ---------------------------------------------------------------------------
# Provision
# ---------------------------------------------------------------------------
def provision(server_ip: str, image: str) -> Path:
    print("Provisioning ...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    project_text = PROJECT_TEMPLATE.read_text()
    project_text = project_text.replace("__SERVER_IP__", server_ip)
    project_text = project_text.replace("__DOCKER_IMAGE__", image)
    project_file = WORK_DIR / "project.yml"
    project_file.write_text(project_text)

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
def patch_resources_json(kit_dir: Path, namespace: str, launcher_class: str, security_context: dict | None = None):
    src = kit_dir / "local" / "resources.json.default"
    dst = kit_dir / "local" / "resources.json"
    r = json.loads(src.read_text())
    replaced = False
    for i, c in enumerate(r["components"]):
        if "process_launcher" in c.get("id", "") or "ProcessJobLauncher" in c.get("path", ""):
            args = {
                "config_file_path": None,
                "workspace_pvc": "nvflws",
                "study_data_pvc_file_path": "/var/tmp/nvflare/etc/study_data_pvc.yaml",
                "namespace": namespace,
                "python_path": "/usr/local/bin/python3",
            }
            if security_context:
                args["security_context"] = security_context
            r["components"][i] = {"id": "k8s_launcher", "path": launcher_class, "args": args}
            replaced = True
    if not replaced:
        raise RuntimeError(f"No ProcessJobLauncher component found in {src}; cannot inject K8sJobLauncher.")
    dst.write_text(json.dumps(r, indent=4))

    etc_dir = kit_dir / "etc"
    etc_dir.mkdir(exist_ok=True)
    (etc_dir / "study_data_pvc.yaml").write_text("default: nvfldata\n")


# ---------------------------------------------------------------------------
# Deploy one participant
# ---------------------------------------------------------------------------
def deploy_participant(p: Participant, prod_dir: Path, server_ip: str | None = None, aws_image: str | None = None):
    kit_dir = prod_dir / p.name
    chart_dir = kit_dir / "helm_chart"

    print(f"\n{'=' * 60}")
    print(f"Deploying {p.name} → {p.namespace}")
    print(f"{'=' * 60}")

    check_auth_for(p.kubeconfig)

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
                "volumes": [
                    {"name": "ws", "persistentVolumeClaim": {"claimName": "nvflws"}},
                    {"name": "etc", "persistentVolumeClaim": {"claimName": "nvfletc"}},
                ],
                "containers": [
                    {
                        "name": "copy",
                        "image": "busybox",
                        "command": ["sleep", "600"],
                        "volumeMounts": [
                            {"name": "ws", "mountPath": "/ws"},
                            {"name": "etc", "mountPath": "/etc-vol"},
                        ],
                    }
                ],
            }
        }

        if pod_exists(p.kubeconfig, p.namespace, pod_name):
            kubectl(p.kubeconfig, "-n", p.namespace, "delete", "pod", pod_name)

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
        kubectl(
            p.kubeconfig,
            "-n",
            p.namespace,
            "cp",
            str(kit_dir / "etc" / "study_data_pvc.yaml"),
            f"{pod_name}:/etc-vol/study_data_pvc.yaml",
        )
        kubectl(p.kubeconfig, "-n", p.namespace, "delete", "pod", pod_name, "--wait=false")

        # 4. Helm install
        print(f"  Helm installing {p.name} ...")
        helm_args = ["install", p.name, str(chart_dir), "-n", p.namespace]
        helm_args += p.helm_overrides

        if p.role == "server" and server_ip:
            helm_args += ["--set", "service.type=LoadBalancer", "--set", f"service.loadBalancerIP={server_ip}"]

        if p.security_context:
            for k, v in _flatten_set("securityContext", p.security_context):
                helm_args += ["--set", f"{k}={v}"]

        if "aws" in p.kubeconfig.lower():
            if not aws_image:
                print("  WARNING: No --aws-image set. AWS client will use GCP image from values.yaml.")
                print("           EKS cannot pull from GCP Artifact Registry without mirroring.")
            else:
                repo, tag = aws_image.rsplit(":", 1)
                helm_args += ["--set", f"image.repository={repo}", "--set", f"image.tag={tag}"]

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
        else:
            result.append((key, str(v)))
    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_up(args):
    check_auth()

    gcp_project = args.gcp_project or run(["gcloud", "config", "get-value", "project"], capture=True).stdout.strip()
    gcp_region = args.gcp_region

    state = load_state()
    participants = default_participants()

    # Reserve IP (skip if already in state)
    if "server_ip" in state and state["server_ip"]:
        server_ip = state["server_ip"]
        ip_name = state.get("ip_name", "")
        print(f"Reusing IP from state: {server_ip}")
    else:
        server_ip, ip_name = reserve_ip(gcp_project, gcp_region, args.server_ip)

    # Save state
    state.update(
        {
            "server_ip": server_ip,
            "ip_name": ip_name,
            "gcp_project": gcp_project,
            "gcp_region": gcp_region,
            "participants": {p.name: {"kubeconfig": p.kubeconfig, "namespace": p.namespace} for p in participants},
        }
    )
    save_state(state)

    # Provision (skip if already done)
    prod_dir = None
    existing = list(WORK_DIR.glob("provision/*/prod_*"))
    if existing and not args.force_provision:
        prod_dir = sorted(existing)[-1]
        print(f"Reusing existing provision: {prod_dir}")
    else:
        prod_dir = provision(server_ip, args.gcp_image)

    # Post-process resources.json
    print("Post-processing resources.json for K8sJobLauncher ...")
    for p in participants:
        patch_resources_json(prod_dir / p.name, p.namespace, p.launcher_class, p.security_context)

    # Deploy each participant
    for p in participants:
        deploy_participant(p, prod_dir, server_ip=server_ip, aws_image=args.aws_image)

    print(f"\n{'=' * 60}")
    print("Deployment complete.")
    print(f"  Server IP:   {server_ip}")
    print(f"  Admin kit:   {prod_dir / 'admin@nvidia.com'}")
    print(f"{'=' * 60}")


def cmd_down(args):
    state = load_state()
    if not state:
        sys.exit("No state file found. Nothing to destroy.")

    participants = state.get("participants", {})
    fail = False

    for name, info in participants.items():
        kc, ns = info["kubeconfig"], info["namespace"]
        print(f"Tearing down {name} ({ns}) ...")
        try:
            check_auth_for(kc)
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
        release_ip(ip_name, state.get("gcp_project", ""), state.get("gcp_region", "us-central1"))

    STATE_FILE.unlink(missing_ok=True)
    print("Destroyed.")


def cmd_status(args):
    state = load_state()
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
    parser = argparse.ArgumentParser(description="Multicloud NVFlare deploy tool")
    sub = parser.add_subparsers(dest="command")

    up = sub.add_parser("up", help="Deploy server + clients")
    up.add_argument("--gcp-image", default=os.environ.get("GCP_IMAGE", ""), help="GCP container image")
    up.add_argument("--aws-image", default=os.environ.get("AWS_IMAGE", ""), help="AWS container image (ECR)")
    up.add_argument("--server-ip", default=os.environ.get("GCP_SERVER_IP"), help="Static IP (omit to auto-reserve)")
    up.add_argument("--gcp-project", default=os.environ.get("GCP_PROJECT"))
    up.add_argument("--gcp-region", default=os.environ.get("GCP_REGION", "us-central1"))
    up.add_argument("--force-provision", action="store_true", help="Re-provision even if output exists")

    sub.add_parser("down", help="Tear down everything")
    sub.add_parser("status", help="Show deployment status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"up": cmd_up, "down": cmd_down, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
