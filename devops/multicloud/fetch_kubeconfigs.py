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

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "all-clouds.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / ".tmp" / "kubeconfigs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch kubeconfigs for the clouds used by a multicloud deploy config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="multicloud deploy config")
    parser.add_argument("--dry-run", action="store_true", help="print cloud CLI commands without writing kubeconfigs")
    return parser.parse_args()


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def quote_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    parts = []
    for key, value in (env or {}).items():
        parts.append(f"{key}={shlex.quote(value)}")
    parts.extend(shlex.quote(part) for part in cmd)
    return "  $ " + " ".join(parts)


def run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(quote_command(cmd, env=env))
    if dry_run:
        return

    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        subprocess.run(cmd, check=True, env=full_env)
    except FileNotFoundError:
        fail(f"command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e


def capture(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        fail(f"command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr, file=sys.stderr, end="")
        raise SystemExit(e.returncode) from e
    return result.stdout.strip()


def resolve_path(path: Path, *, base: Path | None = None) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return path.resolve()


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        fail(f"config not found: {config_path}")
    with config_path.open("rt", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        fail(f"config must be a YAML mapping: {config_path}")
    return config


def used_clouds(config: dict) -> list[str]:
    participants = config.get("participants") or []
    clouds = []
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        cloud = participant.get("cloud")
        if cloud and cloud not in clouds:
            clouds.append(cloud)
    return clouds


def kubeconfig_path(cloud: str, out_dir: Path) -> Path:
    return out_dir / f"{cloud}.yaml"


def split_tsv_rows(output: str) -> list[tuple[str, str]]:
    rows = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def discover_gcp(project: str, cluster: str, location: str) -> tuple[str, str, str]:
    if not project:
        project = capture(["gcloud", "config", "get-value", "project"])
        if project == "(unset)":
            project = ""
    if not project:
        fail("GCP project is not set. Run 'gcloud config set project <project>' or set GCP_PROJECT.")

    if cluster and location:
        return project, cluster, location

    output = capture(["gcloud", "container", "clusters", "list", "--project", project, "--format=value(name,location)"])
    rows = [
        (name, loc)
        for name, loc in split_tsv_rows(output)
        if (not cluster or name == cluster) and (not location or loc == location)
    ]

    if len(rows) == 1:
        cluster, location = rows[0]
        return project, cluster, location
    if not rows:
        fail(f"No GKE clusters found for project {project}. Set GCP_CLUSTER and GCP_LOCATION.")

    print("Multiple GKE clusters matched. Set GCP_CLUSTER and GCP_LOCATION:", file=sys.stderr)
    for name, loc in rows:
        print(f"  {name}|{loc}", file=sys.stderr)
    raise SystemExit(1)


def discover_aws(cluster: str, region: str) -> tuple[str, str]:
    if not region:
        region = capture(["aws", "configure", "get", "region"])
    if not region:
        fail("AWS region is not set. Run 'aws configure set region <region>' or set AWS_REGION.")

    if cluster:
        return cluster, region

    output = capture(["aws", "eks", "list-clusters", "--region", region, "--query", "clusters[]", "--output", "text"])
    clusters = [name for name in output.split() if name]

    if len(clusters) == 1:
        return clusters[0], region
    if not clusters:
        fail(f"No EKS clusters found in region {region}. Set AWS_CLUSTER.")

    print(f"Multiple EKS clusters found in region {region}. Set AWS_CLUSTER:", file=sys.stderr)
    for name in clusters:
        print(f"  {name}", file=sys.stderr)
    raise SystemExit(1)


def discover_azure(cluster: str, resource_group: str) -> tuple[str, str]:
    if cluster and resource_group:
        return cluster, resource_group

    output = capture(["az", "aks", "list", "--query", "[].{name:name,resourceGroup:resourceGroup}", "--output", "tsv"])
    rows = [
        (name, rg)
        for name, rg in split_tsv_rows(output)
        if (not cluster or name == cluster) and (not resource_group or rg == resource_group)
    ]

    if len(rows) == 1:
        cluster, resource_group = rows[0]
        return cluster, resource_group
    if not rows:
        fail("No AKS clusters found in the active Azure subscription. Set AZURE_CLUSTER and AZURE_RESOURCE_GROUP.")

    print(
        "Multiple AKS clusters matched in the active Azure subscription. Set AZURE_CLUSTER and AZURE_RESOURCE_GROUP:",
        file=sys.stderr,
    )
    for name, rg in rows:
        print(f"  {name}|{rg}", file=sys.stderr)
    raise SystemExit(1)


def fetch_gcp(kubeconfig: Path, dry_run: bool) -> None:
    project, cluster, location = discover_gcp(
        env_first("GCP_PROJECT"),
        env_first("GCP_CLUSTER"),
        env_first("GCP_LOCATION"),
    )
    print(f"Fetching GKE kubeconfig: cluster={cluster} location={location} project={project}")
    run(
        ["gcloud", "container", "clusters", "get-credentials", cluster, "--location", location, "--project", project],
        dry_run=dry_run,
        env={"KUBECONFIG": str(kubeconfig)},
    )


def fetch_aws(kubeconfig: Path, dry_run: bool) -> None:
    cluster, region = discover_aws(
        env_first("AWS_CLUSTER", "AWS_CLUSTER_NAME"), env_first("AWS_REGION", "AWS_DEFAULT_REGION")
    )
    print(f"Fetching EKS kubeconfig: cluster={cluster} region={region}")
    run(
        ["aws", "eks", "update-kubeconfig", "--name", cluster, "--region", region, "--kubeconfig", str(kubeconfig)],
        dry_run=dry_run,
    )


def fetch_azure(kubeconfig: Path, dry_run: bool) -> None:
    cluster, resource_group = discover_azure(
        env_first("AZURE_CLUSTER", "AZURE_CLUSTER_NAME"), env_first("AZURE_RESOURCE_GROUP")
    )
    print(f"Fetching AKS kubeconfig: cluster={cluster} resource_group={resource_group}")
    run(
        [
            "az",
            "aks",
            "get-credentials",
            "--resource-group",
            resource_group,
            "--name",
            cluster,
            "--file",
            str(kubeconfig),
            "--overwrite-existing",
        ],
        dry_run=dry_run,
    )
    if shutil.which("kubelogin"):
        run(["kubelogin", "convert-kubeconfig", "-l", "azurecli", "--kubeconfig", str(kubeconfig)], dry_run=dry_run)
    else:
        print("  kubelogin not found; leaving Azure kubeconfig as written by az.")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    out_dir = DEFAULT_OUT_DIR
    config = load_config(config_path)
    clouds_config = config.get("clouds") or {}

    for cloud in used_clouds(config):
        cloud_config = clouds_config.get(cloud)
        if not isinstance(cloud_config, dict):
            fail(f"cloud '{cloud}' is used by participants but is not configured")

        kubeconfig = kubeconfig_path(cloud, out_dir)
        run(["mkdir", "-p", str(kubeconfig.parent)], dry_run=args.dry_run)
        if not args.dry_run:
            kubeconfig.parent.mkdir(parents=True, exist_ok=True)

        if cloud == "gcp":
            fetch_gcp(kubeconfig, args.dry_run)
        elif cloud == "aws":
            fetch_aws(kubeconfig, args.dry_run)
        elif cloud == "azure":
            fetch_azure(kubeconfig, args.dry_run)
        else:
            fail(f"unsupported cloud: {cloud}")
        print(f"Wrote {kubeconfig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
