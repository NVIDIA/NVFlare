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
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "all-clouds.yaml"
DEFAULT_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the NVFlare image and push it to every registry used by a multicloud config."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="multicloud deploy config")
    parser.add_argument("--dockerfile", type=Path, default=DEFAULT_DOCKERFILE, help="Dockerfile to build")
    parser.add_argument("--context", type=Path, default=REPO_ROOT, help="docker build context")
    parser.add_argument("--platform", default="linux/amd64", help="docker build platform")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser.parse_args()


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def resolve_path(path: Path, *, base: Path | None = None) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return path.resolve()


def print_cmd(cmd: list[str]) -> None:
    print("  $ " + " ".join(shlex.quote(part) for part in cmd))


def run(
    cmd: list[str], *, dry_run: bool, input_text: str | None = None, quiet: bool = False
) -> subprocess.CompletedProcess:
    print_cmd(cmd)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    try:
        return subprocess.run(
            cmd,
            check=True,
            input=input_text,
            text=True,
            stdout=subprocess.DEVNULL if quiet else None,
        )
    except FileNotFoundError:
        fail(f"command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e


def capture(cmd: list[str]) -> str:
    print_cmd(cmd)
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        fail(f"command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr, file=sys.stderr, end="")
        raise SystemExit(e.returncode) from e
    return result.stdout


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


def image_for_cloud(cloud_name: str, cloud_config: dict) -> str:
    prepare = cloud_config.get("prepare") or {}
    parent = prepare.get("parent") or {}
    image = parent.get("docker_image")
    if not isinstance(image, str) or not image.strip():
        fail(f"clouds.{cloud_name}.prepare.parent.docker_image is required")
    return image.strip()


def collect_images(config: dict) -> list[str]:
    cloud_configs = config.get("clouds") or {}
    cloud_names = used_clouds(config) or list(cloud_configs)
    images = []
    for cloud_name in cloud_names:
        cloud_config = cloud_configs.get(cloud_name)
        if not isinstance(cloud_config, dict):
            fail(f"cloud '{cloud_name}' is used by participants but is not configured")
        image = image_for_cloud(cloud_name, cloud_config)
        if image not in images:
            images.append(image)
    if not images:
        fail("no clouds.<name>.prepare.parent.docker_image entries found")
    return images


def registry_host(image: str) -> str:
    if "/" not in image:
        return ""
    return image.split("/", 1)[0]


def validate_images(images: list[str], *, dry_run: bool) -> None:
    placeholders = [image for image in images if registry_host(image).endswith(".example.com")]
    if placeholders and not dry_run:
        fail(
            "replace placeholder image registry values before building:\n"
            + "\n".join(f"  {image}" for image in placeholders)
        )


def auth_registry(image: str, *, dry_run: bool) -> None:
    host = registry_host(image)
    if host.endswith(".pkg.dev"):
        run(["gcloud", "auth", "configure-docker", host, "--quiet"], dry_run=dry_run, quiet=True)
    elif ".dkr.ecr." in host and host.endswith(".amazonaws.com"):
        parts = host.split(".")
        if len(parts) < 6:
            fail(f"could not parse AWS ECR region from image host: {host}")
        region = parts[3]
        if dry_run:
            print(
                "  $ "
                + " ".join(
                    [
                        "aws",
                        "ecr",
                        "get-login-password",
                        "--region",
                        shlex.quote(region),
                        "|",
                        "docker",
                        "login",
                        "--username",
                        "AWS",
                        "--password-stdin",
                        shlex.quote(host),
                    ]
                )
            )
            return
        password = capture(["aws", "ecr", "get-login-password", "--region", region])
        run(
            ["docker", "login", "--username", "AWS", "--password-stdin", host],
            input_text=password,
            dry_run=False,
            quiet=True,
        )
    elif host.endswith(".azurecr.io"):
        acr_name = host.removesuffix(".azurecr.io")
        run(["az", "acr", "login", "--name", acr_name], dry_run=dry_run, quiet=True)
    else:
        print(f"warning: unknown registry host {host}; assuming docker is already authenticated", file=sys.stderr)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    dockerfile = resolve_path(args.dockerfile)
    context = resolve_path(args.context)
    config = load_config(config_path)
    images = collect_images(config)
    validate_images(images, dry_run=args.dry_run)

    primary = images[0]
    print(f"primary build tag: {primary}")
    for image in images[1:]:
        print(f"also tag:           {image}")

    for image in images:
        auth_registry(image, dry_run=args.dry_run)

    run(
        ["docker", "build", "--platform", args.platform, "-t", primary, "-f", str(dockerfile), str(context)],
        dry_run=args.dry_run,
    )
    for image in images[1:]:
        run(["docker", "tag", primary, image], dry_run=args.dry_run)
    for image in images:
        run(["docker", "push", image], dry_run=args.dry_run)

    print(f"=== built and pushed {len(images)} tag(s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
