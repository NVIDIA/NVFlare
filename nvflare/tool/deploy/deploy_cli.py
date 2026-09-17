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

"""nvflare deploy subcommands."""

import argparse
import sys
from typing import Optional

from nvflare.tool.cli_output import output_usage_error
from nvflare.tool.cli_schema import handle_schema_flag

_deploy_root_parser: Optional[argparse.ArgumentParser] = None
_deploy_prepare_parser: Optional[argparse.ArgumentParser] = None
_deploy_k8_parser: Optional[argparse.ArgumentParser] = None
_deploy_k8_stage_parser: Optional[argparse.ArgumentParser] = None
_deploy_k8_unstage_parser: Optional[argparse.ArgumentParser] = None

_DEPLOY_PREPARE_EXAMPLES = [
    "nvflare deploy prepare ./site-1",
    "nvflare deploy prepare ./site-1 --output ./site-1-docker --config docker.yaml",
    "nvflare deploy prepare ./site-1 --output ./site-1-k8s --config k8s.yaml",
    "nvflare deploy prepare ./site-1 --output ./site-1-slurm --config slurm.yaml",
]

_DEPLOY_K8_STAGE_EXAMPLES = [
    "nvflare deploy k8 stage ./site-1-k8s --namespace nvflare",
    "nvflare deploy k8 stage ./server-k8s --namespace nvflare --local-configmap server-local",
    "nvflare deploy k8 stage ./site-1-k8s --startup-secret site-1-startup",
]
_DEPLOY_K8_UNSTAGE_EXAMPLES = [
    "nvflare deploy k8 unstage ./site-1-k8s",
    "nvflare deploy k8 unstage ./server-k8s --namespace nvflare",
    "nvflare deploy k8 unstage ./site-1-k8s --kubectl oc",
]
_DEPLOY_K8_KUBECTL_CHOICES = ("kubectl", "oc")


def def_deploy_cli_parser(sub_cmd) -> dict:
    """Register 'nvflare deploy' with the top-level sub_cmd parser."""
    global _deploy_root_parser, _deploy_prepare_parser, _deploy_k8_parser
    global _deploy_k8_stage_parser, _deploy_k8_unstage_parser

    cmd = "deploy"
    parser = sub_cmd.add_parser(cmd, help="prepare startup kits and manage deployment resources")
    _deploy_root_parser = parser
    deploy_subparser = parser.add_subparsers(title="deploy subcommands", metavar="", dest="deploy_sub_cmd")

    prepare_parser = deploy_subparser.add_parser(
        "prepare",
        description="Prepare an existing server/client startup kit for Docker, Kubernetes, or Slurm.",
        help="prepare a startup kit for Docker, Kubernetes, or Slurm",
    )
    prepare_parser.add_argument("kit", nargs="?", help="Existing input startup kit directory.")
    prepare_parser.add_argument("--kit", dest="kit_flag", help="Existing input startup kit directory.")
    prepare_parser.add_argument(
        "--output",
        help="Directory to write the prepared startup kit copy. Defaults to <kit>/prepared/<runtime>.",
    )
    prepare_parser.add_argument(
        "--config",
        help=(
            "YAML runtime config file with top-level runtime: docker, runtime: k8s, or runtime: slurm. "
            "Defaults to <kit>/config.yaml."
        ),
    )
    prepare_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _deploy_prepare_parser = prepare_parser

    k8_parser = deploy_subparser.add_parser(
        "k8",
        aliases=["k8s"],
        description="Kubernetes deployment helper commands.",
        help="Kubernetes deployment helpers",
    )
    _deploy_k8_parser = k8_parser
    k8_subparser = k8_parser.add_subparsers(title="k8 subcommands", metavar="", dest="deploy_k8_sub_cmd")

    stage_parser = k8_subparser.add_parser(
        "stage",
        description=(
            "Create Kubernetes ConfigMap/Secret resources for a prepared kit's local/startup folders "
            "and patch the generated Helm chart to mount them."
        ),
        help="stage local/startup as Kubernetes ConfigMap/Secret",
    )
    stage_parser.add_argument("kit", nargs="?", help="Prepared Kubernetes startup kit directory.")
    stage_parser.add_argument("--kit", dest="kit_flag", help="Prepared Kubernetes startup kit directory.")
    stage_parser.add_argument(
        "--namespace",
        help="Kubernetes namespace for the ConfigMap and Secret. Defaults to the prepared launcher namespace.",
    )
    stage_parser.add_argument(
        "--local-configmap",
        help="ConfigMap name for the prepared local/ folder. Defaults to nvflare-local-<site>.",
    )
    stage_parser.add_argument(
        "--startup-secret",
        help="Secret name for the prepared startup/ folder. Defaults to nvflare-startup-<site>.",
    )
    stage_parser.add_argument(
        "--kubectl",
        choices=_DEPLOY_K8_KUBECTL_CHOICES,
        help="Kubernetes CLI command used for apply. Defaults to the KUBECTL environment variable or kubectl.",
    )
    stage_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _deploy_k8_stage_parser = stage_parser

    unstage_parser = k8_subparser.add_parser(
        "unstage",
        description=(
            "Delete the ConfigMap and Secret created by deploy k8 stage and clear their references "
            "from the prepared Helm chart. Run this after Helm uninstall."
        ),
        help="delete staged Kubernetes ConfigMap/Secret resources",
    )
    unstage_parser.add_argument("kit", nargs="?", help="Prepared Kubernetes startup kit directory used to stage.")
    unstage_parser.add_argument(
        "--kit", dest="kit_flag", help="Prepared Kubernetes startup kit directory used to stage."
    )
    unstage_parser.add_argument(
        "--namespace",
        help="Staged Kubernetes namespace. Defaults to the namespace recorded by stage.",
    )
    unstage_parser.add_argument(
        "--local-configmap",
        help="Exact staged ConfigMap name. Defaults to the name recorded by stage.",
    )
    unstage_parser.add_argument(
        "--startup-secret",
        help="Exact staged Secret name. Defaults to the name recorded by stage.",
    )
    unstage_parser.add_argument(
        "--kubectl",
        choices=_DEPLOY_K8_KUBECTL_CHOICES,
        help="Kubernetes CLI command used for delete. Defaults to the KUBECTL environment variable or kubectl.",
    )
    unstage_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _deploy_k8_unstage_parser = unstage_parser

    return {cmd: parser}


def handle_deploy_cmd(args):
    sub_cmd = getattr(args, "deploy_sub_cmd", None)
    if sub_cmd == "prepare":
        handle_schema_flag(_deploy_prepare_parser, "nvflare deploy prepare", _DEPLOY_PREPARE_EXAMPLES, sys.argv[1:])

        from nvflare.tool.deploy.deploy_commands import prepare_deployment

        prepare_deployment(args)
        return

    if sub_cmd in {"k8", "k8s"}:
        if getattr(args, "deploy_k8_sub_cmd", None) == "stage":
            handle_schema_flag(
                _deploy_k8_stage_parser,
                "nvflare deploy k8 stage",
                _DEPLOY_K8_STAGE_EXAMPLES,
                sys.argv[1:],
            )

            from nvflare.tool.deploy.deploy_commands import stage_k8_deployment

            stage_k8_deployment(args)
            return

        if getattr(args, "deploy_k8_sub_cmd", None) == "unstage":
            handle_schema_flag(
                _deploy_k8_unstage_parser,
                "nvflare deploy k8 unstage",
                _DEPLOY_K8_UNSTAGE_EXAMPLES,
                sys.argv[1:],
            )

            from nvflare.tool.deploy.deploy_commands import unstage_k8_deployment

            unstage_k8_deployment(args)
            return

        output_usage_error(_deploy_k8_parser, "deploy k8 subcommand required", exit_code=4)
        raise SystemExit(4)

    output_usage_error(_deploy_root_parser, "deploy subcommand required", exit_code=4)
    raise SystemExit(4)
