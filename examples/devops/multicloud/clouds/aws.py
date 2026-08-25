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

import json

import yaml

from .base import CloudProvider, service_annotation_args


class AwsProvider(CloudProvider):
    name = "aws"
    auth_check_cmd = ["aws", "sts", "get-caller-identity"]
    auth_failed_message = "AWS auth failed. Run: aws sso login"
    auth_expired_message = "AWS session expired. Run: aws sso login"

    def _resolve_region(self, run, region: str | None) -> str:
        if isinstance(region, str) and region.strip():
            return region.strip()

        r = run(["aws", "configure", "get", "region"], capture=True, check=False)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()

        raise RuntimeError("AWS region is required. Set AWS_REGION or run 'aws configure set region <region>'.")

    def parse_kubeconfig(self, kc_path):
        data = yaml.safe_load(kc_path.read_text())
        current_ctx = data.get("current-context")
        if not current_ctx:
            raise ValueError(f"{kc_path}: no current-context")
        ctx = next((c for c in data.get("contexts", []) if c.get("name") == current_ctx), None)
        if not ctx:
            raise ValueError(f"{kc_path}: current-context {current_ctx!r} not found in contexts")
        cluster_name = ctx["context"]["cluster"]
        if not cluster_name.startswith("arn:aws:eks:"):
            raise ValueError(f"{kc_path}: EKS context cluster {cluster_name!r} is not an ARN")
        parts = cluster_name.split(":")
        if len(parts) < 6:
            raise ValueError(f"{kc_path}: malformed EKS ARN {cluster_name!r}")
        return {"region": parts[3], "eks_cluster_name": parts[5].split("/", 1)[1]}

    def validate_server_config(self, config):
        if not config.aws_eks_cluster_name:
            raise SystemExit("clouds.aws.eks_cluster_name is required when the server is in AWS")

    def reserve_ip(self, *, run, ip_tag, aws_region=None, state=None, **kwargs):
        aws_region = self._resolve_region(run, aws_region)
        addresses = self._find_addresses_by_name(run, ip_tag, aws_region)
        if len(addresses) > 1:
            raise RuntimeError(f"found multiple Elastic IPs tagged Name={ip_tag}; refusing to choose one")
        if addresses:
            address = addresses[0]
            alloc_id = address.get("AllocationId", "")
            ip = address.get("PublicIp", "")
            if state is not None:
                state["aws_eip_allocation_id"] = alloc_id
            print(f"Using Elastic IP {ip_tag}: {ip} ({alloc_id})")
            return ip, ip_tag

        print(f"Allocating Elastic IP {ip_tag} ...")
        r = run(
            [
                "aws",
                "ec2",
                "allocate-address",
                "--domain",
                "vpc",
                "--region",
                aws_region,
                "--tag-specifications",
                f"ResourceType=elastic-ip,Tags=[{{Key=Name,Value={ip_tag}}}]",
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
        if state is not None:
            state["aws_eip_allocation_id"] = alloc_id
        print(f"  Reserved: {ip} ({alloc_id})")
        return ip, ip_tag

    def _find_addresses_by_name(self, run, ip_name: str, region: str) -> list[dict]:
        r = run(
            [
                "aws",
                "ec2",
                "describe-addresses",
                "--filters",
                f"Name=tag:Name,Values={ip_name}",
                "--region",
                region,
                "--query",
                "Addresses[].{PublicIp:PublicIp,AllocationId:AllocationId}",
                "--output",
                "json",
            ],
            capture=True,
            check=False,
        )
        if r.returncode != 0:
            detail = ""
            stderr = getattr(r, "stderr", "") or ""
            if stderr.strip():
                detail = f": {stderr.strip()}"
            raise RuntimeError(f"failed to describe Elastic IPs tagged Name={ip_name} in region {region}{detail}")
        addresses = json.loads(r.stdout) if r.stdout.strip() else []
        if not isinstance(addresses, list):
            raise RuntimeError(f"describe-addresses returned unexpected response: {r.stdout!r}")
        return addresses

    def prepare_server_state(self, *, run, state, config, ip_name, aws_region=None, **kwargs):
        aws_region = self._resolve_region(run, aws_region)
        if not state.get("aws_eip_allocation_id"):
            addresses = self._find_addresses_by_name(run, ip_name, aws_region)
            if len(addresses) != 1:
                raise RuntimeError(f"expected one Elastic IP tagged Name={ip_name}, found {len(addresses)}")
            state["aws_eip_allocation_id"] = addresses[0].get("AllocationId")
        workspace_pvc, required_zones = self._bound_server_workspace_pvc_zones(run, config)
        nlb_subnet = state.get("aws_nlb_subnet_id")
        nlb_zone = state.get("aws_nlb_availability_zone")
        if not nlb_subnet or not nlb_zone or (required_zones and nlb_zone not in required_zones):
            nlb_subnet, nlb_zone = self._discover_public_subnet(
                run,
                config.aws_eks_cluster_name,
                aws_region,
                required_zones=required_zones,
                workspace_pvc=workspace_pvc,
            )
        state["aws_nlb_subnet_id"] = nlb_subnet
        state["aws_nlb_availability_zone"] = nlb_zone

    def _bound_server_workspace_pvc_zones(self, run, config) -> tuple[str | None, set[str]]:
        server = next(
            (participant for participant in getattr(config, "participants", []) if participant.role == "server"), None
        )
        if not server:
            return None, set()

        workspace_pvc = (server.prepare.get("parent") or {}).get("workspace_pvc")
        if not workspace_pvc:
            return None, set()

        r = run(
            [
                "kubectl",
                "--kubeconfig",
                server.kubeconfig,
                "-n",
                server.namespace,
                "get",
                "pvc",
                workspace_pvc,
                "-o",
                "json",
                "--ignore-not-found",
            ],
            capture=True,
            check=False,
        )
        if r.returncode != 0:
            stderr = getattr(r, "stderr", "") or ""
            if "not found" in stderr.lower():
                return workspace_pvc, set()
            raise RuntimeError(f"failed to inspect workspace PVC {server.namespace}/{workspace_pvc}: {stderr.strip()}")
        if not r.stdout.strip():
            return workspace_pvc, set()

        try:
            pvc = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"kubectl returned invalid JSON for workspace PVC {server.namespace}/{workspace_pvc}"
            ) from e
        pv_name = (pvc.get("spec") or {}).get("volumeName")
        if not pv_name:
            return workspace_pvc, set()

        r = run(
            ["kubectl", "--kubeconfig", server.kubeconfig, "get", "pv", pv_name, "-o", "json"],
            capture=True,
            check=False,
        )
        if r.returncode != 0:
            stderr = getattr(r, "stderr", "") or ""
            raise RuntimeError(
                f"failed to inspect PV {pv_name} bound to workspace PVC {server.namespace}/{workspace_pvc}: {stderr.strip()}"
            )
        try:
            pv = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"kubectl returned invalid JSON for PV {pv_name}") from e

        pv_spec = pv.get("spec") or {}
        terms = ((pv_spec.get("nodeAffinity") or {}).get("required") or {}).get("nodeSelectorTerms") or []
        zones = {
            zone
            for term in terms
            for expression in term.get("matchExpressions", [])
            if expression.get("key", "").endswith("/zone") and expression.get("operator") == "In"
            for zone in expression.get("values", [])
            if zone
        }
        labels = (pv.get("metadata") or {}).get("labels") or {}
        zones.update(value for key, value in labels.items() if key.endswith("/zone") and value)
        driver = (pv_spec.get("csi") or {}).get("driver", "")
        if not zones and ("ebs.csi." in driver or pv_spec.get("awsElasticBlockStore")):
            raise RuntimeError(
                f"could not determine the Availability Zone of AWS EBS PV {pv_name} bound to workspace PVC "
                f"{server.namespace}/{workspace_pvc}; refusing to select an NLB subnet independently"
            )
        if zones:
            print(
                f"Workspace PVC {server.namespace}/{workspace_pvc} is bound to {pv_name} "
                f"in Availability Zone(s): {', '.join(sorted(zones))}"
            )
        return workspace_pvc, zones

    def _discover_public_subnet(
        self,
        run,
        cluster_name: str,
        region: str,
        *,
        required_zones: set[str] | None = None,
        workspace_pvc: str | None = None,
    ) -> tuple[str, str]:
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
                "Subnets[].{SubnetId:SubnetId,AvailabilityZone:AvailabilityZone}",
                "--output",
                "json",
            ],
            capture=True,
        )
        subnets = json.loads(r.stdout) if r.stdout.strip() else []
        if not isinstance(subnets, list):
            raise RuntimeError(f"describe-subnets returned unexpected response: {r.stdout!r}")
        candidates = [
            (subnet.get("AvailabilityZone"), subnet.get("SubnetId"))
            for subnet in subnets
            if isinstance(subnet, dict) and subnet.get("AvailabilityZone") and subnet.get("SubnetId")
        ]
        if not candidates:
            raise RuntimeError(f"no public subnet (tag kubernetes.io/role/elb=1) in VPC {vpc_id}")
        if required_zones:
            candidates = [candidate for candidate in candidates if candidate[0] in required_zones]
            if not candidates:
                zones = ", ".join(sorted(required_zones))
                raise RuntimeError(
                    f"no public subnet (tag kubernetes.io/role/elb=1) in VPC {vpc_id} matches Availability "
                    f"Zone(s) {zones} required by bound workspace PVC {workspace_pvc}. Add or tag a public subnet "
                    "in a compatible zone, or preserve any required data and recreate the workspace PVC"
                )
        availability_zone, subnet_id = min(candidates)
        print(f"  Using subnet: {subnet_id} ({availability_zone})")
        return subnet_id, availability_zone

    def release_ip(self, *, run, ip_name, state):
        aws_region = self._resolve_region(run, state.get("aws_region"))
        try:
            addresses = self._find_addresses_by_name(run, ip_name, aws_region)
        except RuntimeError as e:
            print(f"  Warning: {e}. The Elastic IP may still be allocated and require manual cleanup.")
            return
        if not addresses:
            print(f"No Elastic IP tagged Name={ip_name} found.")
            return
        if len(addresses) > 1:
            raise RuntimeError(f"found multiple Elastic IPs tagged Name={ip_name}; refusing to release any")

        alloc_id = addresses[0].get("AllocationId", "")
        print(f"Releasing Elastic IP {ip_name} ({alloc_id}) ...")
        run(
            ["aws", "ec2", "release-address", "--allocation-id", alloc_id, "--region", aws_region],
            check=False,
        )

    def server_service_helm_args(self, *, server_ip, state):
        aws_server_alloc_id = state.get("aws_eip_allocation_id")
        aws_server_subnet = state.get("aws_nlb_subnet_id")
        aws_server_zone = state.get("aws_nlb_availability_zone")
        if not aws_server_alloc_id or not aws_server_subnet or not aws_server_zone:
            raise RuntimeError("AWS server requires EIP allocation id, NLB subnet id, and NLB availability zone")
        args = service_annotation_args(
            {
                "service.beta.kubernetes.io/aws-load-balancer-type": "external",
                "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "ip",
                "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
                "service.beta.kubernetes.io/aws-load-balancer-eip-allocations": aws_server_alloc_id,
                "service.beta.kubernetes.io/aws-load-balancer-subnets": aws_server_subnet,
                "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled": "true",
            }
        )
        args += [
            "--set-string",
            f"nodeSelector.topology\\.kubernetes\\.io/zone={aws_server_zone}",
        ]
        return args
