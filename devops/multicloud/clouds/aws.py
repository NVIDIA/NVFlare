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
        )
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
        nlb_subnet = state.get("aws_nlb_subnet_id") or self._discover_public_subnet(
            run, config.aws_eks_cluster_name, aws_region
        )
        state["aws_nlb_subnet_id"] = nlb_subnet

    def _discover_public_subnet(self, run, cluster_name: str, region: str) -> str:
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

    def release_ip(self, *, run, ip_name, state):
        aws_region = self._resolve_region(run, state.get("aws_region"))
        addresses = self._find_addresses_by_name(run, ip_name, aws_region)
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
        if not aws_server_alloc_id or not aws_server_subnet:
            raise RuntimeError("AWS server requires EIP allocation id and NLB subnet id")
        return service_annotation_args(
            {
                "service.beta.kubernetes.io/aws-load-balancer-type": "external",
                "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "ip",
                "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
                "service.beta.kubernetes.io/aws-load-balancer-eip-allocations": aws_server_alloc_id,
                "service.beta.kubernetes.io/aws-load-balancer-subnets": aws_server_subnet,
                # Single-AZ NLB (one subnet annotation) needs cross-zone to reach pods in other AZs.
                "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled": "true",
            }
        )
