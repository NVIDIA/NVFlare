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

import importlib
import os
import sys
from types import SimpleNamespace

import pytest


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _aws_provider():
    multicloud_dir = os.path.join(_repo_root(), "examples", "devops", "multicloud")
    sys.path.insert(0, multicloud_dir)
    try:
        return importlib.import_module("clouds.aws").AwsProvider()
    finally:
        sys.path.remove(multicloud_dir)


def test_discover_public_subnet_selects_lowest_availability_zone_deterministically():
    provider = _aws_provider()
    calls = []
    responses = iter(
        [
            SimpleNamespace(stdout="vpc-123\n", returncode=0),
            SimpleNamespace(
                stdout=(
                    '[{"SubnetId":"subnet-c","AvailabilityZone":"us-west-2c"},'
                    '{"SubnetId":"subnet-a2","AvailabilityZone":"us-west-2a"},'
                    '{"SubnetId":"subnet-a1","AvailabilityZone":"us-west-2a"}]\n'
                ),
                returncode=0,
            ),
        ]
    )

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return next(responses)

    assert provider._discover_public_subnet(run, "test-cluster", "us-west-2") == (
        "subnet-a1",
        "us-west-2a",
    )
    assert "Subnets[].{SubnetId:SubnetId,AvailabilityZone:AvailabilityZone}" in calls[1][0]
    assert "Subnets[0]" not in calls[1][0]


def test_discover_public_subnet_selects_zone_required_by_bound_workspace_pvc():
    provider = _aws_provider()
    responses = iter(
        [
            SimpleNamespace(stdout="vpc-123\n", returncode=0),
            SimpleNamespace(
                stdout=(
                    '[{"SubnetId":"subnet-a","AvailabilityZone":"us-west-2a"},'
                    '{"SubnetId":"subnet-c","AvailabilityZone":"us-west-2c"}]\n'
                ),
                returncode=0,
            ),
        ]
    )

    def run(cmd, **kwargs):
        return next(responses)

    assert provider._discover_public_subnet(
        run,
        "test-cluster",
        "us-west-2",
        required_zones={"us-west-2c"},
        workspace_pvc="nvflws",
    ) == ("subnet-c", "us-west-2c")


def test_discover_public_subnet_reports_bound_workspace_pvc_zone_mismatch():
    provider = _aws_provider()
    responses = iter(
        [
            SimpleNamespace(stdout="vpc-123\n", returncode=0),
            SimpleNamespace(
                stdout='[{"SubnetId":"subnet-a","AvailabilityZone":"us-west-2a"}]\n',
                returncode=0,
            ),
        ]
    )

    def run(cmd, **kwargs):
        return next(responses)

    with pytest.raises(RuntimeError, match="required by bound workspace PVC nvflws"):
        provider._discover_public_subnet(
            run,
            "test-cluster",
            "us-west-2",
            required_zones={"us-west-2c"},
            workspace_pvc="nvflws",
        )


def test_bound_server_workspace_pvc_zones_reads_pv_node_affinity():
    provider = _aws_provider()
    config = SimpleNamespace(
        participants=[
            SimpleNamespace(
                role="server",
                kubeconfig="/tmp/aws-kubeconfig",
                namespace="nvflare-server",
                prepare={"parent": {"workspace_pvc": "nvflws"}},
            )
        ]
    )
    responses = iter(
        [
            SimpleNamespace(stdout='{"spec":{"volumeName":"pv-123"}}', stderr="", returncode=0),
            SimpleNamespace(
                stdout=(
                    '{"spec":{"nodeAffinity":{"required":{"nodeSelectorTerms":[{"matchExpressions":['
                    '{"key":"topology.ebs.csi.aws.com/zone","operator":"In","values":["us-west-2c"]}'
                    "]}]}}}}"
                ),
                stderr="",
                returncode=0,
            ),
        ]
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return next(responses)

    assert provider._bound_server_workspace_pvc_zones(run, config) == ("nvflws", {"us-west-2c"})
    assert calls[0][0][-1] == "--ignore-not-found"
    assert calls[1][0][-5:] == ["get", "pv", "pv-123", "-o", "json"]


def test_bound_server_workspace_pvc_zones_defaults_to_chart_workspace_pvc():
    provider = _aws_provider()
    config = SimpleNamespace(
        participants=[
            SimpleNamespace(
                role="server",
                kubeconfig="/tmp/aws-kubeconfig",
                namespace="nvflare-server",
                prepare={"parent": {"docker_image": "repo/nvflare:dev"}},
            )
        ]
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    assert provider._bound_server_workspace_pvc_zones(run, config) == ("nvflws", set())
    assert "nvflws" in calls[0]


def test_bound_server_workspace_pvc_zones_ignores_missing_pvc():
    provider = _aws_provider()
    config = SimpleNamespace(
        participants=[
            SimpleNamespace(
                role="server",
                kubeconfig="/tmp/aws-kubeconfig",
                namespace="nvflare-server",
                prepare={"parent": {"workspace_pvc": "nvflws"}},
            )
        ]
    )

    def run(cmd, **kwargs):
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    assert provider._bound_server_workspace_pvc_zones(run, config) == ("nvflws", set())


def test_prepare_server_state_records_subnet_and_availability_zone():
    provider = _aws_provider()
    provider._discover_public_subnet = lambda run, cluster_name, region, **kwargs: ("subnet-a", "us-west-2a")
    state = {"aws_eip_allocation_id": "eipalloc-123"}
    config = SimpleNamespace(aws_eks_cluster_name="test-cluster")

    provider.prepare_server_state(
        run=None,
        state=state,
        config=config,
        ip_name="nvflare-test",
        aws_region="us-west-2",
    )

    assert state["aws_nlb_subnet_id"] == "subnet-a"
    assert state["aws_nlb_availability_zone"] == "us-west-2a"


def test_prepare_server_state_honors_bound_workspace_pvc_zone():
    provider = _aws_provider()
    provider._bound_server_workspace_pvc_zones = lambda run, config: ("nvflws", {"us-west-2c"})
    discovered = []

    def discover(run, cluster_name, region, **kwargs):
        discovered.append(kwargs)
        return "subnet-c", "us-west-2c"

    provider._discover_public_subnet = discover
    state = {
        "aws_eip_allocation_id": "eipalloc-123",
        "aws_nlb_subnet_id": "subnet-a",
        "aws_nlb_availability_zone": "us-west-2a",
    }
    config = SimpleNamespace(aws_eks_cluster_name="test-cluster")

    provider.prepare_server_state(
        run=None,
        state=state,
        config=config,
        ip_name="nvflare-test",
        aws_region="us-west-2",
    )

    assert discovered == [{"required_zones": {"us-west-2c"}, "workspace_pvc": "nvflws"}]
    assert state["aws_nlb_subnet_id"] == "subnet-c"
    assert state["aws_nlb_availability_zone"] == "us-west-2c"


def test_server_service_helm_args_pin_server_to_nlb_availability_zone():
    provider = _aws_provider()

    args = provider.server_service_helm_args(
        server_ip="203.0.113.10",
        state={
            "aws_eip_allocation_id": "eipalloc-123",
            "aws_nlb_subnet_id": "subnet-a",
            "aws_nlb_availability_zone": "us-west-2a",
        },
    )

    assert "nodeSelector.topology\\.kubernetes\\.io/zone=us-west-2a" in args
    assert "service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-subnets=subnet-a" in args


def test_server_service_helm_args_require_availability_zone():
    provider = _aws_provider()

    with pytest.raises(RuntimeError, match="NLB availability zone"):
        provider.server_service_helm_args(
            server_ip="203.0.113.10",
            state={"aws_eip_allocation_id": "eipalloc-123", "aws_nlb_subnet_id": "subnet-a"},
        )
