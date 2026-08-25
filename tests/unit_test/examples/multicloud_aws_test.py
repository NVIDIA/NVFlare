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


def test_prepare_server_state_records_subnet_and_availability_zone():
    provider = _aws_provider()
    provider._discover_public_subnet = lambda run, cluster_name, region: ("subnet-a", "us-west-2a")
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
