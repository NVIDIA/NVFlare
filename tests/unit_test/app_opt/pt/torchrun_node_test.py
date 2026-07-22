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

import sys

import pytest

from nvflare.app_opt.pt.torchrun_node import TorchrunNodeError, build_torchrun_argv

_GROUP_ENV = {
    "NVFL_NNODES": "2",
    "NVFL_NODE_RANK": "1",
    "NVFL_MASTER_ADDR": "node-0",
    "NVFL_MASTER_PORT": "29512",
    "SLURM_JOB_ID": "4242",
}


def test_single_node_defaults_to_standalone_torchrun():
    argv = build_torchrun_argv(["--", "custom/client.py", "--epochs", "2"], {})

    assert argv[:4] == [sys.executable, "-u", "-m", "torch.distributed.run"]
    assert "--standalone" in argv
    assert "--nproc_per_node=auto" in argv
    assert argv[-3:] == ["custom/client.py", "--epochs", "2"]
    assert not any(word.startswith("--rdzv") for word in argv)


def test_node_group_environment_maps_to_rendezvous_arguments():
    argv = build_torchrun_argv(["--nproc-per-node=8", "--", "custom/client.py"], dict(_GROUP_ENV))

    for expected in (
        "--nproc_per_node=8",
        "--nnodes=2",
        "--node_rank=1",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=node-0:29512",
        "--rdzv_id=4242",
        "--rdzv_conf=join_timeout=600",
    ):
        assert expected in argv
    assert "--standalone" not in argv
    assert argv[-1] == "custom/client.py"


def test_join_timeout_must_be_positive():
    with pytest.raises(TorchrunNodeError, match="join-timeout"):
        build_torchrun_argv(["--join-timeout=0", "--", "custom/client.py"], dict(_GROUP_ENV))
