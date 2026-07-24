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

import pytest

from nvflare.app_common.multinode import NodeGroup, NodeGroupError, split_training_argv


def test_absent_contract_means_single_node():
    group = NodeGroup.from_env({})

    assert group.nnodes == 1
    assert group.node_rank == 0
    assert group.master_addr is None
    assert group.master_port == 29400
    assert not group.is_multi_node


def test_full_contract_is_parsed_and_validated():
    group = NodeGroup.from_env(
        {
            "NVFL_NNODES": "4",
            "NVFL_NODE_RANK": "3",
            "NVFL_MASTER_ADDR": "node-0",
            "NVFL_MASTER_PORT": "29512",
            "NVFL_RUN_ID": "run-7",
        }
    )

    assert group == NodeGroup(nnodes=4, node_rank=3, master_addr="node-0", master_port=29512, run_id="run-7")
    assert group.is_multi_node


@pytest.mark.parametrize(
    "environ, message",
    [
        ({"NVFL_NNODES": "0"}, "NVFL_NNODES"),
        ({"NVFL_NNODES": "x"}, "NVFL_NNODES"),
        ({"NVFL_NNODES": "2", "NVFL_NODE_RANK": "2", "NVFL_MASTER_ADDR": "node-0"}, "NVFL_NODE_RANK"),
        ({"NVFL_NNODES": "2", "NVFL_NODE_RANK": "-1", "NVFL_MASTER_ADDR": "node-0"}, "NVFL_NODE_RANK"),
        ({"NVFL_NNODES": "2"}, "NVFL_MASTER_ADDR"),
        ({"NVFL_NNODES": "2", "NVFL_MASTER_ADDR": "node-0", "NVFL_MASTER_PORT": "abc"}, "NVFL_MASTER_PORT"),
    ],
)
def test_invalid_contract_is_rejected(environ, message):
    with pytest.raises(NodeGroupError, match=message):
        NodeGroup.from_env(environ)


def test_split_training_argv_at_boundary():
    options, training = split_training_argv(["--nproc-per-node=8", "--", "custom/client.py", "--epochs", "2"])

    assert options == ("--nproc-per-node=8",)
    assert training == ("custom/client.py", "--epochs", "2")


@pytest.mark.parametrize("argv, message", [(["custom/client.py"], "'--' is required"), (["--"], "training script")])
def test_split_training_argv_rejects_missing_boundary_or_script(argv, message):
    with pytest.raises(NodeGroupError, match=message):
        split_training_argv(argv)
