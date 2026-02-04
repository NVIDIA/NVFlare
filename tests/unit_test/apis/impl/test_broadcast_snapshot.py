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

"""Unit tests for broadcast snapshot protection in WFCommServer."""

import numpy as np
import torch

from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.shareable import Shareable


class TestBroadcastSnapshot:
    """Test that broadcast creates snapshots to prevent in-place modification corruption."""

    def test_snapshot_protects_numpy_arrays(self):
        """Verify that broadcast snapshot protects numpy arrays from in-place modifications."""
        server = WFCommServer()

        # Create task with numpy data
        original_data = {"weights": np.array([1.0, 2.0, 3.0])}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=original_data)
        task = Task(name="train", data=dxo.to_shareable())

        # Create snapshot
        snapshot_task = server._create_broadcast_snapshot(task)

        # Modify original data in-place
        original_data["weights"][0] = 999.0

        # Verify snapshot is NOT affected
        snapshot_dxo = from_shareable(snapshot_task.data)
        snapshot_weights = snapshot_dxo.data["weights"]

        assert snapshot_weights[0] == 1.0, "Snapshot should not be affected by in-place modifications"
        assert original_data["weights"][0] == 999.0, "Original data should be modified"

    def test_snapshot_protects_torch_tensors(self):
        """Verify that broadcast snapshot protects PyTorch tensors from in-place modifications."""
        server = WFCommServer()

        # Create task with PyTorch tensor data
        original_tensor = torch.tensor([1.0, 2.0, 3.0])
        original_data = {"weights": original_tensor}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=original_data)
        task = Task(name="train", data=dxo.to_shareable())

        # Create snapshot
        snapshot_task = server._create_broadcast_snapshot(task)

        # Modify original tensor in-place
        original_tensor.add_(100.0)

        # Verify snapshot is NOT affected
        snapshot_dxo = from_shareable(snapshot_task.data)
        snapshot_weights = snapshot_dxo.data["weights"]

        assert snapshot_weights[0].item() == 1.0, "Snapshot should not be affected by in-place modifications"
        assert original_tensor[0].item() == 101.0, "Original tensor should be modified"

    def test_snapshot_creates_independent_memory(self):
        """Verify that snapshot creates completely independent memory."""
        server = WFCommServer()

        # Create task with nested data
        original_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        original_data = {"layer1": original_array, "layer2": np.array([5.0, 6.0])}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=original_data)
        task = Task(name="train", data=dxo.to_shareable())

        # Create snapshot
        snapshot_task = server._create_broadcast_snapshot(task)

        # Get snapshot data
        snapshot_dxo = from_shareable(snapshot_task.data)
        snapshot_data = snapshot_dxo.data

        # Verify different memory addresses
        assert id(original_data) != id(snapshot_data), "Data dicts should have different IDs"
        assert id(original_data["layer1"]) != id(snapshot_data["layer1"]), "Arrays should have different IDs"
        assert not np.shares_memory(original_data["layer1"], snapshot_data["layer1"]), "Arrays should not share memory"

    def test_snapshot_protects_nested_structures(self):
        """Verify that snapshot protects deeply nested data structures."""
        server = WFCommServer()

        # Create task with deeply nested data
        nested_array = np.array([1.0, 2.0])
        original_data = {"model": {"layers": {"conv1": nested_array, "conv2": np.array([3.0, 4.0])}}}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=original_data)
        task = Task(name="train", data=dxo.to_shareable())

        # Create snapshot
        snapshot_task = server._create_broadcast_snapshot(task)

        # Modify deeply nested original data
        nested_array[0] = 999.0

        # Verify snapshot is NOT affected
        snapshot_dxo = from_shareable(snapshot_task.data)
        snapshot_nested = snapshot_dxo.data["model"]["layers"]["conv1"]

        assert snapshot_nested[0] == 1.0, "Deeply nested snapshot should not be affected"
        assert nested_array[0] == 999.0, "Original nested data should be modified"

    def test_snapshot_works_with_non_dxo_shareable(self):
        """Verify snapshot works with Shareable that doesn't contain DXO."""
        server = WFCommServer()

        # Create task with plain Shareable (no DXO)
        data = Shareable()
        data["model_weights"] = np.array([1.0, 2.0, 3.0])
        data["metadata"] = {"round": 1}
        task = Task(name="train", data=data)

        # Create snapshot
        snapshot_task = server._create_broadcast_snapshot(task)

        # Modify original
        data["model_weights"][0] = 999.0

        # Verify snapshot is NOT affected
        assert snapshot_task.data["model_weights"][0] == 1.0, "Snapshot should not be affected"
        assert data["model_weights"][0] == 999.0, "Original should be modified"

    def test_snapshot_preserves_shareable_headers(self):
        """Verify that snapshot preserves Shareable headers and cookies."""
        server = WFCommServer()

        # Create task with headers and cookies
        original_data = {"weights": np.array([1.0, 2.0, 3.0])}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=original_data)
        shareable = dxo.to_shareable()
        shareable.set_header("round", 5)
        shareable.add_cookie("client_id", "client_1")
        task = Task(name="train", data=shareable)

        # Create snapshot
        snapshot_task = server._create_broadcast_snapshot(task)

        # Verify headers and cookies are preserved
        assert snapshot_task.data.get_header("round") == 5, "Headers should be preserved"
        assert snapshot_task.data.get_cookie("client_id") == "client_1", "Cookies should be preserved"

    def test_snapshot_handles_all_data_kinds(self):
        """Verify snapshot works with all DataKind types."""
        server = WFCommServer()

        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF, DataKind.METRICS]

        for kind in data_kinds:
            # Create task
            original_data = {"data": np.array([1.0, 2.0, 3.0])}
            dxo = DXO(data_kind=kind, data=original_data)
            task = Task(name="train", data=dxo.to_shareable())

            # Create snapshot
            snapshot_task = server._create_broadcast_snapshot(task)

            # Modify original
            original_data["data"][0] = 999.0

            # Verify snapshot is NOT affected
            snapshot_dxo = from_shareable(snapshot_task.data)
            assert snapshot_dxo.data["data"][0] == 1.0, f"Snapshot should not be affected for {kind}"
