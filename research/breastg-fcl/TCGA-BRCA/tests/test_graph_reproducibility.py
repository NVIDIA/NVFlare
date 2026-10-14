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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Graph normalization and evaluation reproducibility across saved artifacts."""

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from breastgfcl import ParallelServerGFedCL, add_laplace_noise_to_graph
from federated.runtime import execute_client, operation_seed
from federated.state import capture_final_state
from model.client import ModifiedClient


class GraphNoiseTest(unittest.TestCase):
    def test_clipped_empty_rows_become_uniform_without_changing_nonempty_ratios(self):
        graph = np.full((3, 3), 1 / 3, dtype=np.float32)
        original = graph.copy()
        noisy = np.asarray([[-1, -2, -3], [-1, 2, 6], [0, 0, 0]], dtype=np.float32)
        with patch("breastgfcl.add_laplace_noise", return_value=noisy):
            actual = add_laplace_noise_to_graph(graph, 1.0)
        expected = np.asarray([[1 / 3] * 3, [0, 0.25, 0.75], [1 / 3] * 3], dtype=np.float32)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual.sum(axis=1), np.ones(3))
        np.testing.assert_array_equal(graph, original)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())

    def test_tiny_positive_rows_are_normalized_without_a_probability_floor(self):
        graph = np.asarray([[1e-12, 3e-12], [0, 1e-12]], dtype=np.float32)
        actual = add_laplace_noise_to_graph(graph, 0.0)
        np.testing.assert_allclose(actual, [[0.25, 0.75], [0, 1]])
        np.testing.assert_allclose(actual.sum(axis=1), np.ones(2))

    def test_single_client_empty_row_has_unit_weight(self):
        with patch("breastgfcl.add_laplace_noise", return_value=np.asarray([[-2.0]], dtype=np.float32)):
            actual = add_laplace_noise_to_graph(np.ones((1, 1), dtype=np.float32), 1.0)
        np.testing.assert_array_equal(actual, [[1.0]])

    def test_disabled_normalization_only_clips_negative_entries(self):
        noisy = np.asarray([[-2, -1], [2, 6]], dtype=np.float32)
        with patch("breastgfcl.add_laplace_noise", return_value=noisy):
            actual = add_laplace_noise_to_graph(np.eye(2, dtype=np.float32), 1.0, normalize=False)
        np.testing.assert_array_equal(actual, [[0, 0], [2, 6]])

    def test_zero_noise_preserves_valid_graph_and_seeded_noise_is_reproducible(self):
        graph = np.asarray([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32)
        original = graph.copy()
        np.testing.assert_array_equal(add_laplace_noise_to_graph(graph, 0.0), graph)
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(57)
        first = add_laplace_noise_to_graph(graph, 1.0)
        np.random.seed(57)
        second = add_laplace_noise_to_graph(graph, 1.0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.sum(axis=1), np.ones(2))
        np.testing.assert_array_equal(graph, original)


class LocalRecordingTransport:
    """Run the shared client operations locally while recording dispatched graphs."""

    def __init__(self, opt):
        self.opt = opt
        self.round_index = 0
        self.seen = []

    def set_round(self, task, round_index):
        self.round_index = round_index

    def submit(self, operation, client, task, graphs, loader, epochs=1):
        self.seen.append((operation, task, [None if graph is None else graph.copy() for graph in graphs]))
        return execute_client(
            operation,
            client,
            task,
            graphs,
            loader,
            epochs=epochs,
            seed=operation_seed(self.opt.seed, task, self.round_index, client.getId(), operation),
        )

    def generate_encodings(self, client, task, graphs, loader):
        return self.submit("encode", client, task, graphs, loader)

    def train_client(self, client, task, graphs, loader, epochs):
        return self.submit("train", client, task, graphs, loader, epochs)

    def test_client(self, client, task, loader, graphs):
        return self.submit("test", client, task, graphs, loader)

    def gather(self, pending):
        return pending


class GraphArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def workflow(self, scale):
        output = Path(self.directory.name) / str(scale)
        output.mkdir(exist_ok=True)
        opt = SimpleNamespace(
            output_dir=str(output),
            seed=71,
            device="cpu",
            b=scale,
            num_clients=2,
            num_task=2,
            num_classes=2,
            num_rounds=1,
            num_local_epochs=1,
            batch_size=3,
            input_dim=4,
            nh=8,
            nt=2,
            noise_dim=3,
            no_bn=False,
            p=0.0,
            lr_e=0.001,
            lr_f=0.002,
            lr_g=0.003,
            lr_d=0.004,
            beta1=0.9,
            beta2=0.999,
            lambda_gan=0.5,
            replay=True,
            max_in_flight=2,
            temporal_window=2,
            attention_temperature=1.0,
            graph_epsilon=1e-8,
            gat_hidden_dim=8,
            gat_embedding_dim=4,
            gat_heads=2,
            gat_dropout=0.0,
            gat_epochs=0,
            client_spatial_features=[np.asarray([[1, 2], [3, 1]], dtype=np.float32) for _ in range(2)],
            client_temporal_features=[
                np.asarray([[1 + task, 2], [3, 1 + task]], dtype=np.float32) for task in range(2)
            ],
        )
        loaders = {}
        for client in range(opt.num_clients):
            loaders[client] = {}
            for task in range(opt.num_task):
                inputs = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20 + client + task
                labels = (torch.arange(5) + client + task) % 2
                loaders[client][task] = {
                    split: DataLoader(TensorDataset(inputs, labels), batch_size=3, shuffle=False)
                    for split in ("train", "test")
                }
        transport = LocalRecordingTransport(opt)
        return ParallelServerGFedCL(opt, transport=transport, dataloaders=loaders)

    def test_each_task_saves_the_exact_graph_returned_with_and_without_noise(self):
        for scale in (0.0, 1.0):
            with self.subTest(scale=scale):
                workflow = self.workflow(scale)
                graph_dir = Path(workflow.opt.output_dir) / "relational_graphs"
                for task in range(workflow.opt.num_task):
                    actual = workflow._generate_relational_graph(task)
                    saved = np.load(graph_dir / f"task_{task + 1}_private.npy", allow_pickle=False)
                    clean = np.load(graph_dir / f"task_{task + 1}_fused.npy", allow_pickle=False)
                    np.testing.assert_array_equal(saved, actual)
                    np.testing.assert_allclose(saved.sum(axis=1), np.ones(2), atol=1e-7)
                    if scale:
                        self.assertFalse(np.array_equal(clean, saved))
                    else:
                        np.testing.assert_allclose(saved, clean, atol=1e-7)

    def test_saved_models_and_task_graphs_restore_predictions_metrics_and_replay_inputs(self):
        workflow = self.workflow(1.0)
        metrics, rounds, _ = workflow.train_GFedCL()
        snapshot = capture_final_state(workflow, metrics, rounds)
        self.assertIn("relational_graphs", snapshot)
        path = Path(workflow.opt.output_dir) / "final_state.pt"
        torch.save(snapshot, path)
        # This is a trusted, locally written checkpoint, not an RPC payload.
        restored_state = torch.load(path, map_location="cpu", weights_only=False)
        graphs = restored_state["relational_graphs"]
        self.assertEqual(len(graphs), workflow.opt.num_task)
        for operation, task, sent in workflow.transport.seen:
            with self.subTest(operation=operation, task=task):
                for index, graph in enumerate(sent):
                    if graph is not None:
                        np.testing.assert_array_equal(graph, graphs[index])
        for client_id, client in enumerate(workflow.clients):
            restored = ModifiedClient(client_id, workflow.opt)
            restored.set_weights(restored_state["weights"])
            restored.set_training_state(restored_state["training_states"][client_id])
            restored.eval()
            client.eval()
            for task in range(workflow.opt.num_task):
                with self.subTest(client_id=client_id, task=task):
                    saved_file = np.load(
                        path.parent / "relational_graphs" / f"task_{task + 1}_private.npy", allow_pickle=False
                    )
                    np.testing.assert_array_equal(saved_file, graphs[task])
                    loader = workflow.dataloaders[client_id][task]["test"]
                    expected = client.test(task, loader, workflow.relational_graphs)
                    actual = restored.test(task, loader, graphs)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual["acc"], restored_state["metrics"]["client_task_acc"][client_id][task])
                    inputs, labels = next(iter(loader))
                    with torch.no_grad():
                        row = client._graph_row(workflow.relational_graphs, task, len(labels))
                        restored_row = restored._graph_row(graphs, task, len(labels))
                        expected_logits = client.netF(client.netE(inputs, row))
                        actual_logits = restored.netF(restored.netE(inputs, restored_row))
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
                    torch.testing.assert_close(restored.task_label_counts[task], client.task_label_counts[task])
        original = snapshot["relational_graphs"][0].copy()
        workflow.relational_graphs[0].fill(-1)
        np.testing.assert_array_equal(snapshot["relational_graphs"][0], original)
        np.testing.assert_array_equal(graphs[0], original)


if __name__ == "__main__":
    unittest.main()
