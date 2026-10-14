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

"""Latent perturbation must happen before a client upload reaches the server."""

import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from breastgfcl import ParallelServerGFedCL
from federated import client as collab_client
from federated.runtime import execute_client, seeded_rng, snapshot_rng
from model.client import ModifiedClient
from model.server import Server
from utils import privacy_utils


def make_opt(scale=0.5):
    return SimpleNamespace(
        device="cpu",
        batch_size=4,
        input_dim=8,
        nh=16,
        ni=16,
        nt=4,
        nd_out=4,
        num_clients=4,
        num_classes=2,
        num_task=3,
        noise_dim=5,
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
        shuffle=False,
        seed=71,
        b=scale,
    )


class LatentPrivacyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(107)
        self.opt = make_opt()
        self.inputs = torch.randn(6, 8)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1])
        self.loader = DataLoader(TensorDataset(self.inputs, self.labels), batch_size=4, shuffle=False)
        self.graphs = [np.roll(np.eye(4, dtype=np.float32), task, axis=1) for task in range(3)]
        self.client = ModifiedClient(0, self.opt)
        self.discriminator = Server(self.opt).get_discriminator()
        self.client.set_server_discriminator(self.discriminator)
        for task in range(3):
            self.client.register_task(task, self.loader)

    def clone(self, scale=0.5):
        opt = copy.copy(self.opt)
        opt.b = scale
        client = ModifiedClient(0, opt)
        client.set_weights(self.client.get_weights())
        client.set_training_state(self.client.get_training_state())
        client.set_server_discriminator(self.discriminator)
        return client

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_model_perturbs_real_current_generated_and_historical_batches_once(self):
        clean_client = self.clone(0.0)
        noisy_client = self.clone(0.5)
        with seeded_rng(51):
            clean = clean_client.generate_encodings(2, self.graphs, self.loader)
        draws = []

        def draw(location, scale, size):
            self.assertEqual(location, 0)
            self.assertEqual(scale, 0.5)
            offset = (len(draws) + 1) * 0.125
            draws.append((size, offset))
            return np.full(size, offset, dtype=np.float64)

        with patch.object(privacy_utils.np.random, "laplace", side_effect=draw), seeded_rng(51):
            noisy = noisy_client.generate_encodings(2, self.graphs, self.loader)
        self.assertEqual(len(draws), 8)  # Two batches times E + G(task 0, 1, 2).
        self.assertEqual(len(noisy["encodings"]), len(draws))
        for original, actual, (size, offset), graph_rows in zip(
            clean["encodings"], noisy["encodings"], draws, noisy["graph_embeddings"]
        ):
            self.assertEqual(tuple(size), tuple(original.shape))
            self.assertEqual(actual.shape[0], graph_rows.shape[0])
            torch.testing.assert_close(actual, original + offset, rtol=0, atol=0)
            self.assertFalse(actual.requires_grad)
        self.assert_nested_equal(noisy["graph_embeddings"], clean["graph_embeddings"])
        expected_tasks = [2, 0, 1, 2] * 2
        for rows, task in zip(noisy["graph_embeddings"], expected_tasks):
            torch.testing.assert_close(rows, torch.tensor(self.graphs[task][0]).expand(len(rows), -1))

    def test_synthetic_only_replay_upload_is_noised_without_reading_historical_inputs(self):
        unreadable = Mock()
        unreadable.__iter__ = Mock(side_effect=AssertionError("Historical data must not be read"))
        clean_client, noisy_client = self.clone(0), self.clone(0.5)
        with seeded_rng(72):
            clean = clean_client.generate_encodings(0, self.graphs, unreadable, generate=True)
        with (
            patch.object(
                privacy_utils.np.random, "laplace", side_effect=lambda _loc, _scale, size: np.full(size, 0.25)
            ) as draw,
            seeded_rng(72),
        ):
            actual = noisy_client.generate_encodings(0, self.graphs, unreadable, generate=True)
        self.assertEqual(draw.call_count, 2)
        for original, noised in zip(clean["encodings"], actual["encodings"]):
            torch.testing.assert_close(noised, original + 0.25, rtol=0, atol=0)
        self.assert_nested_equal(actual["graph_embeddings"], clean["graph_embeddings"])
        unreadable.__iter__.assert_not_called()

    def test_runtime_returns_noised_latents_reproducibly_and_restores_rng(self):
        first_client, second_client, clean_client = self.clone(), self.clone(), self.clone(0)
        before = snapshot_rng()
        first = execute_client("encode", first_client, 2, self.graphs, self.loader, seed=117)
        self.assert_nested_equal(snapshot_rng(), before)
        second = execute_client("encode", second_client, 2, self.graphs, self.loader, seed=117)
        self.assert_nested_equal(first, second)
        clean = execute_client("encode", clean_client, 2, self.graphs, self.loader, seed=117)
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(first["encodings"], clean["encodings"])))
        self.assert_nested_equal(first["graph_embeddings"], clean["graph_embeddings"])
        another = execute_client("encode", self.clone(), 2, self.graphs, self.loader, seed=118)
        self.assertFalse(torch.equal(first["encodings"][0], another["encodings"][0]))

    def test_noise_failure_restores_rng_and_cannot_return_a_partial_upload(self):
        client = self.clone()
        before = snapshot_rng()
        with patch.object(privacy_utils.np.random, "laplace", side_effect=ValueError("Noise sampler failed")):
            with self.assertRaisesRegex(ValueError, "Noise sampler failed"):
                execute_client("encode", client, 2, self.graphs, self.loader, seed=117)
        self.assert_nested_equal(snapshot_rng(), before)

    def test_collab_encode_already_contains_the_same_noised_payload_as_client_runtime(self):
        bundle = {
            "opt": vars(self.opt),
            "client_id": 0,
            "data": {
                task: {split: {"x": self.inputs, "y": self.labels} for split in ("train", "test")} for task in range(3)
            },
        }
        request = dict(
            task=2,
            graphs=self.graphs,
            weights=self.client.get_weights(),
            training_state=self.client.get_training_state(),
            discriminator=self.discriminator,
            seed=317,
        )
        expected = execute_client("encode", self.clone(), 2, self.graphs, self.loader, seed=317)
        clean = execute_client("encode", self.clone(0), 2, self.graphs, self.loader, seed=317)
        with (
            patch.object(collab_client, "collab", SimpleNamespace(site_name="site-1", is_aborted=False)),
            patch.object(collab_client, "load_bundle", return_value=bundle),
        ):
            worker = collab_client.BreastGFCLClient()
            before = snapshot_rng()
            worker.initialize()
            actual = worker.encode(**request)
            self.assert_nested_equal(snapshot_rng(), before)
        self.assert_nested_equal(actual, expected)
        self.assertFalse(torch.equal(actual["encodings"][0], clean["encodings"][0]))

    def test_local_training_and_evaluation_do_not_apply_upload_noise(self):
        with patch.object(
            privacy_utils.np.random, "laplace", side_effect=AssertionError("Upload noise in local step")
        ) as draw:
            self.client.learn(0, 2, self.graphs, self.loader)
            self.client.test(2, self.loader, self.graphs)
        draw.assert_not_called()

    def test_coordinator_consumes_uploaded_latents_without_second_noise(self):
        opt = copy.copy(self.opt)
        opt.num_task, opt.num_rounds, opt.num_local_epochs, opt.max_in_flight = 1, 1, 1, 4
        server = Server(opt)
        clients = []
        uploaded = []
        for client_id in range(4):
            client = Mock()
            client.getId.return_value = client_id
            clients.append(client)
            uploaded.append(
                {
                    "encodings": [torch.full((2, 16), 0.375 + client_id)],
                    "graph_embeddings": [torch.tensor(self.graphs[0][client_id]).expand(2, -1).clone()],
                }
            )
        transport = Mock()
        transport.generate_encodings.side_effect = lambda client, *_args: uploaded[client.getId()]
        transport.gather.side_effect = lambda results: results
        transport.train_client.side_effect = RuntimeError("Stop after discriminator consumes upload")
        loaders = {i: {0: {"train": None, "test": None}} for i in range(4)}
        workflow = ParallelServerGFedCL.from_components(opt, server, Mock(), clients, loaders, transport)
        with (
            patch.object(workflow, "_generate_relational_graph", return_value=self.graphs[0]),
            patch.object(
                privacy_utils.np.random, "laplace", side_effect=AssertionError("Server applied latent noise")
            ) as draw,
            patch.object(server, "train_discriminator", wraps=server.train_discriminator) as train,
        ):
            with self.assertRaisesRegex(RuntimeError, "Stop after discriminator"):
                workflow.train_GFedCL()
        draw.assert_not_called()
        train.assert_called_once()
        self.assert_nested_equal(train.call_args.args[0], [entry["encodings"][0] for entry in uploaded])
        self.assert_nested_equal(train.call_args.args[1], [entry["graph_embeddings"][0] for entry in uploaded])
        self.assertTrue(server.optimizer_D.state_dict()["state"])


class LaplaceHelperTest(unittest.TestCase):
    def test_tensor_dtype_device_and_float32_noise_draw_are_preserved(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                values = torch.tensor([[1, 2], [3, 4]], dtype=dtype).T
                before = values.clone()
                noise = np.asarray([[0.123456789, -0.2], [0.3, -0.4]], dtype=np.float64)
                with patch.object(privacy_utils.np.random, "laplace", return_value=noise) as draw:
                    actual = privacy_utils.add_laplace_noise(values, 0.5)
                draw.assert_called_once_with(0, 0.5, values.shape)
                expected = values + torch.as_tensor(noise.astype(np.float32), dtype=dtype, device=values.device)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, values.device)
                torch.testing.assert_close(values, before, rtol=0, atol=0)

    def test_numpy_values_keep_their_dtype_without_inplace_modification(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                values = np.asarray([[1, 2]], dtype=dtype)
                with patch.object(privacy_utils.np.random, "laplace", return_value=np.asarray([[0.25, -0.5]])):
                    actual = privacy_utils.add_laplace_noise(values, 0.5)
                self.assertEqual(actual.dtype, values.dtype)
                np.testing.assert_array_equal(actual, [[1.25, 1.5]])
                np.testing.assert_array_equal(values, [[1, 2]])

    def test_zero_scale_keeps_values_and_random_state_unchanged(self):
        for values in (np.ones((2, 3), dtype=np.float32), torch.ones(2, 3)):
            with self.subTest(values=type(values).__name__), patch.object(privacy_utils.np.random, "laplace") as draw:
                actual = privacy_utils.add_laplace_noise(values, 0)
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(values))
                self.assertIsNot(actual, values)
                self.assertFalse(np.shares_memory(np.asarray(actual), np.asarray(values)))
                draw.assert_not_called()

    def test_invalid_scale_is_rejected_before_a_random_draw(self):
        for scale in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(scale=scale), patch.object(privacy_utils.np.random, "laplace") as draw:
                with self.assertRaises(ValueError):
                    privacy_utils.add_laplace_noise(torch.ones(2, 3), scale)
                draw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
