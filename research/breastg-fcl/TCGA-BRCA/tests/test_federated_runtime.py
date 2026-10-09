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

import os
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from federated.runtime import execute_client, operation_seed
from model.client import ModifiedClient
from model.server import Server


def make_opt():
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
        noise_dim=5,
        no_bn=False,
        p=0.2,
        lr_e=0.001,
        lr_f=0.002,
        lr_g=0.003,
        lr_d=0.004,
        beta1=0.9,
        beta2=0.999,
        lambda_gan=0.5,
        replay=True,
        b=0.0,
    )


class RNGProbeClient:
    device = "cpu"

    def generate_encodings(self, task, relational_graphs, dataloader, generate):
        return [random.random(), np.random.rand(), torch.rand(3)]


class FailingRNGProbeClient(RNGProbeClient):
    def generate_encodings(self, task, relational_graphs, dataloader, generate):
        super().generate_encodings(task, relational_graphs, dataloader, generate)
        raise RuntimeError("Failure after consuming all three RNGs")


class FederatedRuntimeTest(unittest.TestCase):
    def setUp(self):
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        self.opt = make_opt()
        self.client = ModifiedClient(0, self.opt)
        self.discriminator = Server(self.opt).get_discriminator()
        self.client.set_server_discriminator(self.discriminator)
        self.graphs = [np.roll(np.eye(4, dtype=np.float32), task, axis=1) for task in range(3)]
        self.loader = DataLoader(
            TensorDataset(torch.randn(7, 8), torch.tensor([0, 1, 1, 0, 0, 1, 0])),
            batch_size=4,
            shuffle=True,
        )
        for task in range(3):
            self.client.register_task(task, self.loader)

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

    def rng_state(self):
        return [random.getstate(), np.random.get_state(), torch.get_rng_state().clone()]

    def clone_client_snapshot(self):
        restored = ModifiedClient(self.client.getId(), self.opt)
        restored.set_server_discriminator(self.discriminator)
        restored.set_weights(self.client.get_weights())
        restored.set_training_state(self.client.get_training_state())
        return restored

    def test_fixed_seed_reproduces_encodings_including_all_historical_tasks(self):
        first = self.clone_client_snapshot()
        second = self.clone_client_snapshot()

        first_result = execute_client("encode", first, 2, self.graphs, self.loader, seed=177)
        torch.rand(11)
        np.random.rand(13)
        random.random()
        second_result = execute_client("encode", second, 2, self.graphs, self.loader, seed=177)

        self.assert_nested_equal(first_result, second_result)
        targets = torch.cat(first_result["graph_embeddings"])
        self.assertEqual(
            {
                task: (targets == torch.tensor(graph[0])).all(dim=1).sum().item()
                for task, graph in enumerate(self.graphs)
            },
            {0: 7, 1: 7, 2: 14},
        )

    def test_fixed_seed_reproduces_multiple_training_epochs_and_optimizer_state(self):
        first = self.clone_client_snapshot()
        second = self.clone_client_snapshot()

        first_result = execute_client("train", first, 2, self.graphs, self.loader, epochs=2, seed=281)
        torch.rand(17)
        second_result = execute_client("train", second, 2, self.graphs, self.loader, epochs=2, seed=281)

        self.assertEqual(set(first_result), {"encoder", "predictor", "generator", "training_state"})
        self.assert_nested_equal(first_result, second_result)
        self.assertTrue(first_result["training_state"]["optimizer"]["state"])
        self.assertEqual(first_result["training_state"]["scheduler"]["last_epoch"], 2)
        self.assertEqual(set(first_result["training_state"]["task_label_counts"]), {0, 1, 2})

    def test_training_results_restore_a_worker_and_continue_identically(self):
        first_result = execute_client("train", self.client, 2, self.graphs, self.loader, seed=181)
        restored = ModifiedClient(0, self.opt)
        restored.set_server_discriminator(self.discriminator)
        restored.set_weights(first_result)
        restored.set_training_state(first_result["training_state"])

        expected = execute_client("train", self.client, 2, self.graphs, self.loader, seed=182)
        actual = execute_client("train", restored, 2, self.graphs, self.loader, seed=182)

        self.assert_nested_equal(actual, expected)

    def test_every_operation_preserves_the_callers_random_state(self):
        for operation in ("encode", "train", "test"):
            with self.subTest(operation=operation):
                client = self.clone_client_snapshot()
                before = self.rng_state()
                execute_client(operation, client, 2, self.graphs, self.loader, seed=891)
                self.assert_nested_equal(self.rng_state(), before)

    def test_local_seed_controls_all_rngs_without_consuming_outer_state(self):
        before = self.rng_state()
        first = execute_client("encode", RNGProbeClient(), 0, [], None, seed=456)
        second = execute_client("encode", RNGProbeClient(), 0, [], None, seed=456)
        different = execute_client("encode", RNGProbeClient(), 0, [], None, seed=457)

        self.assert_nested_equal(first, second)
        self.assertNotEqual(first[0], different[0])
        self.assertNotEqual(first[1], different[1])
        self.assertFalse(torch.equal(first[2], different[2]))
        self.assert_nested_equal(self.rng_state(), before)

    def test_omitted_seed_and_failed_operations_also_restore_rng_state(self):
        before = self.rng_state()
        execute_client("encode", RNGProbeClient(), 0, [], None)
        self.assert_nested_equal(self.rng_state(), before)

        with self.assertRaisesRegex(RuntimeError, "Failure after consuming"):
            execute_client("encode", FailingRNGProbeClient(), 0, [], None, seed=219)
        self.assert_nested_equal(self.rng_state(), before)

    def test_operation_seeds_distinguish_round_task_client_and_operation(self):
        identities = [
            (41, task, round_index, client, operation)
            for task in range(3)
            for round_index in range(2)
            for client in range(4)
            for operation in ("encode", "train", "test")
        ]
        seeds = [operation_seed(*identity) for identity in identities]

        self.assertEqual(len(set(seeds)), len(identities))
        self.assertEqual(seeds, [operation_seed(*identity) for identity in identities])
        self.assertTrue(all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in seeds))
        self.assertNotEqual(operation_seed(41, 0, 0, 0, "train"), operation_seed(42, 0, 0, 0, "train"))

    def test_invalid_operations_and_epoch_counts_do_not_mutate_client_or_rng(self):
        before = self.client.get_weights()
        before_rng = self.rng_state()
        for operation, epochs in (("unknown", 1), ("train", 0), ("train", -1), ("train", 1.5)):
            with self.subTest(operation=operation, epochs=epochs):
                with self.assertRaises(ValueError):
                    execute_client(operation, self.client, 2, self.graphs, self.loader, epochs=epochs, seed=73)
                self.assert_nested_equal(self.client.get_weights(), before)
                self.assert_nested_equal(self.rng_state(), before_rng)
        with self.assertRaises(ValueError):
            operation_seed(41, 0, 0, 0, "unknown")


if __name__ == "__main__":
    unittest.main()
