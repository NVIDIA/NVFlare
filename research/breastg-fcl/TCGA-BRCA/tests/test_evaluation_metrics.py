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

"""Evaluation must reject empty data and weight metrics by sample count."""

import math
import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.client import ModifiedClient


class IdentityEncoder(nn.Module):
    """Pass controlled predictions through the real client metric path."""

    def forward(self, inputs, graph_row):
        return inputs


class EvaluationMetricsTest(unittest.TestCase):
    def setUp(self):
        opt = SimpleNamespace(
            device="cpu",
            batch_size=3,
            input_dim=2,
            nh=8,
            ni=8,
            nt=4,
            nd_out=4,
            num_clients=3,
            num_classes=2,
            noise_dim=5,
            no_bn=True,
            p=0.0,
            lr_e=0.001,
            lr_f=0.002,
            lr_g=0.003,
            beta1=0.9,
            beta2=0.999,
        )
        self.client = ModifiedClient(2, opt)
        self.graphs = [torch.eye(opt.num_clients) for _ in range(4)]

    def test_empty_dataloader_raises_with_client_and_task(self):
        dataset = TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=3)

        with self.assertRaisesRegex(ValueError, r"Client 2: task 3 has no evaluation samples"):
            self.client.test(3, loader, self.graphs)

    def test_empty_iterator_raises_with_client_and_task(self):
        with self.assertRaisesRegex(ValueError, r"Client 2: task 1 has no evaluation samples"):
            self.client.test(1, iter(()), self.graphs)

    def test_metrics_are_sample_weighted_across_unequal_batches(self):
        probabilities = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]])
        labels = torch.tensor([0, 0, 0, 1, 0])
        dataset = TensorDataset(probabilities.log(), labels)
        self.client.netE = IdentityEncoder()
        self.client.netF = nn.Identity()
        expected_loss = -sum(math.log(value) for value in (0.9, 0.8, 0.7, 0.4, 0.1)) / 5

        for batch_size in (1, 3, 5):
            with self.subTest(batch_size=batch_size):
                result = self.client.test(3, DataLoader(dataset, batch_size=batch_size), self.graphs)
                self.assertAlmostEqual(result["loss"], expected_loss, places=6)
                self.assertEqual(result["acc"], 60.0)


if __name__ == "__main__":
    unittest.main()
