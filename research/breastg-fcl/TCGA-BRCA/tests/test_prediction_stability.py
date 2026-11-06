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

import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.modules import PredNet


class PredictionStabilityTest(unittest.TestCase):
    def setUp(self):
        self.predictor = PredNet(SimpleNamespace(nh=4, num_classes=2))
        # Isolate the public prediction interface from the learned logit mapping.
        self.predictor.net = torch.nn.Identity()

    def test_extreme_wrong_prediction_has_finite_unclipped_loss(self):
        logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
        log_probs, probabilities = self.predictor(logits, return_softmax=True)
        loss = F.nll_loss(log_probs, torch.tensor([1, 0]))

        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 2000.0)
        torch.testing.assert_close(probabilities, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def test_extreme_wrong_prediction_retains_corrective_gradient(self):
        logits = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
        loss = F.nll_loss(self.predictor(logits), torch.tensor([1]))
        loss.backward()

        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad, torch.tensor([[1.0, -1.0]]))

    def test_normal_logits_preserve_probability_contract(self):
        logits = torch.tensor([[1.0, 2.0], [-2.0, 0.5]], dtype=torch.float64)
        original = logits.clone()
        log_probs, probabilities = self.predictor(logits, return_softmax=True)
        default_result = self.predictor(logits)

        torch.testing.assert_close(log_probs.exp(), probabilities)
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(probabilities, F.softmax(logits, dim=-1))
        torch.testing.assert_close(default_result, log_probs)
        torch.testing.assert_close(logits, original)

    def test_leading_batch_dimensions_are_restored(self):
        for shape in [(3, 2), (2, 3, 2), (2, 3, 4, 2)]:
            with self.subTest(shape=shape):
                logits = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
                log_probs, probabilities = self.predictor(logits, return_softmax=True)

                self.assertEqual(log_probs.shape, logits.shape)
                self.assertEqual(probabilities.shape, logits.shape)
                torch.testing.assert_close(log_probs, F.log_softmax(logits, dim=-1))
                torch.testing.assert_close(probabilities, F.softmax(logits, dim=-1))
                torch.testing.assert_close(self.predictor(logits), log_probs)


if __name__ == "__main__":
    unittest.main()
