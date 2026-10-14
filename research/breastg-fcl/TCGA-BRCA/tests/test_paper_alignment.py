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

import copy
import io
import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

TEST_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.modules import AdditiveAttentionHead, BreastGraphGenerator, TemporalAttention


def make_opt(**overrides):
    values = {
        "num_clients": 3,
        "temporal_window": 2,
        "attention_temperature": 1.0,
        "graph_epsilon": 1e-8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BreastGraphGeneratorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.spatial = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        self.temporal = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]], dtype=np.float32)

    def make_generator(self, **overrides):
        return BreastGraphGenerator(make_opt(**overrides), spatial_dim=2, temporal_dim=2)

    def learn(self, generator, task_id=0, temporal=None, spatial=None):
        return generator.learn(
            0,
            task_id=task_id,
            spatial_features=self.spatial if spatial is None else spatial,
            temporal_features=self.temporal if temporal is None else temporal,
        )

    def test_temporal_attention_uses_scaled_query_key_scores(self):
        attention = TemporalAttention(input_dim=2, key_dim=2, temperature=0.7)
        with torch.no_grad():
            attention.query.weight.copy_(torch.tensor([[1.0, 2.0], [0.0, 1.0]]))
            attention.key.weight.copy_(torch.tensor([[0.5, 0.0], [1.0, -1.0]]))
        summaries = torch.tensor(self.temporal)
        queries = summaries @ attention.query.weight.T
        keys = summaries @ attention.key.weight.T
        expected = torch.softmax(queries @ keys.T / (math.sqrt(2) * 0.7), dim=1)

        actual = attention(summaries)

        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, actual.T))

    def test_spatial_head_uses_additive_leaky_relu_scores(self):
        attention = AdditiveAttentionHead(input_dim=2, head_dim=1, temperature=0.5)
        with torch.no_grad():
            attention.W.copy_(torch.tensor([[1.0], [1.0]]))
            attention.a.copy_(torch.tensor([[1.0], [-2.0]]))
        # Projected client values are 0, 1, 2. The score is LeakyReLU(h_i - 2 h_j).
        scores = torch.tensor([[0.0, -0.4, -0.8], [1.0, -0.2, -0.6], [2.0, 0.0, -0.4]])
        expected = torch.softmax(scores / 0.5, dim=1)

        actual = attention(torch.tensor(self.spatial))

        torch.testing.assert_close(actual, expected)

    def test_seed_and_inference_are_reproducible_and_restore_mode(self):
        first = self.make_generator()
        torch.manual_seed(17)
        second = self.make_generator()
        self.assertTrue(first.training)

        first_graph = self.learn(first)
        second_graph = self.learn(second)
        repeated_graph = self.learn(first)

        np.testing.assert_array_equal(first_graph, second_graph)
        np.testing.assert_array_equal(first_graph, repeated_graph)
        self.assertTrue(first.training)
        first.eval()
        np.testing.assert_array_equal(first_graph, self.learn(first))
        self.assertFalse(first.training)

    def test_attention_and_multiplicative_fusion_follow_paper(self):
        generator = self.make_generator(graph_epsilon=0.01)

        graph = self.learn(generator)

        self.assertEqual(graph.shape, (3, 3))
        self.assertEqual(graph.dtype, np.float32)
        for attention in (
            generator.last_spatial_attention,
            generator.last_temporal_patterns,
        ):
            self.assertTrue(np.isfinite(attention).all())
            self.assertTrue(np.all(attention >= 0))
            np.testing.assert_allclose(attention.sum(axis=1), np.ones(3), atol=1e-6)
        product = generator.last_spatial_attention * generator.last_temporal_patterns
        expected = product / (product.sum(axis=1, keepdims=True) + 0.01)
        np.testing.assert_allclose(graph, expected, atol=1e-7)

    def test_forward_has_gradients_and_allows_external_optimization(self):
        generator = self.make_generator().eval()
        spatial = torch.tensor(self.spatial, requires_grad=True)
        temporal = torch.tensor(self.temporal, requires_grad=True)
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
        original = {name: parameter.detach().clone() for name, parameter in generator.named_parameters()}

        graph = generator(spatial, temporal, task_id=0)
        weights = torch.tensor([[0.1, 0.7, -0.2], [0.4, -0.3, 0.8], [0.9, 0.2, -0.5]])
        loss = (graph * weights).sum()
        loss.backward()

        self.assertTrue(graph.requires_grad)
        for inputs in (spatial, temporal):
            self.assertIsNotNone(inputs.grad)
            self.assertTrue(torch.isfinite(inputs.grad).all())
            self.assertGreater(inputs.grad.abs().sum().item(), 0)
        for prefix in ("spatial_attention.", "temporal_attention."):
            gradients = [
                parameter.grad
                for name, parameter in generator.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
        optimizer.step()
        for prefix in ("spatial_attention.", "temporal_attention."):
            self.assertTrue(
                any(
                    not torch.equal(original[name], parameter.detach())
                    for name, parameter in generator.named_parameters()
                    if name.startswith(prefix)
                ),
                prefix,
            )

    def test_temporal_window_slides_and_excludes_future_tasks(self):
        generator = self.make_generator()
        temporal_1 = self.temporal + np.asarray([[0.2, 0.0], [0.0, 0.1], [0.1, 0.2]], dtype=np.float32)
        temporal_2 = self.temporal + np.asarray([[0.4, 0.3], [0.2, 0.5], [0.6, 0.1]], dtype=np.float32)
        first_graph = self.learn(generator)
        np.testing.assert_array_equal(generator.last_temporal_window, self.temporal)
        second_graph = self.learn(generator, task_id=1, temporal=temporal_1)
        np.testing.assert_array_equal(
            generator.last_temporal_window, np.concatenate([self.temporal, temporal_1], axis=1)
        )
        self.learn(generator, task_id=2, temporal=temporal_2)
        np.testing.assert_array_equal(generator.last_temporal_window, np.concatenate([temporal_1, temporal_2], axis=1))

        np.testing.assert_array_equal(second_graph, self.learn(generator, task_id=1, temporal=temporal_1))
        np.testing.assert_array_equal(first_graph, self.learn(generator))
        self.assertEqual(len(generator.temporal_history), 3)

    def test_short_windows_are_left_padded_for_fixed_temporal_network(self):
        generator = self.make_generator()
        observed = []
        hook = generator.temporal_attention.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0].detach().cpu().numpy().copy())
        )
        try:
            self.learn(generator)
            self.learn(generator, task_id=1, temporal=self.temporal + 1)
        finally:
            hook.remove()

        standardized = (self.temporal - self.temporal.mean(axis=0)) / self.temporal.std(axis=0)
        np.testing.assert_allclose(
            observed[0],
            np.concatenate([np.zeros_like(self.temporal), standardized], axis=1),
            atol=1e-6,
        )
        np.testing.assert_allclose(observed[1], np.concatenate([standardized, standardized], axis=1), atol=1e-6)

    def test_history_does_not_retain_gradients_or_alias_input(self):
        generator = self.make_generator().eval()
        first_temporal = torch.tensor(self.temporal, requires_grad=True)
        generator(torch.tensor(self.spatial), first_temporal, task_id=0)
        second_temporal = torch.tensor(self.temporal + 1, requires_grad=True)
        graph = generator(torch.tensor(self.spatial), second_temporal, task_id=1)
        (graph * torch.arange(9).reshape(3, 3)).sum().backward()

        self.assertIsNone(first_temporal.grad)
        self.assertIsNotNone(second_temporal.grad)
        with torch.no_grad():
            first_temporal.fill_(99)
        self.learn(generator, task_id=1, temporal=self.temporal + 1)
        np.testing.assert_array_equal(generator.last_temporal_window[:, :2], self.temporal)

    def test_single_client_and_constant_summaries_are_finite(self):
        for count in (1, 3):
            with self.subTest(num_clients=count):
                generator = self.make_generator(num_clients=count)
                graph = self.learn(
                    generator,
                    spatial=np.ones((count, 2), dtype=np.float32),
                    temporal=np.ones((count, 2), dtype=np.float32),
                )
                self.assertEqual(graph.shape, (count, count))
                self.assertTrue(np.isfinite(graph).all())
                np.testing.assert_allclose(graph, np.full((count, count), 1 / count), atol=1e-6)

    def test_invalid_summaries_do_not_modify_history(self):
        generator = self.make_generator()
        self.learn(generator)
        previous_history = copy.deepcopy(generator.temporal_history)
        invalid_cases = [
            {"spatial_features": None},
            {"temporal_features": None},
            {"spatial_features": np.zeros((2, 2), dtype=np.float32)},
            {"spatial_features": np.zeros((3, 3), dtype=np.float32)},
            {"temporal_features": np.zeros((3, 3), dtype=np.float32)},
            {"spatial_features": np.full((3, 2), np.nan, dtype=np.float32)},
            {"temporal_features": np.full((3, 2), np.inf, dtype=np.float32)},
            {"task_id": -1},
            {"task_id": 3},
            {"task_id": 1.5},
            {"task_id": True},
        ]
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                arguments = {
                    "task_id": 1,
                    "spatial_features": self.spatial,
                    "temporal_features": self.temporal,
                }
                arguments.update(invalid)
                with self.assertRaises((ValueError, RuntimeError)) as raised:
                    generator.learn(0, **arguments)
                self.assertTrue(str(raised.exception))
                self.assertEqual(len(generator.temporal_history), len(previous_history))
                for actual, expected in zip(generator.temporal_history, previous_history):
                    np.testing.assert_array_equal(actual, expected)

    def test_checkpoint_restores_parameters_and_temporal_history(self):
        generator = self.make_generator()
        self.learn(generator)
        self.learn(generator, task_id=1, temporal=self.temporal + 1)
        checkpoint = io.BytesIO()
        torch.save(generator.state_dict(), checkpoint)
        checkpoint.seek(0)
        state = torch.load(checkpoint, weights_only=True)
        restored = self.make_generator()
        restored.load_state_dict(state)

        expected = self.learn(generator, task_id=2, temporal=self.temporal + 2)
        actual = self.learn(restored, task_id=2, temporal=self.temporal + 2)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(restored.last_temporal_window, generator.last_temporal_window)
        self.assertEqual(len(restored.temporal_history), 3)

    def test_feature_dimensions_can_be_inferred_from_loaded_summaries(self):
        generator = BreastGraphGenerator(
            make_opt(
                client_spatial_features=[self.spatial, self.spatial + 1],
                client_temporal_features=[self.temporal, self.temporal + 1],
            )
        )

        self.assertEqual(self.learn(generator).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
