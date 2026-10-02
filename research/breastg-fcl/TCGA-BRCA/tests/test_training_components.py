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
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.client import ModifiedClient
from model.modules import BreastGraphGenerator
from model.server import Server


def make_opt(**overrides):
    options = dict(
        device="cpu",
        batch_size=4,
        use_g_encode=True,
        input_dim=8,
        nh=16,
        ni=16,
        nt=4,
        nd_out=4,
        num_clients=4,
        num_classes=2,
        noise_dim=5,
        no_bn=False,
        p=0.0,
        lr_e=0.001,
        lr_f=0.002,
        lr_g=0.003,
        lr_d=0.004,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        lambda_gan=0.5,
        replay=True,
        b=0.0,
        temporal_window=2,
        attention_temperature=1.0,
        graph_epsilon=1e-8,
        gat_hidden_dim=32,
        gat_embedding_dim=16,
        gat_heads=2,
        gat_dropout=0.0,
    )
    options.update(overrides)
    return SimpleNamespace(**options)


class GuardedLoader:
    """Allow task registration, then detect any subsequent historical raw read."""

    def __init__(self, loader):
        self.loader = loader
        self.blocked = False

    def __iter__(self):
        if self.blocked:
            raise AssertionError("Historical raw samples must not be read during replay")
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class TrainingComponentsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.opt = make_opt()
        self.inputs = torch.randn(4, self.opt.input_dim)
        self.labels = torch.tensor([0, 1, 0, 1])
        self.loader = DataLoader(TensorDataset(self.inputs, self.labels), batch_size=4)
        self.graphs = [np.roll(np.eye(4, dtype=np.float32), task, axis=1) for task in range(3)]
        self.server = Server(self.opt)
        self.client = ModifiedClient(0, self.opt)
        self.client.set_server_discriminator(self.server.get_discriminator())

    def register_tasks(self, client=None, through=2):
        client = self.client if client is None else client
        loaders = []
        for task in range(through + 1):
            loader = GuardedLoader(self.loader)
            client.register_task(task, loader)
            loader.blocked = True
            loaders.append(loader)
        return loaders

    def snapshot(self, module):
        return {name: value.detach().clone() for name, value in module.named_parameters()}

    def assert_changed(self, module, before):
        self.assertTrue(any(not torch.equal(before[name], value.detach()) for name, value in module.named_parameters()))

    def assert_unchanged(self, module, before):
        for name, value in module.named_parameters():
            torch.testing.assert_close(value.detach(), before[name], rtol=0, atol=0)

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
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

    def test_client_optimizes_encoder_predictor_generator_with_separate_rates(self):
        groups = self.client.optimizer_EFG.param_groups
        self.assertEqual(len(groups), 3)
        expected = [
            (self.client.netE, self.opt.lr_e),
            (self.client.netF, self.opt.lr_f),
            (self.client.netG, self.opt.lr_g),
        ]
        for group, (module, rate) in zip(groups, expected):
            self.assertEqual(group["lr"], rate)
            self.assertEqual({id(item) for item in group["params"]}, {id(item) for item in module.parameters()})
        self.register_tasks(through=0)
        before = [self.snapshot(module) for module, _rate in expected]
        local_discriminator = self.snapshot(self.client.server_discriminator)
        global_discriminator = self.snapshot(self.server.global_discriminator)

        result = self.client.learn(0, 0, self.graphs, self.loader)

        self.assertTrue(result["loss_values"])
        self.assertTrue(all(np.isfinite(value) for value in result["loss_values"].values()))
        for (module, _rate), original in zip(expected, before):
            self.assert_changed(module, original)
        self.assert_unchanged(self.client.server_discriminator, local_discriminator)
        self.assert_unchanged(self.server.global_discriminator, global_discriminator)
        self.assertFalse(self.client.server_discriminator.training)
        for parameter in self.client.server_discriminator.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_adversarial_loss_backpropagates_through_frozen_discriminator(self):
        self.client.eval()
        graph_rows = torch.tensor(self.graphs[0][0]).expand(4, -1)
        real = self.client.netE(self.inputs, graph_rows)
        generated = self.client.netG(torch.randn(4, self.opt.noise_dim), self.labels, graph_rows)
        loss = -F.mse_loss(self.client.server_discriminator(real), graph_rows)
        loss = loss - F.mse_loss(self.client.server_discriminator(generated), graph_rows)

        loss.backward()

        for module in (self.client.netE, self.client.netG):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.client.server_discriminator.parameters()))

        self.register_tasks(through=0)
        differentiable_outputs = []
        hook = self.client.server_discriminator.register_forward_hook(
            lambda _module, _inputs, output: differentiable_outputs.append(output.requires_grad)
        )
        try:
            self.client.learn(0, 0, self.graphs, self.loader)
        finally:
            hook.remove()
        self.assertTrue(differentiable_outputs)
        self.assertTrue(all(differentiable_outputs))

    def test_test_predictions_do_not_depend_on_ground_truth_labels_or_generator(self):
        outputs = []
        hook = self.client.netF.register_forward_hook(
            lambda _module, _inputs, output: outputs.append(
                (output[0] if isinstance(output, tuple) else output).detach().clone()
            )
        )
        flipped = DataLoader(TensorDataset(self.inputs, 1 - self.labels), batch_size=4)
        try:
            with patch.object(
                self.client.netG, "forward", side_effect=AssertionError("Real prediction must not invoke G")
            ):
                self.client.test(0, self.loader, self.graphs)
                self.client.test(0, flipped, self.graphs)
        finally:
            hook.remove()

        self.assertEqual(len(outputs), 2)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_joint_training_replays_every_previous_task_without_old_raw_data(self):
        self.register_tasks()
        seen_rows = []
        hook = self.client.netG.register_forward_pre_hook(
            lambda _module, args: seen_rows.append(args[2].detach().clone())
        )
        try:
            self.client.learn(0, 2, self.graphs, self.loader)
        finally:
            hook.remove()

        self.assertTrue(seen_rows)
        unique_rows = {tuple(row.tolist()) for batch in seen_rows for row in batch}
        self.assertEqual(unique_rows, {tuple(graph[0].tolist()) for graph in self.graphs})

    def test_disabling_replay_still_trains_current_task_generator(self):
        self.opt.replay = False
        self.register_tasks()
        original = self.snapshot(self.client.netG)
        seen_rows = []
        hook = self.client.netG.register_forward_pre_hook(
            lambda _module, args: seen_rows.append(args[2].detach().clone())
        )
        try:
            self.client.learn(0, 2, self.graphs, self.loader)
        finally:
            hook.remove()

        self.assert_changed(self.client.netG, original)
        unique_rows = {tuple(row.tolist()) for batch in seen_rows for row in batch}
        self.assertEqual(unique_rows, {tuple(self.graphs[2][0].tolist())})

    def test_synthetic_only_task_zero_replay_never_reads_loader_or_encoder(self):
        old_loaders = self.register_tasks()
        with patch.object(
            self.client.netE, "forward", side_effect=AssertionError("Synthetic replay must use G directly")
        ):
            generated = self.client.generate_encodings(0, self.graphs, old_loaders[0], generate=True)
            original = self.snapshot(self.client.netG)
            self.client.learn(0, 0, self.graphs, old_loaders[0], generate=True)

        self.assert_changed(self.client.netG, original)
        self.assertTrue(generated["encodings"])
        self.assertEqual(len(generated["encodings"]), len(generated["graph_embeddings"]))
        for latents, targets in zip(generated["encodings"], generated["graph_embeddings"]):
            self.assertEqual(latents.shape[1], self.opt.nh)
            expected = torch.tensor(self.graphs[0][0]).expand(latents.shape[0], -1)
            torch.testing.assert_close(targets, expected)

    def test_uploaded_latents_have_aligned_current_and_historical_graph_rows(self):
        self.register_tasks()

        payload = self.client.generate_encodings(2, self.graphs, self.loader)

        self.assertEqual(len(payload["encodings"]), len(payload["graph_embeddings"]))
        self.assertTrue(payload["encodings"])
        all_targets = []
        for latents, targets in zip(payload["encodings"], payload["graph_embeddings"]):
            self.assertEqual(latents.ndim, 2)
            self.assertEqual(latents.shape[1], self.opt.nh)
            self.assertEqual(targets.shape, (latents.shape[0], self.opt.num_clients))
            self.assertFalse(latents.requires_grad)
            self.assertFalse(targets.requires_grad)
            all_targets.append(targets)
        targets = torch.cat(all_targets)
        counts = {
            task: (targets == torch.tensor(graph[0])).all(dim=1).sum().item() for task, graph in enumerate(self.graphs)
        }
        self.assertEqual(counts, {0: 4, 1: 4, 2: 8})

    def test_weights_include_generator_and_are_independent_snapshots(self):
        weights = self.client.get_weights()
        self.assertEqual(set(weights), {"encoder", "predictor", "generator"})
        restored = ModifiedClient(0, self.opt)
        restored.set_weights(weights)
        self.assert_nested_equal(restored.get_weights(), weights)
        with torch.no_grad():
            next(self.client.netG.parameters()).add_(1)
        self.assert_nested_equal(restored.get_weights(), weights)
        self.assertTrue(
            any(
                not torch.equal(self.client.get_weights()["generator"][name], value)
                for name, value in weights["generator"].items()
            )
        )

    def test_training_state_restores_optimizer_schedule_and_replay_metadata(self):
        self.register_tasks()
        self.client.learn(0, 2, self.graphs, self.loader)
        weights = self.client.get_weights()
        state = copy.deepcopy(self.client.get_training_state())
        restored = ModifiedClient(0, self.opt)
        restored.set_server_discriminator(self.server.get_discriminator())
        restored.set_weights(weights)
        restored.set_training_state(state)
        self.assert_nested_equal(restored.get_training_state(), state)

        torch.manual_seed(73)
        self.client.learn(1, 2, self.graphs, self.loader)
        torch.manual_seed(73)
        restored.learn(1, 2, self.graphs, self.loader)

        self.assert_nested_equal(restored.get_weights(), self.client.get_weights())
        self.assert_nested_equal(restored.get_training_state(), self.client.get_training_state())

    def test_server_updates_only_discriminator_and_uses_correct_paired_targets(self):
        latents = [torch.randn(2, self.opt.nh, requires_grad=True), torch.randn(3, self.opt.nh, requires_grad=True)]
        targets = [
            torch.tensor(self.graphs[0][0]).unsqueeze(0).requires_grad_(),
            torch.tensor(self.graphs[1][0]).unsqueeze(0).requires_grad_(),
        ]
        original = self.snapshot(self.server.global_discriminator)
        with torch.no_grad():
            paired_targets = torch.cat([targets[0].expand(2, -1), targets[1].expand(3, -1)])
            expected_loss = F.mse_loss(self.server.global_discriminator(torch.cat(latents)), paired_targets).item()

        actual_loss = self.server.train_discriminator(latents, targets)

        self.assertAlmostEqual(actual_loss, expected_loss, places=6)
        self.assert_changed(self.server.global_discriminator, original)
        self.assertTrue(all(item.grad is None for item in latents + targets))

    def test_server_rejects_misaligned_pairs_without_updating_discriminator(self):
        original = self.snapshot(self.server.global_discriminator)
        cases = [
            ([torch.zeros(2, 16), torch.zeros(3, 16)], [torch.zeros(3, 4), torch.zeros(2, 4)]),
            ([torch.zeros(4, 16)], [torch.zeros(2, 4)]),
            ([torch.zeros(4, 16)], [torch.zeros(4, 3)]),
            ([torch.zeros(4, 15)], [torch.zeros(4, 4)]),
            ([torch.zeros(4, 16)], []),
            ([torch.full((4, 16), float("nan"))], [torch.zeros(4, 4)]),
        ]
        for latents, targets in cases:
            with self.subTest(
                shapes=([tuple(value.shape) for value in latents], [tuple(value.shape) for value in targets])
            ):
                with self.assertRaises(ValueError):
                    self.server.train_discriminator(latents, targets)
                self.assert_unchanged(self.server.global_discriminator, original)

    def test_attention_parameters_remain_fixed_during_client_server_workflow(self):
        attention = BreastGraphGenerator(self.opt, spatial_dim=2, temporal_dim=2)
        before = self.snapshot(attention)
        summaries = np.asarray([[0, 1], [1, 0], [2, 1], [1, 3]], dtype=np.float32)
        graphs = [
            attention.learn(0, task_id=task, spatial_features=summaries, temporal_features=summaries + task)
            for task in range(3)
        ]
        self.register_tasks()
        payload = self.client.generate_encodings(2, graphs, self.loader)
        discriminator_before = self.snapshot(self.server.global_discriminator)
        generator_before = self.snapshot(self.client.netG)

        self.server.train_discriminator(payload["encodings"], payload["graph_embeddings"])
        self.client.set_server_discriminator(self.server.get_discriminator())
        self.client.learn(0, 2, graphs, self.loader)

        self.assert_changed(self.server.global_discriminator, discriminator_before)
        self.assert_changed(self.client.netG, generator_before)
        self.assert_unchanged(attention, before)
        self.assertTrue(all(parameter.grad is None for parameter in attention.parameters()))


if __name__ == "__main__":
    unittest.main()
