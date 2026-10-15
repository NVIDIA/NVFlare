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

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.client import ModifiedClient
from model.server import Server


def make_opt(**overrides):
    options = dict(
        device="cpu",
        batch_size=4,
        input_dim=8,
        nh=16,
        nt=4,
        num_clients=4,
        num_classes=2,
        no_bn=False,
        p=0.0,
        lr_e=0.001,
        lr_f=0.002,
        lr_g=0.003,
        lr_d=0.004,
        beta1=0.9,
        beta2=0.999,
    )
    options.update(overrides)
    return SimpleNamespace(**options)


def fill_parameters(optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.fill_(0.5)


def step_without_data_gradient(optimizer):
    optimizer.zero_grad(set_to_none=True)
    # Give every parameter a real zero gradient so any update is due to decay.
    loss = sum(parameter.sum() * 0.0 for group in optimizer.param_groups for parameter in group["params"])
    loss.backward()
    optimizer.step()


class WeightDecayTest(unittest.TestCase):
    def assert_decay_step(self, optimizer, expected_rates):
        fill_parameters(optimizer)
        step_without_data_gradient(optimizer)

        self.assertEqual(len(optimizer.param_groups), len(expected_rates))
        for group, rate in zip(optimizer.param_groups, expected_rates):
            self.assertEqual(group["lr"], rate)
            for parameter in group["params"]:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))
                # Adam's first step shrinks a positive parameter by about lr.
                torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5 - rate))
            self.assertEqual(group["weight_decay"], 0.1)

    def test_positive_decay_updates_every_client_network_without_data_gradient(self):
        opt = make_opt(weight_decay=0.1)
        client = ModifiedClient(0, opt)

        self.assert_decay_step(client.optimizer_EFG, [opt.lr_e, opt.lr_f, opt.lr_g])

    def test_positive_decay_updates_server_without_data_gradient(self):
        opt = make_opt(weight_decay=0.1)
        server = Server(opt)

        self.assert_decay_step(server.optimizer_D, [opt.lr_d])

    def assert_no_decay_update(self, opt):
        client = ModifiedClient(0, opt)
        server = Server(opt)
        for name, optimizer in (("client", client.optimizer_EFG), ("server", server.optimizer_D)):
            with self.subTest(optimizer=name):
                fill_parameters(optimizer)
                step_without_data_gradient(optimizer)
                for group in optimizer.param_groups:
                    self.assertEqual(group["weight_decay"], 0.0)
                    for parameter in group["params"]:
                        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5), rtol=0, atol=0)

    def test_zero_decay_keeps_parameters_unchanged(self):
        self.assert_no_decay_update(make_opt(weight_decay=0.0))

    def test_missing_decay_preserves_legacy_zero_decay_behavior(self):
        self.assert_no_decay_update(make_opt())

    def test_client_training_state_preserves_decay_and_next_update(self):
        client = ModifiedClient(0, make_opt(weight_decay=0.1))
        fill_parameters(client.optimizer_EFG)
        step_without_data_gradient(client.optimizer_EFG)
        restored = ModifiedClient(0, make_opt(weight_decay=0.0))
        restored.set_weights(client.get_weights())
        restored.set_training_state(client.get_training_state())

        for group in restored.optimizer_EFG.param_groups:
            self.assertEqual(group["weight_decay"], 0.1)
        step_without_data_gradient(client.optimizer_EFG)
        step_without_data_gradient(restored.optimizer_EFG)
        for expected, actual in zip(client.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
