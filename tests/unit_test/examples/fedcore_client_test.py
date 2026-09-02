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
import torch

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


def test_local_training_is_deterministic_with_dropout():
    with fedcore_import_context():
        import client
        from model import LogitCompletionModel

        payload = {
            "paired_mask": torch.ones(8, dtype=torch.bool),
            "missing_features": torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32.0,
            "missing_logits": torch.zeros(8),
            "full_logits": torch.tensor([-2.0, 2.0] * 4),
            "labels": torch.tensor([0, 1] * 4),
        }
        first = LogitCompletionModel(input_dim=4, hidden_dim=8, dropout=0.5, seed=11)
        second = LogitCompletionModel(input_dim=4, hidden_dim=8, dropout=0.5, seed=11)
        kwargs = {
            "payload": payload,
            "local_epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-2,
            "task_weight": 1.0,
            "effect_weight": 1.0,
            "seed": 13,
            "current_round": 2,
        }
        first_metrics = client._train_round(model=first, **kwargs)
        second_metrics = client._train_round(model=second, **kwargs)

        assert first_metrics == second_metrics
        for name, tensor in first.state_dict().items():
            assert torch.equal(tensor, second.state_dict()[name])


def test_sigterm_handler_requests_single_entry_shutdown(monkeypatch):
    with fedcore_import_context():
        import client

        calls = []
        monkeypatch.setattr(client.flare, "shutdown", lambda: calls.append("shutdown"))
        client._TERMINATION_REQUESTED = False
        client._shutdown_on_signal(None, None)

    assert client._TERMINATION_REQUESTED is True
    assert calls == []


def test_local_training_rejects_non_positive_sizes():
    with fedcore_import_context():
        import client
        from model import LogitCompletionModel

        payload = {
            "paired_mask": torch.ones(1, dtype=torch.bool),
            "missing_features": torch.zeros((1, 2)),
            "missing_logits": torch.zeros(1),
            "full_logits": torch.ones(1),
            "labels": torch.ones(1),
        }
        model = LogitCompletionModel(input_dim=2, hidden_dim=2)
        common = {
            "model": model,
            "payload": payload,
            "learning_rate": 1e-3,
            "task_weight": 1.0,
            "effect_weight": 1.0,
            "seed": 7,
            "current_round": 0,
        }
        with pytest.raises(ValueError, match="local_epochs"):
            client._train_round(local_epochs=0, batch_size=1, **common)
        with pytest.raises(ValueError, match="batch_size"):
            client._train_round(local_epochs=1, batch_size=0, **common)
