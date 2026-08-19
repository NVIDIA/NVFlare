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

import numpy as np
import torch
from model import LogitCompletionModel, completed_logits, effect_target
from src.evaluation import PreparedSite, select_alpha, select_alpha_from_statistics, validation_sufficient_statistics
from src.federated import aggregation_meta, state_dict_for_update


def _missing_site(delta):
    return PreparedSite(
        site="site-3",
        labels=np.asarray([0, 1], dtype=np.int64),
        image_available=np.asarray([False, False]),
        missing_logits=np.asarray([0.0, 0.0]),
        full_logits=np.asarray([np.nan, np.nan]),
        predicted_delta=np.asarray(delta, dtype=np.float64),
    )


def test_effect_target_and_identity_completion():
    full = torch.tensor([2.0, -1.0])
    missing = torch.tensor([0.5, -0.5])
    predicted = torch.tensor([10.0, -10.0])
    assert torch.equal(effect_target(full, missing), torch.tensor([1.5, -0.5]))
    assert torch.equal(completed_logits(missing, predicted, alpha=0.0), missing)


def test_validation_selection_promotes_helpful_completion():
    selected, rows = select_alpha([_missing_site([-4.0, 4.0])], [0.0, 1.0])
    assert selected == 1.0
    assert all("aggregate_log_loss" in row for row in rows)


def test_validation_selection_retains_identity_when_completion_harms():
    selected, rows = select_alpha([_missing_site([4.0, -4.0])], [0.0, 1.0])
    assert selected == 0.0
    assert next(row for row in rows if row["alpha"] == 1.0)["feasible"] is False


def test_validation_selection_aggregates_only_sufficient_statistics():
    statistics = validation_sufficient_statistics([_missing_site([-4.0, 4.0])], [0.0, 1.0])
    assert set(statistics[0]) == {
        "site",
        "alpha",
        "missing_loss_sum",
        "missing_count",
        "aggregate_loss_sum",
        "aggregate_count",
    }
    selected, _ = select_alpha_from_statistics(statistics)
    assert selected == 1.0


def test_clients_without_paired_supervision_send_empty_update():
    model = LogitCompletionModel(input_dim=4, hidden_dim=2, dropout=0.0)
    assert state_dict_for_update(model, paired_examples=0) == {}
    assert aggregation_meta(0)["NUM_STEPS_CURRENT_ROUND"] == 0.0
    assert state_dict_for_update(model, paired_examples=3)
    assert aggregation_meta(3)["NUM_STEPS_CURRENT_ROUND"] == 3.0
