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

import logging

import pytest

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.comps.simple_intime_model_selector import SimpleIntimeModelSelector


class MockSimpleEngine:
    def __init__(self, job_id="unit_test"):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="__mock_simple_engine",
            job_id=job_id,
            public_stickers={},
            private_stickers={},
        )
        self.last_event = None

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        self.last_event = event_type
        return True


def _make_contribution_ctx(engine, metric_value, client_name="client1", current_round=1):
    dxo = DXO(
        DataKind.WEIGHTS,
        data=dict(),
        meta={MetaKey.INITIAL_METRICS: metric_value},
    )
    shareable = dxo.to_shareable()
    shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, current_round)

    peer_ctx = FLContext()
    peer_ctx.set_prop(ReservedKey.IDENTITY_NAME, client_name, private=False, sticky=False)

    fl_ctx = engine.fl_ctx_mgr.new_context()
    fl_ctx.set_peer_context(peer_ctx)
    fl_ctx.set_prop(AppConstants.TRAINING_RESULT, shareable, private=True, sticky=False)
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=False)
    return fl_ctx


class TestSimpleIntimeModelSelector:
    @pytest.mark.parametrize(
        "metric_value,expected_val_metric",
        [
            (0.75, 0.75),
            ({"val_accuracy": 0.75, "val_loss": 0.1}, 0.75),
        ],
    )
    def test_accepts_scalar_and_dict_metrics(self, metric_value, expected_val_metric):
        handler = SimpleIntimeModelSelector()
        engine = MockSimpleEngine()

        fl_ctx = _make_contribution_ctx(engine, metric_value)
        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)
        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)

        assert handler.val_metric == expected_val_metric
        assert engine.last_event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE

    def test_dict_metrics_use_configured_key_metric(self):
        handler = SimpleIntimeModelSelector(key_metric="accuracy")
        engine = MockSimpleEngine()

        fl_ctx = _make_contribution_ctx(engine, {"accuracy": 0.6, "val_loss": 0.2})
        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)
        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)

        assert handler.val_metric == 0.6
        assert engine.last_event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE

    def test_dict_metrics_missing_key_metric_skips_contribution(self):
        handler = SimpleIntimeModelSelector()
        engine = MockSimpleEngine()

        fl_ctx = _make_contribution_ctx(engine, {"val_loss": 0.1})
        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)

        # assert before BEFORE_AGGREGATION: the aggregation handler resets the sums on the accepted path too
        assert handler.validation_metric_sum_of_weights == 0

        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        assert engine.last_event != AppEventType.GLOBAL_BEST_MODEL_AVAILABLE

    @pytest.mark.parametrize("bad_value", [None, "not-a-number", [0.1, 0.2]])
    def test_non_numeric_metric_value_skips_contribution(self, bad_value):
        handler = SimpleIntimeModelSelector()
        engine = MockSimpleEngine()

        fl_ctx = _make_contribution_ctx(engine, {"val_accuracy": bad_value})
        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)

        assert handler.validation_metric_sum_of_weights == 0

        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        assert engine.last_event != AppEventType.GLOBAL_BEST_MODEL_AVAILABLE

    def test_negate_key_metric_supports_lower_is_better(self):
        handler = SimpleIntimeModelSelector(key_metric="val_loss", negate_key_metric=True)
        engine = MockSimpleEngine()

        fl_ctx = _make_contribution_ctx(engine, {"val_loss": 0.25})
        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)
        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)

        assert handler.val_metric == -0.25
        assert engine.last_event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE

    def test_aggregation_weights_use_client_identity(self):
        handler = SimpleIntimeModelSelector(aggregation_weights={"client1": 3.0, "client2": 1.0})
        engine = MockSimpleEngine()

        fl_ctx = _make_contribution_ctx(engine, {"val_accuracy": 1.0}, client_name="client1")
        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)
        fl_ctx = _make_contribution_ctx(engine, {"val_accuracy": 0.0}, client_name="client2")
        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)
        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)

        assert handler.val_metric == pytest.approx(0.75)

    @pytest.mark.parametrize(
        "key_metric,negate_key_metric,expect_warning",
        [
            ("val_loss", False, True),
            ("val_loss", True, False),
            ("val_accuracy", False, False),
        ],
    )
    def test_lower_is_better_key_metric_warns_at_construction(
        self, key_metric, negate_key_metric, expect_warning, caplog
    ):
        with caplog.at_level(logging.WARNING):
            SimpleIntimeModelSelector(key_metric=key_metric, negate_key_metric=negate_key_metric)

        assert ("looks like a lower-is-better metric" in caplog.text) == expect_warning
