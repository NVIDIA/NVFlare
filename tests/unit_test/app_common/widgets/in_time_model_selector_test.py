# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector


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


class TestInTimeModelSelector:
    @pytest.mark.parametrize(
        "initial,received,expected",
        [
            (
                1,
                {
                    "client1": {"weight": 0.5, "iter_number": 1, "metric": 10},
                },
                True,
            ),
            (
                1,
                {
                    "client1": {"weight": 0.5, "iter_number": 1, "metric": 1},
                    "client2": {"weight": 0.5, "iter_number": 1, "metric": 0.2},
                },
                False,
            ),
        ],
    )
    def test_model_selection(self, initial, received, expected):
        aggregation_weights = {k: v["weight"] for k, v in received.items()}
        handler = IntimeModelSelector(aggregation_weights=aggregation_weights)
        handler.best_val_metric = initial
        engine = MockSimpleEngine()
        fl_ctx = engine.fl_ctx_mgr.new_context()
        for k, v in received.items():
            peer_ctx = FLContext()
            peer_ctx.set_prop(FLContextKey.CLIENT_NAME, k, private=False)
            dxo = DXO(
                DataKind.WEIGHT_DIFF,
                data=dict(),
                meta={
                    MetaKey.INITIAL_METRICS: v["metric"],
                    MetaKey.NUM_STEPS_CURRENT_ROUND: v["iter_number"],
                    AppConstants.CURRENT_ROUND: 10,
                },
            )

            peer_ctx.set_prop(FLContextKey.SHAREABLE, dxo.to_shareable(), private=True)
            fl_ctx = engine.fl_ctx_mgr.new_context()
            fl_ctx.set_peer_context(peer_ctx)

            handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)
        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        assert (engine.last_event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE) == expected

    def test_model_selection_publishes_metrics_selection_info(self):
        handler = IntimeModelSelector(key_metric="loss", negate_key_metric=True)
        engine = MockSimpleEngine()
        peer_ctx = FLContext()
        dxo = DXO(
            DataKind.WEIGHT_DIFF,
            data=dict(),
            meta={
                MetaKey.INITIAL_METRICS: {"loss": 0.2},
                MetaKey.NUM_STEPS_CURRENT_ROUND: 1,
            },
        )
        shareable = dxo.to_shareable()
        shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, 1)
        peer_ctx.set_prop(FLContextKey.SHAREABLE, shareable, private=True)
        fl_ctx = engine.fl_ctx_mgr.new_context()
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1, private=True, sticky=False)
        fl_ctx.set_peer_context(peer_ctx)

        handler.handle_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)
        handler.handle_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)

        assert engine.last_event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE
        assert fl_ctx.get_prop(AppConstants.VALIDATION_RESULT) == -0.2
        assert fl_ctx.get_prop(AppConstants.METRICS_SELECTION_INFO) == {
            "source": "IntimeModelSelector",
            "metric_source": MetaKey.INITIAL_METRICS,
            "key_metric": {
                "name": "loss",
                "mode": "min",
                "mode_source": "IntimeModelSelector.negate_key_metric",
            },
            "best_round": 1,
            "best_metrics": {"loss": 0.2},
        }
