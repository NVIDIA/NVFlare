# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import threading
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.shareablegenerators.passthru import PassthroughShareableGenerator
from nvflare.edge.assessor import Assessment, Assessor
from nvflare.fuel.utils.validation_utils import check_str
from nvflare.security.logging import secure_format_exception


class SGAPAssessor(Assessor):

    def __init__(self, shareable_generator_id: str, aggregator_id: str, persistor_id: str):
        """This assessor implements its required logic by using a Shareable Generator, an Aggregator, and a
        Persistor (SGAP).

        Args:
            shareable_generator_id: component ID of the Shareable Generator. If empty, the PassthroughShareableGenerator
                will be used.
            aggregator_id: component ID of the Aggregator.
            persistor_id: component ID of the Persistor. If not specified, the Persistor will load initial model
                and save the final model.
        """
        Assessor.__init__(self)
        check_str("persistor_id", persistor_id)
        check_str("shareable_generator_id", shareable_generator_id)
        check_str("aggregator_id", aggregator_id)

        self.aggregator_id = aggregator_id
        self.shareable_generator_id = shareable_generator_id
        self.persistor_id = persistor_id
        self._global_weights = make_model_learnable({}, {})
        self._aggr_lock = threading.Lock()

        self.shareable_gen = None
        self.aggregator = None
        self.persistor = None

        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.aggregator = engine.get_component(self.aggregator_id)
        if not isinstance(self.aggregator, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_id} must be an Aggregator type object but got {type(self.aggregator)}",
                fl_ctx,
            )
            return

        if self.shareable_generator_id:
            self.shareable_gen = engine.get_component(self.shareable_generator_id)
            if not isinstance(self.shareable_gen, ShareableGenerator):
                self.system_panic(
                    f"Shareable generator {self.shareable_generator_id} must be a ShareableGenerator type object, "
                    f"but got {type(self.shareable_gen)}",
                    fl_ctx,
                )
                return
        else:
            self.shareable_gen = PassthroughShareableGenerator()

        if self.persistor_id:
            self.persistor = engine.get_component(self.persistor_id)
            if not isinstance(self.persistor, LearnablePersistor):
                self.system_panic(
                    f"Persistor {self.persistor_id} must be a LearnablePersistor type object, "
                    f"but got {type(self.persistor)}",
                    fl_ctx,
                )
                return

        if self.persistor:
            model = self.persistor.load(fl_ctx)

            if not isinstance(model, ModelLearnable):
                self.system_panic(
                    reason=f"Expected model loaded by persistor to be `ModelLearnable` but received {type(model)}",
                    fl_ctx=fl_ctx,
                )
                return

            self._global_weights = model
            fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=True)
            self.fire_event(AppEventType.INITIAL_MODEL_LOADED, fl_ctx)

    def start_task(self, fl_ctx: FLContext) -> Shareable:
        # Use the Shareable Generator to generate task data
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_info(fl_ctx, f"starting round {current_round}")
        return self.shareable_gen.learnable_to_shareable(self._global_weights, fl_ctx)

    def process_child_update(self, data: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        # Process update from child.
        with self._aggr_lock:
            accepted = self.aggregator.accept(data, fl_ctx)
        return accepted, None

    def end_task(self, fl_ctx: FLContext):
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_info(fl_ctx, f"Start aggregation for round {current_round}")
        self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        with self._aggr_lock:
            try:
                aggr_result = self.aggregator.aggregate(fl_ctx)
                self.aggregator.reset(fl_ctx)
            except Exception as ex:
                self.log_exception(fl_ctx, f"aggregation error from {type(self.aggregator)}")
                self.system_panic(f"aggregation error: {secure_format_exception(ex)}", fl_ctx)
                return

        self.fire_event_with_data(AppEventType.AFTER_AGGREGATION, fl_ctx, AppConstants.AGGREGATION_RESULT, aggr_result)

        self.log_info(fl_ctx, f"End aggregation for round {current_round}.")

        self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
        self._global_weights = self.shareable_gen.shareable_to_learnable(aggr_result, fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
        self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

        if self.persistor:
            self.log_info(fl_ctx, f"Start persist model on server for round {current_round}.")
            self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
            self.persistor.save(self._global_weights, fl_ctx)
            self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
            self.log_info(fl_ctx, f"End persist model on server for round {current_round}")

    def do_assessment(self, fl_ctx: FLContext):
        return Assessment.CONTINUE

    def assess(self, fl_ctx: FLContext) -> Assessment:
        return self.do_assessment(fl_ctx)
