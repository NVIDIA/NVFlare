# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os.path

import joblib
import tensorboard

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception


class SKLearnExecutor(Executor):
    def __init__(self, learner_id: str):
        super().__init__()
        self.client_id = None
        self.writer = None
        self.learner_id = learner_id
        self.local_model_path = None
        self.global_model_path = None
        self.learner = None
        self.fl_ctx = None

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self._print_configs(fl_ctx)
        self.load_log_tracker()

        engine = fl_ctx.get_engine()
        self.learner = engine.get_component(self.learner_id)
        self.learner.initialize(fl_ctx)

        # set the paths according to fl_ctx
        app_dir = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.local_model_path = os.path.join(app_dir, "model_local.joblib")
        self.global_model_path = os.path.join(app_dir, "model_global.joblib")

    def get_global_params(self, shareable: Shareable, fl_ctx: FLContext):
        # retrieve current global center download from server's shareable
        dxo = from_shareable(shareable)
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round)
        return current_round, dxo.data

    def train(self, current_round, global_param, fl_ctx: FLContext) -> Shareable:
        self._msg_log(f"Client {self.client_id} perform local train")
        # sklearn algorithms usually needs two different processing schemes
        # one for first round (generate initial centers for clustering, regular training for svm)
        # the other for following rounds (regular training for clustering, no further training for svm)
        # hence the current round is fed to learner to distinguish the two
        params, model = self.learner.train(current_round, global_param, fl_ctx)
        # save model and return dxo containing the params
        self.save_model_local(model)
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, self.learner.n_samples)
        self._msg_log("Local epochs finished. Returning shareable")

        return dxo.to_shareable()

    def validate(self, current_round, global_param, fl_ctx: FLContext) -> Shareable:
        # retrieve current global center download from server's shareable
        self._msg_log(f"Client {self.client_id} perform local evaluation")
        metrics, model = self.learner.validate(current_round, global_param, fl_ctx)
        self.save_model_global(model)
        for key, value in metrics.items():
            self.log_value(key, value, current_round)

    def finalize(self, fl_ctx: FLContext):
        self.learner.finalize(fl_ctx)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        self._msg_log(f"Client trainer got task: {task_name}")
        if abort_signal.triggered:
            self.finalize(fl_ctx)
            return make_reply(ReturnCode.TASK_ABORTED)
        try:
            if task_name == AppConstants.TASK_TRAIN:
                (current_round, global_params) = self.get_global_params(shareable, fl_ctx)
                if current_round > 0:
                    # first round for parameter initialization
                    # no model evaluation
                    self.validate(current_round, global_params, fl_ctx)
                return self.train(current_round, global_params, fl_ctx)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"execute exception: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def _print_configs(self, fl_ctx: FLContext):
        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with configs: \n {fl_args}",
        )

    def load_log_tracker(self):
        app_dir = self.fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.writer = tensorboard.summary.Writer(app_dir)

    def _msg_log(self, msg: str):
        self.log_info(self.fl_ctx, msg)

    def log_value(self, key, value, step):
        if self.writer:
            self.writer.add_scalar(key, value, step)
            self.writer.flush()

    def save_model_local(self, model: any) -> None:
        joblib.dump(model, self.local_model_path)

    def save_model_global(self, model: any) -> None:
        joblib.dump(model, self.global_model_path)
