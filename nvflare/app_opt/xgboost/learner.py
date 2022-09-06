# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_opt.xgboost.learner_base import XGBFedLearnerBase, XGBoostParams
from nvflare.utils.import_utils import optional_import


class XGBFedLearner(XGBFedLearnerBase):
    def xgb_train(self, params: XGBoostParams, fl_ctx: FLContext) -> Shareable:
        xgb, flag = optional_import(module="xgboost")
        if not flag:
            self.log_error(fl_ctx, "Can't import xgboost")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        callback, flag = optional_import(module="xgboost.callback")
        if not flag:
            self.log_error(fl_ctx, "Can't import xgboost.callback")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        with xgb.rabit.RabitContext([e.encode() for e in params.rabit_env]):

            # Load file, file will not be sharded in federated mode.
            dtrain = xgb.DMatrix(params.train_data)
            dtest = xgb.DMatrix(params.test_data)

            # Specify validations set to watch performance
            watchlist = [(dtest, "eval"), (dtrain, "train")]

            # Run training, all the features in training API is available.
            bst = xgb.train(
                params.xgb_params,
                dtrain,
                params.num_rounds,
                evals=watchlist,
                early_stopping_rounds=params.early_stopping_rounds,
                verbose_eval=params.verbose_eval,
                callbacks=[callback.EvaluationMonitor(rank=self.rank)],
            )

            # Save the model.
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
            run_dir = workspace.get_run_dir(run_number)
            bst.save_model(os.path.join(run_dir, "test.model.json"))
            xgb.rabit.tracker_print("Finished training\n")

            return make_reply(ReturnCode.OK)
