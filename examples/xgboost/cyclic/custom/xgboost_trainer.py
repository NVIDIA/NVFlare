# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pandas as pd
import xgboost as xgb
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

np.random.seed(0)

class XGBoostTrainer(Executor):
    def __init__(self, trees_per_round, site_num):
        super().__init__()
        self.trees_per_round = trees_per_round
        self.site_num = site_num
        self.train_X, self.train_y = [], []
        self.test_X, self.test_y = [], []
        self.model = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.init(fl_ctx)

    def init(self, fl_ctx: FLContext):
        credit_df = pd.read_csv('/media/ziyuexu/Research/Experiment/XGBoost/Data/UCI_Credit_Card.csv')
        # Target values
        y = credit_df['default.payment.next.month']
        # One hot encoding for categorical vars 'SEX', 'MARRIAGE' & 'EDUCATION'
        X = pd.get_dummies(data=credit_df, columns=['SEX', 'MARRIAGE', 'EDUCATION'])
        X.drop(columns=['default.payment.next.month'], axis=1, inplace=True)
        train_X, self.test_X, train_y, self.test_y = train_test_split(X, y, random_state=77, test_size=0.25)
        # Split training set to site_num sites
        # simulate separate datasets for each client by dividing dataset into number of clients
        site_index = np.random.random_integers(0, self.site_num - 1, len(train_y))
        client_name = fl_ctx.get_identity_name()
        for site in range(self.site_num):
            if client_name == "site-" + str(site+1):
                self.train_X = train_X[site_index==site]
                self.train_y = train_y[site_index==site]
                inst_num = len(self.train_y)
                print(f"Site {site}: total instances {inst_num}")


    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: dispatched task
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != "train":
            return make_reply(ReturnCode.TASK_UNKNOWN)

        dxo = from_shareable(shareable)
        model_global = dxo.data

        model = xgb.XGBClassifier(objective='reg:squarederror', learning_rate=0.1, max_depth=5,
                                  n_estimators=self.trees_per_round)
        if not model_global:
            print("********************************")
            print("Initial Training from scratch")
            model.fit(self.train_X, self.train_y)
        else:
            print("********************************")
            print("Training from global model received")
            # save it to local temp file
            with open('temp_model.json', 'w') as f:
                json.dump(model_global, f)
            # load global model
            model_global = xgb.XGBClassifier()
            model_global.load_model('temp_model.json')
            # train local model starting with global model
            model.fit(self.train_X, self.train_y, xgb_model=model_global)

        pred_y = model.predict(self.test_X)
        acc = accuracy_score(pred_y, self.test_y)
        print("---------------------------------------")
        print("Total boosted rounds:", model.get_booster().num_boosted_rounds())
        print("Accuracy {:.2f} %".format(100 * acc))

        model.save_model('temp_model_local.json')
        # report updated weights in shareable
        with open('temp_model_local.json') as json_file:
            model_new = json.load(json_file)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_new)
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()
        return new_shareable
