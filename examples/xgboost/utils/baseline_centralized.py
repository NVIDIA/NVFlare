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
import time

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

# Specify training params
data_path = "../dataset/HIGGS_UCI.csv"
model_path_root = "../workspaces/centralized"
round_num = 100

# Set mode file paths
model_path = model_path_root + "/model_centeralized.json"
# Set tensorboard output
writer = SummaryWriter(model_path_root)

# Load data
start = time.time()
higgs = pd.read_csv(data_path, header=None)
print(higgs.info())
print(higgs.head())
total_data_num = higgs.shape[0]
valid_num = 1000000
print(f"Total data count: {total_data_num}")
# split to feature and label
X_higgs = higgs.iloc[:, 1:]
y_higgs = higgs.iloc[:, 0]
print(y_higgs.value_counts())
end = time.time()
lapse_time = end - start
print(f"Data loading time: {lapse_time}")

# construct xgboost DMatrix
# split to training and validation
dmat_higgs = xgb.DMatrix(X_higgs, label=y_higgs)
dmat_valid = dmat_higgs.slice(X_higgs.index[0:valid_num])
dmat_train = dmat_higgs.slice(X_higgs.index[valid_num:])

# setup parameters for xgboost
# use logistic regression loss for binary classification
# learning rate 0.1 max_depth 5
# use auc as metric
param = {}
param["objective"] = "binary:logistic"
param["eta"] = 0.1
param["max_depth"] = 8
param["eval_metric"] = "auc"
param["nthread"] = 16

# xgboost training
start = time.time()
for round in range(round_num):
    # Train model
    if os.path.exists(model_path):
        # Validate the last round's model
        bst_last = xgb.Booster(param, model_file=model_path)
        y_pred = bst_last.predict(dmat_valid)
        roc = roc_auc_score(y_higgs[0:1000000], y_pred)
        print(f"Round: {bst_last.num_boosted_rounds()} model testing AUC {roc}")
        writer.add_scalar("AUC", roc, round - 1)
        # Train new model
        print(f"Round: {round} Base ", end="")
        bst = xgb.train(
            param,
            dmat_train,
            num_boost_round=1,
            xgb_model=model_path,
            evals=[(dmat_valid, "validate"), (dmat_train, "train")],
        )
    else:
        # Round 0
        print(f"Round: {round} Base ", end="")
        bst = xgb.train(param, dmat_train, num_boost_round=1, evals=[(dmat_valid, "validate"), (dmat_train, "train")])
    bst.save_model(model_path)

end = time.time()
lapse_time = end - start
print(f"Training time: {lapse_time}")

# test model
bst = xgb.Booster(param, model_file=model_path)
y_pred = bst.predict(dmat_valid)
roc = roc_auc_score(y_higgs[0:1000000], y_pred)
print(f"Base model: {roc}")
writer.add_scalar("AUC", roc, round_num - 1)

writer.close()
