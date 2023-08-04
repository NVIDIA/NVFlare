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

import json
import os
import time

import pandas as pd
import xgboost as xgb

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader

def _get_data_intersection(df, intersection_path):
    # Note: the order of the intersection must be maintained

    with open(intersection_path) as intersection_file:
        intersection = intersection_file.read().splitlines()

    intersection_df =  df[df["uid"].isin(intersection)].copy()
    intersection_df["sort"] = pd.Categorical(intersection_df["uid"], categories=intersection, ordered=True)
    intersection_df = intersection_df.sort_values("sort")
    intersection_df = intersection_df.drop(["uid", "sort"], axis=1)

    return intersection_df

def _split_train_val(df, train_proportion):
    num_train = int(df.shape[0] * train_proportion)
    train_df = df.iloc[:num_train].copy()
    valid_df = df.iloc[num_train:].copy()

    return train_df, valid_df

class HIGGSDataLoader(XGBDataLoader):
    def __init__(self, data_split_dir, train_proportion):
        """Reads HIGGS dataset and return data paths to train and valid sets.

        Args:
            data_split_dir: directory containing data splits
        """
        self.data_split_dir = data_split_dir
        self.train_proportion = train_proportion

    def load_data(self, fl_ctx: FLContext):
        job_dir = os.path.dirname(os.path.abspath(fl_ctx.get_prop(FLContextKey.APP_ROOT)))
        client_id = fl_ctx.get_identity_name()
        psi_dir = os.path.join(job_dir, client_id, "psi")

        df = pd.read_csv(os.path.join(self.data_split_dir, client_id, "higgs.data.csv"))
        intersection_df = _get_data_intersection(df, os.path.join(psi_dir, "intersection.txt"))
        if intersection_df.empty:
            raise ValueError("private set intersection must not be empty")
        train_df, valid_df = _split_train_val(intersection_df, self.train_proportion)

        train_path = os.path.join(psi_dir, "higgs.train.csv")
        valid_path = os.path.join(psi_dir, "higgs.test.csv")
        train_df.to_csv(path_or_buf=train_path, header=False, index=False)
        valid_df.to_csv(path_or_buf=valid_path, header=False, index=False)

        return train_path, valid_path
