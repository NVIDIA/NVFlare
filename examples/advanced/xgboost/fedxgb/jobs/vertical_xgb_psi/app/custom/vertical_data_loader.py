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

import os

import pandas as pd
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader


def _get_data_intersection(df, intersection_path, id_col):
    with open(intersection_path) as intersection_file:
        intersection = intersection_file.read().splitlines()
    intersection.sort()

    # Note: the order of the intersection must be maintained
    intersection_df = df[df[id_col].isin(intersection)].copy()
    intersection_df["sort"] = pd.Categorical(intersection_df[id_col], categories=intersection, ordered=True)
    intersection_df = intersection_df.sort_values("sort")
    intersection_df = intersection_df.drop([id_col, "sort"], axis=1)

    if intersection_df.empty:
        raise ValueError("private set intersection must not be empty")

    return intersection_df


def _split_train_val(df, train_proportion):
    num_train = int(df.shape[0] * train_proportion)
    train_df = df.iloc[:num_train].copy()
    valid_df = df.iloc[num_train:].copy()

    return train_df, valid_df


class VerticalDataLoader(XGBDataLoader):
    def __init__(self, data_split_path, psi_path, id_col, label_owner, train_proportion):
        """Reads intersection of dataset and returns train and validation XGB data matrices with column split mode.

        Args:
            data_split_path: path to data split file
            psi_path: path to intersection file
            id_col: column id used for psi
            label_owner: client id that owns the label
            train_proportion: proportion of intersected data to use for training
        """
        self.data_split_path = data_split_path
        self.psi_path = psi_path
        self.id_col = id_col
        self.label_owner = label_owner
        self.train_proportion = train_proportion

    def load_data(self):
        client_data_split_path = self.data_split_path.replace("site-x", self.client_id)
        client_psi_path = self.psi_path.replace("site-x", self.client_id)

        data_split_dir = os.path.dirname(client_data_split_path)
        train_path = os.path.join(data_split_dir, "train.csv")
        valid_path = os.path.join(data_split_dir, "valid.csv")

        if not (os.path.exists(train_path) and os.path.exists(valid_path)):
            df = pd.read_csv(client_data_split_path)
            intersection_df = _get_data_intersection(df, client_psi_path, self.id_col)
            train_df, valid_df = _split_train_val(intersection_df, self.train_proportion)

            train_df.to_csv(path_or_buf=train_path, header=False, index=False)
            valid_df.to_csv(path_or_buf=valid_path, header=False, index=False)

        if self.client_id == self.label_owner:
            label = "&label_column=0"
        else:
            label = ""

        # for Vertical XGBoost, read from csv with label_column and set data_split_mode to 1 for column mode
        dtrain = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)
        dvalid = xgb.DMatrix(valid_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)

        return dtrain, dvalid
