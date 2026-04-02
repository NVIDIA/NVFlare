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

import glob
import os
import traceback
from typing import Dict, Optional

import pandas as pd
import sklearn
from misc.data import all_model_parameters, clean_dataframe, numerical_features, prepare_dataset
from misc.data_io import load_csv_data_from_path, print_directory_tree, validate_data_features
from misc.experiments import data_paths

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore


class FinancialStatistics(DFStatisticsCore):
    def __init__(
        self,
        data_selection,
        data_features=["FRAUD_FLAG"],
    ):
        super().__init__()
        self.data_selection = data_selection
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.data_features = data_features

    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"load data for client {client_name}")

        # Display directory tree of /workspace/dataset
        self.log_info(fl_ctx, f"\n=== Directory tree of {client_name} at /workspace/dataset ===")
        self.log_info(fl_ctx, "/workspace/dataset")
        print_directory_tree("/workspace/dataset", max_depth=3, endswith=".csv")  # TODO: user logger instead of print
        self.log_info(fl_ctx, "=" * 45 + "\n")

        # Load CSV data using the utility function
        self.log_info(
            fl_ctx,
            f"Loading data for client {client_name} with selection {self.data_selection}",
        )

        data_selection_paths = data_paths[self.data_selection][client_name]
        data_root = data_selection_paths["data_root"]
        train_data_path = os.path.join(data_root, data_selection_paths["train_data_path"])
        test_data_path_pattern = os.path.join(data_root, data_selection_paths["test_data_path"])
        scaling_data_path = os.path.join(data_root, data_selection_paths["scaling_data_path"])

        if not os.path.isfile(train_data_path):
            raise FileNotFoundError(f"No valid train filepath at: {train_data_path}")

        # Check if test_data_path contains wildcards
        if "*" in test_data_path_pattern or "?" in test_data_path_pattern:
            # Use glob to find matching files
            test_data_paths = sorted(glob.glob(test_data_path_pattern))
            if not test_data_paths:
                raise FileNotFoundError(f"No test files found matching pattern: {test_data_path_pattern}")
            self.log_info(
                fl_ctx,
                f"Found {len(test_data_paths)} test files matching pattern: {test_data_path_pattern}",
            )
            for path in test_data_paths:
                self.log_info(fl_ctx, f"  - {path}")

            assert len(test_data_paths) == 4, "Expected 4 test files, got " + str(len(test_data_paths))
        else:
            # Single test file
            if not os.path.isfile(test_data_path_pattern):
                raise FileNotFoundError(f"No valid test filepath at: {test_data_path_pattern}")
            test_data_paths = [test_data_path_pattern]
            self.log_info(fl_ctx, f"Test data path: {test_data_path_pattern}")

        if not os.path.isfile(scaling_data_path):
            self.log_info(fl_ctx, f"[WARNING] No valid scaling filepath at: {scaling_data_path}")

        self.log_info(fl_ctx, f"Train data path: {train_data_path}")
        self.log_info(fl_ctx, f"Scaling data path: {scaling_data_path}")

        try:
            # Load CSV data using the utility function
            df_train = load_csv_data_from_path(
                data_path=train_data_path,
                data_features=None,  # all features are loaded
                na_values="?",
            )

            # Load all test files
            test_dataframes = {}
            for test_path in test_data_paths:
                # Extract a meaningful name from the file path
                test_name = os.path.basename(test_path).replace(".csv", "")
                df_test_tmp = load_csv_data_from_path(
                    data_path=test_path,
                    data_features=None,  # all features are loaded
                    na_values="?",
                )
                test_dataframes[test_name] = df_test_tmp
                self.log_info(
                    fl_ctx,
                    f"Loaded test dataset '{test_name}' with {len(df_test_tmp)} samples",
                )

            # Load scaler data if available
            # Load and concatenate all scaler data files
            if os.path.isfile(scaling_data_path):
                df_scaling = load_csv_data_from_path(
                    data_path=scaling_data_path,
                    data_features=None,  # all features are loaded
                    na_values="?",
                )

                # Concatenate all scaler dataframes
                global_scaler = sklearn.preprocessing.StandardScaler()
                global_scaler = global_scaler.fit(prepare_dataset(df_scaling).loc[:, numerical_features])
            else:
                self.log_info(fl_ctx, "[WARNING] No valid scaler data files found")
                df_scaling = None
                global_scaler = None

            # Prepare dataset
            self.log_info(fl_ctx, f"Preparing data with features: {all_model_parameters}")
            df_train = prepare_dataset(df_train, scaler=global_scaler)

            # Prepare all test datasets
            for test_name, df_test in test_dataframes.items():
                test_dataframes[test_name] = prepare_dataset(df_test, scaler=global_scaler)

            # Validate the loaded data
            validate_data_features(df_train, self.data_features)
            df_train = df_train.loc[:, self.data_features]
            for test_name, df_test in test_dataframes.items():
                validate_data_features(df_test, self.data_features)
                test_dataframes[test_name] = df_test.loc[:, self.data_features]

            # Clean data: remove NaN, inf, and ensure data is numeric
            self.log_info(fl_ctx, "Cleaning data: removing NaN and inf values")

            # Clean train data
            df_train = clean_dataframe(df_train, dataset_name="train", verbose=True)

            # Ensure train data is not empty
            if len(df_train) == 0:
                raise ValueError(f"Train dataset is empty after cleaning for client {client_name}")

            # Clean test datasets
            for test_name, df_test in test_dataframes.items():
                df_test = clean_dataframe(df_test, dataset_name=test_name, verbose=True)
                test_dataframes[test_name] = df_test

                # Ensure test data is not empty
                if len(df_test) == 0:
                    raise ValueError(f"Test dataset '{test_name}' is empty after cleaning for client {client_name}")

            if global_scaler is not None:
                self.log_info(fl_ctx, f"Global scaler: {global_scaler}")

            self.log_info(fl_ctx, f"Load data done for client {client_name}")

            # Return dataframes as a dictionary
            dataframes = {}
            dataframes["train"] = df_train
            for test_name, df_test in test_dataframes.items():
                test_name = test_name.replace("[", "").replace("]", "")
                test_name = test_name.split("_", 1)[1] if "_" in test_name else test_name
                dataframes[test_name] = df_test
                break  # only return one test dataframe for now

            self.log_info(fl_ctx, f"Return dataframes for client {client_name}:")
            for name, df_test in dataframes.items():
                self.log_info(fl_ctx, f"  - {name}: {df_test.shape}")
            self.log_info(fl_ctx, "=" * 45 + "\n")
            return dataframes
        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Load data for client {client_name} failed! {e}")

    def initialize(self, fl_ctx: FLContext):
        self.data = self.load_data(fl_ctx)
