# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import argparse
import os

import pandas as pd

files = ["train", "test"]

bic_to_bank = {
    "ZHSZUS33": "Bank_1",
    "SHSHKHH1": "Bank_2",
    "YXRXGB22": "Bank_3",
    "WPUWDEFF": "Bank_4",
    "YMNYFRPP": "Bank_5",
    "FBSFCHZH": "Bank_6",
    "YSYCESMM": "Bank_7",
    "ZNZZAU3M": "Bank_8",
    "HCBHSGSG": "Bank_9",
    "XITXUS33": "Bank_10",
}

original_columns = [
    "UETR",
    "Timestamp",
    "Amount",
    "trans_volume",
    "total_amount",
    "average_amount",
    "hist_trans_volume",
    "hist_total_amount",
    "hist_average_amount",
    "x2_y1",
    "x3_y2",
]


def main():
    args = define_parser()
    root_path = args.input_dir
    original_feat_postfix = "_normalized.csv"
    embed_feat_postfix = "_embedding.csv"
    out_feat_postfix = "_combined.csv"

    for bic in bic_to_bank.keys():
        print("Processing BIC: ", bic)
        for file in files:
            original_feat_file = os.path.join(root_path, bic + "_" + bic_to_bank[bic], file + original_feat_postfix)
            embed_feat_file = os.path.join(root_path, bic + "_" + bic_to_bank[bic], file + embed_feat_postfix)
            out_feat_file = os.path.join(root_path, bic + "_" + bic_to_bank[bic], file + out_feat_postfix)

            # Load the original and embedding features
            original_feat = pd.read_csv(original_feat_file)
            embed_feat = pd.read_csv(embed_feat_file)

            # Select the columns of the original features
            original_feat = original_feat[original_columns]

            # Combine the features, matching the rows by "UETR"
            out_feat = pd.merge(original_feat, embed_feat, on="UETR")

            # Save the combined features
            out_feat.to_csv(out_feat_file, index=False)


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        nargs="?",
        default="/tmp/nvflare/xgb/credit_card",
        help="output directory, default to '/tmp/nvflare/xgb/credit_card'",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
