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

import pandas as pd

pd_readers = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "json": pd.read_json,
}


def _to_data_tuple(data):
    data_num = data.shape[0]
    # split to feature and label
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x.to_numpy(), y.to_numpy(), data_num


def get_pandas_reader(data_path: str):
    from nvflare.app_common.utils.file_utils import get_file_format

    file_format = get_file_format(data_path)
    reader = pd_readers.get(file_format, None)
    if reader is None:
        raise ValueError(f"no pandas reader for given file format {file_format}")
    return reader


def load_data(data_path: str, require_header: bool = False):
    reader = get_pandas_reader(data_path)
    if hasattr(reader, "header"):
        if require_header:
            data = reader(data_path)
        else:
            data = reader(data_path, header=None)
    else:
        data = reader(data_path)

    return _to_data_tuple(data)


def load_data_for_range(data_path: str, start: int, end: int, require_header: bool = False):
    reader = get_pandas_reader(data_path)

    if hasattr(reader, "skiprows"):
        data_size = end - start
        if hasattr(reader, "header") and not require_header:
            data = reader(data_path, header=None, skiprows=start, nrows=data_size)
        else:
            data = reader(data_path, skiprows=start, nrows=data_size)
    else:
        data = reader(data_path).iloc[start:end]

    return _to_data_tuple(data)
