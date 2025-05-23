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

import pytest

from nvflare.tool.job import job_cli
from nvflare.tool.job.job_cli import convert_args_list_to_dict


class TestJobCLI:
    @pytest.mark.parametrize("inputs, result", [(["a=1", "b=2", "c = 3"], dict(a="1", b="2", c="3"))])
    def test_convert_args_list_to_dict(self, inputs, result):
        r = convert_args_list_to_dict(inputs)
        assert r == result

    @pytest.mark.parametrize(
        "directory, path, expected",
        [("/home/user/project", "/home/user/project/subdir", True), (".", ".", True), ("./code", ".", False)],
    )
    def test_is_sub_dir(self, path, directory, expected):
        print(f"{input=}, {directory=}, {expected=}")
        assert expected == job_cli.is_subdir(path, directory)
