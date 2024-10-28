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


import unittest
from unittest.mock import MagicMock, patch

from nvflare.fuel.utils.function_utils import find_task_fn


class TestFindTaskFn(unittest.TestCase):
    @patch("importlib.import_module")
    def test_find_task_fn_with_module(self, mock_import_module):
        # Test find_task_fn when a module is specified in task_fn_path
        task_fn_path = "nvflare.utils.cli_utils.get_home_dir"
        mock_module = MagicMock()
        mock_import_module.return_value = mock_module

        result = find_task_fn(task_fn_path)

        mock_import_module.assert_called_once_with("nvflare.utils.cli_utils")
        self.assertTrue(callable(result))

    def test_find_task_fn_without_module(self):
        # Test find_task_fn when no module is specified in task_fn_path
        task_fn_path = "get_home_dir"
        with self.assertRaises(ModuleNotFoundError) as context:
            result = find_task_fn(task_fn_path)
