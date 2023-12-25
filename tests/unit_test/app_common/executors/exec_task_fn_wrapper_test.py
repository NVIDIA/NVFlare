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
from unittest.mock import patch

from nvflare.app_common.executors.exec_task_fn_wrapper import ExecTaskFuncWrapper
from nvflare.fuel.data_event.data_bus import DataBus


class TestExecTaskFuncWrapper(unittest.TestCase):
    def test_init_with_required_args(self):
        # Test initialization with a function that requires arguments
        task_fn_path = "nvflare.fuel.utils.class_utils.instantiate_class"
        task_fn_args = {"class_path": "foo", "init_params": {}}
        wrapper = ExecTaskFuncWrapper(task_fn_path, task_fn_args)

        self.assertEqual(wrapper.task_fn_path, task_fn_path)
        self.assertEqual(wrapper.task_fn_args, task_fn_args)
        self.assertTrue(wrapper.task_fn_requre_args)

    def test_init_with_optional_args(self):
        # Test initialization with a function that does not require arguments
        task_fn_path = "nvflare.utils.cli_utils.get_home_dir"
        task_fn_args = {"class_path": "foo", "init_params": {}}
        wrapper = ExecTaskFuncWrapper(task_fn_path, task_fn_args, read_interval=3.0)

        self.assertEqual(wrapper.task_fn_path, task_fn_path)
        self.assertEqual(wrapper.task_fn_args, task_fn_args)
        self.assertFalse(wrapper.task_fn_requre_args)

    def test_init_with_missing_required_args(self):
        # Test initialization with a function that requires arguments but none are provided
        task_fn_path = "nvflare.fuel.utils.class_utils.instantiate_class"
        # task_fn_args = {"class_path": "foo", "init_params": {}}

        with self.assertRaises(ValueError) as context:
            wrapper = ExecTaskFuncWrapper(task_fn_path)

        expected_msg = f"function '{task_fn_path}' requires arguments, but none provided"
        self.assertEqual(str(context.exception), expected_msg)

    def test_init_with_partial_missing_required_args(self):
        # Test initialization with a function that requires arguments but only partially are provided
        task_fn_path = "nvflare.fuel.utils.class_utils.instantiate_class"
        task_fn_args = {"init_params": {}}

        with self.assertRaises(ValueError) as context:
            wrapper = ExecTaskFuncWrapper(task_fn_path, task_fn_args)

        expected_msg = f"function '{task_fn_path}' requires 2 arguments, but 1 provided"
        self.assertEqual(str(context.exception), expected_msg)

    def test_init_with_partial_missing_required_args_with_default(self):
        # Test initialization with a function that requires arguments but only partially are provided
        # the missing arg has default value
        # def augment(to_dict: dict, from_dict: dict, from_override_to=False, append_list="components")
        task_fn_path = "nvflare.fuel.utils.dict_utils.augment"
        task_fn_args = {"to_dict": {}, "from_dict": {}}
        wrapper = ExecTaskFuncWrapper(task_fn_path, task_fn_args)

        self.assertEqual(wrapper.task_fn_path, task_fn_path)
        self.assertEqual(wrapper.task_fn_args, task_fn_args)
        self.assertTrue(wrapper.task_fn_requre_args)

    def test_run(self):
        message_bus = DataBus()
        message_bus.send_data("job_metadata", {})
        message_bus.send_data("mem_pipe", {})

        # Test the run method
        task_fn_path = "nvflare.fuel.utils.dict_utils.augment"
        task_fn_args = {"to_dict": {}, "from_dict": {}}
        wrapper = ExecTaskFuncWrapper(task_fn_path, task_fn_args)

        with patch.object(wrapper, "run") as mock_task_fn:
            wrapper.run()
            mock_task_fn.assert_called_once_with()
