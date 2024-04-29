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
import os
import unittest

from nvflare.app_common.executors.task_script_runner import TaskScriptRunner


class TestExecTaskFuncWrapper(unittest.TestCase):
    def test_app_scripts_and_args(self):
        curr_dir = os.getcwd()
        script_path = "cli.py"
        script_args = "--batch_size 4"
        wrapper = TaskScriptRunner(script_path=script_path, script_args=script_args)

        self.assertTrue(wrapper.script_path.endswith(script_path))
        self.assertEqual(wrapper.get_sys_argv(), [os.path.join(curr_dir, "nvflare", "cli.py"), "--batch_size", "4"])
