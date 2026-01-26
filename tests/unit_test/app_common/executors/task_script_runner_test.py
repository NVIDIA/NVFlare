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
import shutil
import sys
import unittest

import pytest

from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.in_process.api import TOPIC_ABORT, TOPIC_STOP


class TestTaskScriptRunner(unittest.TestCase):
    def setUp(self) -> None:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        splits = file_dir.split(os.sep)
        self.nvflare_root = os.sep.join(splits[0:-4])
        # sometimes a build dir is generated, we need to remove
        # to ensure test correctness
        build_dir = os.path.join(self.nvflare_root, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)

    def test_app_scripts_and_args(self):
        script_path = "nvflare/cli.py"
        script_args = "--batch_size 4"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path, script_args=script_args)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self.assertEqual(
            wrapper.get_sys_argv(), [os.path.join(self.nvflare_root, "nvflare", "cli.py"), "--batch_size", "4"]
        )

    def test_app_scripts_and_args2(self):
        # curr_dir = os.getcwd()
        script_path = "nvflare/cli.py"
        script_args = "--batch_size 4"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path, script_args=script_args)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self.assertEqual(
            wrapper.get_sys_argv(), [os.path.join(self.nvflare_root, "nvflare", "cli.py"), "--batch_size", "4"]
        )

    def test_app_scripts_with_sub_dirs1(self):
        # curr_dir = os.getcwd()
        script_path = "nvflare/__init__.py"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self.assertEqual(wrapper.get_sys_argv(), [os.path.join(self.nvflare_root, "nvflare", "__init__.py")])

    def test_app_scripts_with_sub_dirs2(self):
        # curr_dir = os.getcwd()
        script_path = "nvflare/app_common/executors/__init__.py"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self.assertEqual(
            wrapper.get_sys_argv(),
            [os.path.join(self.nvflare_root, "nvflare", "app_common", "executors", "__init__.py")],
        )

    def test_app_scripts_with_sub_dirs3(self):
        # curr_dir = os.getcwd()
        script_path = "executors/task_script_runner.py"
        sub_dir = os.path.join(self.nvflare_root, "nvflare/app_common")
        wrapper = TaskScriptRunner(custom_dir=sub_dir, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self.assertEqual(
            wrapper.get_sys_argv(),
            [os.path.join(self.nvflare_root, "nvflare", "app_common", "executors", "task_script_runner.py")],
        )

    def test_app_scripts_with_sub_dirs4(self):
        # curr_dir = os.getcwd()
        script_path = "in_process/api.py"
        sub_dir = os.path.join(self.nvflare_root, "nvflare/client")
        wrapper = TaskScriptRunner(custom_dir=sub_dir, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self.assertEqual(
            wrapper.get_sys_argv(), [os.path.join(self.nvflare_root, "nvflare", "client", "in_process", "api.py")]
        )

    def test_file_not_found_with_exception(self):
        # curr_dir = os.getcwd()
        script_path = "in_process/api.py"
        with pytest.raises(ValueError, match="Can not find in_process/api.py"):
            sub_dir = os.path.join(self.nvflare_root, "site-1")
            wrapper = TaskScriptRunner(custom_dir=sub_dir, script_path=script_path)
            self.assertTrue(wrapper.script_full_path.endswith(script_path))
            self.assertEqual(
                wrapper.get_sys_argv(), [os.path.join(self.nvflare_root, "nvflare", "client", "in_process", "api.py")]
            )

    def test_run_scripts_with_sub_dirs(self):
        old_sys_path = sys.path
        script_args = "--batch_size 4"
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/server/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom"))

        try:
            script_path = "train.py"
            sub_dir = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1")
            wrapper = TaskScriptRunner(
                custom_dir=sub_dir, script_path=script_path, script_args=script_args, redirect_print_to_log=False
            )
            self.assertTrue(wrapper.script_full_path.endswith(script_path))
            expected_path = os.path.join(
                self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom/train.py"
            )
            self.assertEqual(wrapper.get_sys_argv(), [expected_path, "--batch_size", "4"])
            wrapper.run()
        finally:
            sys.path = old_sys_path

    def test_run_scripts_with_sub_dirs2(self):
        old_sys_path = sys.path
        script_args = "--batch_size 4"
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/server/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom"))

        try:
            script_path = "train.py"
            sub_dir = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/server")
            wrapper = TaskScriptRunner(
                custom_dir=sub_dir, script_path=script_path, script_args=script_args, redirect_print_to_log=False
            )
            self.assertTrue(wrapper.script_full_path.endswith(script_path))
            expected_path = os.path.join(
                self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/server/custom/train.py"
            )
            self.assertEqual(wrapper.get_sys_argv(), [expected_path, "--batch_size", "4"])
            wrapper.run()
        finally:
            sys.path = old_sys_path

    def test_run_scripts_with_sub_dirs3(self):
        old_sys_path = sys.path
        script_args = "--batch_size 4"
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/server/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom"))

        try:
            script_path = "src/train.py"
            wrapper = TaskScriptRunner(
                custom_dir=self.nvflare_root,
                script_path=script_path,
                script_args=script_args,
                redirect_print_to_log=False,
            )
            self.assertTrue(wrapper.script_full_path.endswith(script_path))
            expected_path = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/custom/src/train.py")
            self.assertEqual(wrapper.get_sys_argv(), [expected_path, "--batch_size", "4"])
            wrapper.run()
        finally:
            sys.path = old_sys_path

    def test_run_failed_scripts(self):
        old_sys_path = sys.path
        script_args = "--batch_size 4"
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/server/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom"))

        try:
            script_path = "failed_train.py"
            sub_dir = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1")
            wrapper = TaskScriptRunner(
                custom_dir=sub_dir, script_path=script_path, script_args=script_args, redirect_print_to_log=False
            )
            wrapper.event_manager.data_bus.subscribe([TOPIC_ABORT, TOPIC_STOP], self.abort_callback)

            self.assertTrue(wrapper.script_full_path.endswith(script_path))
            with pytest.raises(ValueError, match="failed to train model"):
                # 1 ) check if the exception is through,
                # 2 ) more important to see if the callback is trigger.
                wrapper.run()
        finally:
            sys.path = old_sys_path

    def abort_callback(self, topic, data, databus):
        print("\n ===== calling abort_callback begin")
        # assert failure here will not cause test to fail
        self.assertEqual(topic, TOPIC_ABORT)
        print("\n ===== calling abort_callback end")

    def test_run_relative_import_scripts(self):
        old_sys_path = sys.path
        script_args = "--batch_size 4"
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/server/custom"))
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom"))

        try:
            script_path = "relative_import_train.py"
            sub_dir = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1")
            wrapper = TaskScriptRunner(
                custom_dir=sub_dir, script_path=script_path, script_args=script_args, redirect_print_to_log=False
            )
            self.assertTrue(wrapper.script_full_path.endswith(script_path))
            path = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom")
            msg = f"attempted relative import with no known parent package, the relative import is not support. python import is based off the sys.path: {path}"
            with pytest.raises(ImportError, match=msg):
                # check the ImportError
                wrapper.run()
        finally:
            sys.path = old_sys_path

    def test_run_abs_path_scripts(self):
        old_sys_path = sys.path
        script_args = "--batch_size 4"

        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom"))

        try:
            # path doesn't exist
            script_path = "/foo/dummy/train.py"
            with pytest.raises(ValueError, match="script_path='/foo/dummy/train.py' not found"):
                sub_dir = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1")
                wrapper = TaskScriptRunner(
                    custom_dir=sub_dir, script_path=script_path, script_args=script_args, redirect_print_to_log=False
                )
        finally:
            sys.path = old_sys_path

    def test_run_abs_path_scripts2(self):
        old_sys_path = sys.path
        script_args = "--batch_size 4"
        sys.path.append(os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom"))

        try:
            script_path = os.path.join(
                self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1/custom/train.py"
            )
            sub_dir = os.path.join(self.nvflare_root, "tests/unit_test/data/jobs/in_proc_job/site-1")
            wrapper = TaskScriptRunner(
                custom_dir=sub_dir, script_path=script_path, script_args=script_args, redirect_print_to_log=False
            )
            wrapper.run()
        finally:
            sys.path = old_sys_path

    def test_empty_custom_dir(self):
        with pytest.raises(Exception):
            script_path = "cli.py"
            script_args = "--batch_size 4"
            wrapper = TaskScriptRunner(custom_dir="", script_path=script_path, script_args=script_args)

    def test_empty_script_path(self):
        with pytest.raises(Exception):
            script_args = "--batch_size 4"
            wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path="", script_args=script_args)
