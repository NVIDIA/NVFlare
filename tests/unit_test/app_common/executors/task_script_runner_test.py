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

import builtins
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import pytest

from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.in_process.api import TOPIC_ABORT, TOPIC_STOP
from nvflare.fuel.utils.secret_utils import secret_file_ref


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

    def _assert_argv_path(self, wrapper, expected_path, expected_args=None):
        argv = wrapper.get_sys_argv()
        expected_suffix = os.path.relpath(expected_path, self.nvflare_root)
        assert os.path.normpath(argv[0]).endswith(os.path.normpath(expected_suffix))
        assert argv[1:] == (expected_args or [])

    def test_app_scripts_and_args(self):
        script_path = "nvflare/cli.py"
        script_args = "--batch_size 4"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path, script_args=script_args)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self._assert_argv_path(wrapper, os.path.join(self.nvflare_root, "nvflare", "cli.py"), ["--batch_size", "4"])

    def test_app_scripts_and_args2(self):
        # curr_dir = os.getcwd()
        script_path = "nvflare/cli.py"
        script_args = "--batch_size 4"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path, script_args=script_args)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self._assert_argv_path(wrapper, os.path.join(self.nvflare_root, "nvflare", "cli.py"), ["--batch_size", "4"])

    def test_secret_ref_args_resolved_from_env(self):
        script_path = "nvflare/cli.py"
        script_args = "--api_key ${secret:TEST_SECRET_REF_VAR}"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path, script_args=script_args)

        with patch.dict(os.environ, {"TEST_SECRET_REF_VAR": "resolved secret"}):
            argv = wrapper.get_sys_argv()

        # the value from the site env is injected as a single argument, even with whitespace
        assert argv[1:] == ["--api_key", "resolved secret"]
        # the configured args (what job configs carry) still hold only the placeholder
        assert "${secret:TEST_SECRET_REF_VAR}" in wrapper.script_args
        assert "resolved secret" not in wrapper.script_args

    def test_secret_ref_in_quoted_composite_argument(self):
        wrapper = TaskScriptRunner(
            custom_dir=self.nvflare_root,
            script_path="nvflare/cli.py",
            script_args='--authorization "Bearer ${secret:TEST_SECRET_REF_VAR}"',
        )

        with patch.dict(os.environ, {"TEST_SECRET_REF_VAR": "resolved secret"}):
            argv = wrapper.get_sys_argv()

        assert argv[1:] == ["--authorization", "Bearer resolved secret"]

    def test_secret_ref_does_not_change_unrelated_backslashes_or_quotes(self):
        wrapper = TaskScriptRunner(
            custom_dir=self.nvflare_root,
            script_path="nvflare/cli.py",
            script_args=(
                r'--regex \d+ --path C:\data\x --legacy "two words" '
                r'--authorization "Bearer ${secret:TEST_SECRET_REF_VAR}"'
            ),
        )

        with patch.dict(os.environ, {"TEST_SECRET_REF_VAR": "resolved"}):
            argv = wrapper.get_sys_argv()

        assert argv[1:] == [
            "--regex",
            r"\d+",
            "--path",
            r"C:\data\x",
            "--legacy",
            '"two',
            'words"',
            "--authorization",
            "Bearer resolved",
        ]

    def test_secret_file_ref_content_is_one_argument(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = os.path.join(temp_dir, "api-key")
            with open(secret_path, "w", encoding="utf-8") as secret_file:
                secret_file.write("resolved secret --not-option")

            placeholder = secret_file_ref(secret_path)
            wrapper = TaskScriptRunner(
                custom_dir=self.nvflare_root,
                script_path="nvflare/cli.py",
                script_args=f"--api_key {placeholder}",
            )

            argv = wrapper.get_sys_argv()

        assert argv[1:] == ["--api_key", "resolved secret --not-option"]
        assert placeholder in wrapper.script_args
        assert "resolved secret" not in wrapper.script_args

    def test_secret_ref_args_missing_env_var_raises(self):
        script_path = "nvflare/cli.py"
        script_args = "--api_key ${secret:TEST_UNSET_SECRET_VAR}"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path, script_args=script_args)

        os.environ.pop("TEST_UNSET_SECRET_VAR", None)
        with pytest.raises(ValueError, match="TEST_UNSET_SECRET_VAR"):
            wrapper.get_sys_argv()

    def test_run_restores_sys_argv_when_script_fails(self):
        wrapper = TaskScriptRunner(
            custom_dir=self.nvflare_root,
            script_path="nvflare/cli.py",
            script_args="--api_key ${secret:TEST_SECRET_REF_VAR}",
            redirect_print_to_log=False,
        )
        original_argv = sys.argv

        with patch.dict(os.environ, {"TEST_SECRET_REF_VAR": "resolved secret"}):
            with patch(
                "nvflare.app_common.executors.task_script_runner.runpy.run_path", side_effect=RuntimeError("boom")
            ):
                with pytest.raises(RuntimeError, match="boom"):
                    wrapper.run()

        assert sys.argv is original_argv

    def test_run_skips_script_when_runtime_was_already_released(self):
        wrapper = TaskScriptRunner(
            custom_dir=self.nvflare_root,
            script_path="nvflare/cli.py",
            script_args="--api_key ${secret:TEST_UNSET_SECRET_VAR}",
            redirect_print_to_log=False,
        )
        wrapper.release_runtime()
        os.environ.pop("TEST_UNSET_SECRET_VAR", None)

        with patch("nvflare.app_common.executors.task_script_runner.runpy.run_path") as mock_run_path:
            wrapper.run()

        mock_run_path.assert_not_called()

    def test_app_scripts_with_sub_dirs1(self):
        # curr_dir = os.getcwd()
        script_path = "nvflare/__init__.py"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self._assert_argv_path(wrapper, os.path.join(self.nvflare_root, "nvflare", "__init__.py"))

    def test_app_scripts_with_sub_dirs2(self):
        # curr_dir = os.getcwd()
        script_path = "nvflare/app_common/executors/__init__.py"
        wrapper = TaskScriptRunner(custom_dir=self.nvflare_root, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self._assert_argv_path(
            wrapper, os.path.join(self.nvflare_root, "nvflare", "app_common", "executors", "__init__.py")
        )

    def test_app_scripts_with_sub_dirs3(self):
        # curr_dir = os.getcwd()
        script_path = "executors/task_script_runner.py"
        sub_dir = os.path.join(self.nvflare_root, "nvflare/app_common")
        wrapper = TaskScriptRunner(custom_dir=sub_dir, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self._assert_argv_path(
            wrapper, os.path.join(self.nvflare_root, "nvflare", "app_common", "executors", "task_script_runner.py")
        )

    def test_app_scripts_with_sub_dirs4(self):
        # curr_dir = os.getcwd()
        script_path = "in_process/api.py"
        sub_dir = os.path.join(self.nvflare_root, "nvflare/client")
        wrapper = TaskScriptRunner(custom_dir=sub_dir, script_path=script_path)

        self.assertTrue(wrapper.script_full_path.endswith(script_path))
        self._assert_argv_path(wrapper, os.path.join(self.nvflare_root, "nvflare", "client", "in_process", "api.py"))

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
            self._assert_argv_path(wrapper, expected_path, ["--batch_size", "4"])
            wrapper.run()
        finally:
            sys.path = old_sys_path

    def test_run_redirects_print_from_imported_module(self):
        old_sys_path = sys.path.copy()
        original_print = builtins.print
        helper_module = "task_script_runner_print_helper"

        with tempfile.TemporaryDirectory() as temp_dir:
            helper_path = os.path.join(temp_dir, f"{helper_module}.py")
            script_path = os.path.join(temp_dir, "train.py")
            with open(helper_path, "w", encoding="utf-8") as helper_file:
                helper_file.write('print("helper output")\n')
            with open(script_path, "w", encoding="utf-8") as script_file:
                script_file.write(f'import {helper_module}\nprint("script output")\n')

            sys.path.insert(0, temp_dir)
            try:
                wrapper = TaskScriptRunner(custom_dir=temp_dir, script_path="train.py")
                with patch.object(wrapper.logger, "info") as mock_log:
                    wrapper.run()

                mock_log.assert_any_call("helper output")
                mock_log.assert_any_call("script output")
                assert builtins.print is original_print
            finally:
                sys.modules.pop(helper_module, None)
                sys.path[:] = old_sys_path

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
            self._assert_argv_path(wrapper, expected_path, ["--batch_size", "4"])
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
            self._assert_argv_path(wrapper, expected_path, ["--batch_size", "4"])
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


def test_print_redirection_is_scoped_to_runner_thread(tmp_path, capsys):
    (tmp_path / "train.py").write_text("")
    runner = TaskScriptRunner(custom_dir=str(tmp_path), script_path="train.py")
    trainer_started = threading.Event()
    release_trainer = threading.Event()

    def run_path(*args, **kwargs):
        print("trainer output")
        trainer_started.set()
        release_trainer.wait(timeout=5.0)

    with (
        patch("nvflare.app_common.executors.task_script_runner.runpy.run_path", side_effect=run_path),
        patch.object(TaskScriptRunner.logger, "info") as log_info,
    ):
        trainer_thread = threading.Thread(target=runner.run)
        trainer_thread.start()
        try:
            assert trainer_started.wait(timeout=2.0)
            print("main-thread output")
            assert capsys.readouterr().out == "main-thread output\n"
            log_info.assert_any_call("trainer output")
        finally:
            release_trainer.set()
            trainer_thread.join(timeout=2.0)

    assert not trainer_thread.is_alive()
