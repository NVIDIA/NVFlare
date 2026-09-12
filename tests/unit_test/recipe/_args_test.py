# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import importlib
import os
import warnings

import pytest


def test_recipe_args_import_strips_export_flags_from_sys_argv(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir", "/tmp/out", "--other", "value"])

    import nvflare.recipe._args as args_module

    with pytest.warns(UserWarning, match="system-level recipe arguments"):
        importlib.reload(args_module)

    assert sys.argv == ["python", "job.py", "--other", "value"]
    assert args_module._peek_recipe_args() == (True, "/tmp/out")


@pytest.mark.parametrize("flags", [["--log_config", "full"], ["--log_config=full"]])
def test_shared_logging_argument_overrides_script_default(monkeypatch, flags):
    import sys

    import nvflare.recipe._args as args_module
    from nvflare.recipe.sim_env import SimEnv

    monkeypatch.setenv("FL_LOG_LEVEL", "verbose")
    monkeypatch.setattr(args_module, "_RECIPE_LOG_CONFIG", None)
    monkeypatch.setattr(sys, "argv", ["job.py", *flags, "--rounds", "3"])
    importlib.reload(args_module)
    # Existing scripts may still define the option. Their parser's default must
    # not undo the shared command-line override when constructing SimEnv.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--log_config", default="concise")
    parser.add_argument("--rounds", type=int)
    args = parser.parse_args()
    assert args.rounds == 3
    assert SimEnv(num_clients=2, log_config=args.log_config).log_config == "full"
    assert os.environ["FL_LOG_LEVEL"] == "full"
    assert args_module._peek_recipe_args() == (False, args_module.DEFAULT_EXPORT_DIR)


@pytest.mark.parametrize("flags", [["--log_config"], ["--log_config", "--rounds"], ["--log_config="]])
def test_shared_logging_argument_requires_value(monkeypatch, flags):
    import sys

    import nvflare.recipe._args as args_module

    monkeypatch.setattr(args_module, "_RECIPE_LOG_CONFIG", None)
    monkeypatch.setenv("FL_LOG_LEVEL", "concise")
    original_argv = ["job.py", *flags]
    monkeypatch.setattr(sys, "argv", original_argv.copy())
    with pytest.raises(SystemExit) as ex:
        importlib.reload(args_module)
    assert ex.value.code == 2
    assert sys.argv == original_argv
    assert os.environ["FL_LOG_LEVEL"] == "concise"


def test_system_arguments_stop_at_double_dash(monkeypatch):
    import sys

    import nvflare.recipe._args as args_module

    monkeypatch.setattr(args_module, "_RECIPE_LOG_CONFIG", None)
    monkeypatch.setenv("FL_LOG_LEVEL", "concise")
    original_argv = ["job.py", "--", "--log_config", "full", "--export"]
    monkeypatch.setattr(sys, "argv", original_argv.copy())
    importlib.reload(args_module)
    assert sys.argv == original_argv
    assert os.environ["FL_LOG_LEVEL"] == "concise"
    assert args_module._peek_recipe_args() == (False, args_module.DEFAULT_EXPORT_DIR)


def test_recipe_args_import_bare_export_uses_default_dir(monkeypatch):
    import sys

    # The canonical invocation: `python job.py --export` with no --export-dir.
    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--other"])

    import nvflare.recipe._args as args_module

    with pytest.warns(UserWarning, match="system-level recipe arguments"):
        importlib.reload(args_module)

    assert sys.argv == ["python", "job.py", "--other"]
    assert args_module._peek_recipe_args() == (True, args_module.DEFAULT_EXPORT_DIR)


def test_recipe_args_import_strips_export_dir_equals_form(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir=out", "--other"])

    import nvflare.recipe._args as args_module

    with pytest.warns(UserWarning, match="system-level recipe arguments"):
        importlib.reload(args_module)

    assert sys.argv == ["python", "job.py", "--other"]
    assert args_module._peek_recipe_args() == (True, "out")


@pytest.mark.parametrize(
    ("export_dir_args", "expected_export_dir"),
    [
        (["--export-dir", "/tmp/user-output"], "/tmp/user-output"),
        (["--export-dir=/tmp/user-output"], "/tmp/user-output"),
    ],
)
def test_recipe_args_import_warns_when_export_dir_is_unused(monkeypatch, export_dir_args, expected_export_dir):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--rounds", "3", *export_dir_args])

    import nvflare.recipe._args as args_module

    with pytest.warns(UserWarning, match="system-level recipe arguments.*without '--export'"):
        importlib.reload(args_module)

    assert sys.argv == ["python", "job.py", "--rounds", "3"]
    assert args_module._peek_recipe_args() == (False, expected_export_dir)


def test_recipe_args_import_warns_script_parser_about_reserved_args(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["job.py", "--export", "--export-dir", "/tmp/out"])

    import nvflare.recipe._args as args_module

    with pytest.warns(UserWarning, match="system-level recipe arguments.*Rename any script-defined arguments"):
        importlib.reload(args_module)

    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-dir")
    args = parser.parse_args()

    assert args.export is False
    assert args.export_dir is None
    assert sys.argv == ["job.py"]
    assert args_module._peek_recipe_args() == (True, "/tmp/out")


@pytest.mark.parametrize(
    "recipe_args",
    [
        ["--export", "--export-dir", "/tmp/user-output"],
        ["--export-dir", "/tmp/user-output"],
    ],
)
def test_consume_recipe_args_warning_as_error_preserves_sys_argv(monkeypatch, recipe_args):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py"])

    import nvflare.recipe._args as args_module

    importlib.reload(args_module)
    monkeypatch.setattr(args_module, "_CONSUMED", False)
    original_argv = ["job.py", *recipe_args, "--other"]
    monkeypatch.setattr(sys, "argv", original_argv.copy())

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="system-level recipe arguments"):
            args_module._consume_recipe_args()

    assert sys.argv == original_argv


@pytest.mark.parametrize(
    ("local_flag", "should_parse"),
    [
        ("--max_train_samples", True),
        ("--max_train_sample", False),
        ("--max_trian_samples", False),
    ],
)
def test_recipe_export_flags_allow_strict_local_argument_parsing(monkeypatch, local_flag, should_parse):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["job.py", "--export", "--export-dir", "/tmp/out", local_flag, "100"],
    )

    import nvflare.recipe._args as args_module

    with pytest.warns(UserWarning, match="system-level recipe arguments"):
        importlib.reload(args_module)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--max_train_samples", type=int)

    if should_parse:
        args = parser.parse_args()
        assert args.max_train_samples == 100
        assert args_module._peek_recipe_args() == (True, "/tmp/out")
    else:
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args()
        assert exc_info.value.code == 2


def test_consume_recipe_args_dangling_export_dir_does_not_raise(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir"])

    import nvflare.recipe._args as args_module

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        importlib.reload(args_module)  # malformed input must not raise or warn on import

    # Transactional: the whole pass is abandoned -- export stays disabled and sys.argv
    # is left untouched so the caller's own parser can surface the leftover flags.
    assert args_module._peek_recipe_args() == (False, args_module.DEFAULT_EXPORT_DIR)
    assert sys.argv == ["python", "job.py", "--export", "--export-dir"]


def test_consume_recipe_args_freezes_import_time_decision(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir"])

    import nvflare.recipe._args as args_module

    importlib.reload(args_module)

    # A later direct call returns the recorded import-time decision even if sys.argv
    # has since changed -- it must not re-scan and flip to a different answer.
    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir", "out"])
    assert args_module._consume_recipe_args() == (False, args_module.DEFAULT_EXPORT_DIR)
