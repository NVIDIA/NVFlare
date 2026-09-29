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

"""Import-time handling of shared Recipe command-line arguments."""

import argparse
import os
import sys
import warnings

DEFAULT_EXPORT_DIR = "./fl_job"

_CONSUMED = False
_RECIPE_LOG_CONFIG = None
_RECIPE_ARG_ERROR = None


def _record_error(message):
    global _CONSUMED, _RECIPE_ARG_ERROR
    _RECIPE_ARG_ERROR = message
    _CONSUMED = True
    return False, DEFAULT_EXPORT_DIR


def _consume_recipe_args() -> tuple:
    """Consume shared arguments without exposing them to the script's parser."""
    global _CONSUMED, _RECIPE_LOG_CONFIG
    if _CONSUMED:
        return _RECIPE_EXPORT, _RECIPE_EXPORT_DIR

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-dir")
    parser.add_argument("--log_config")
    errors = {
        "--export-dir": "--export-dir requires a non-empty directory",
        "--log_config": "--log_config requires a non-empty configuration",
    }
    try:
        args, remaining = parser.parse_known_args(sys.argv[1:])
    except argparse.ArgumentError as e:
        return _record_error(errors.get(e.argument_name, str(e)))

    for option, value in (("--export-dir", args.export_dir), ("--log_config", args.log_config)):
        if value is not None and not value.strip():
            return _record_error(errors[option])

    export_dir_seen = args.export_dir is not None
    export_dir = args.export_dir if export_dir_seen else DEFAULT_EXPORT_DIR
    if args.export or export_dir_seen:
        message = (
            "NVFlare reserves '--export' and '--export-dir' as system-level recipe arguments and consumes them "
            "before the script's argument parser runs. Rename any script-defined arguments that use these names."
        )
        if export_dir_seen and not args.export:
            message += " '--export-dir' was provided without '--export', so the directory will not be used."
        warnings.warn(message, UserWarning, stacklevel=2)

    sys.argv[1:] = remaining
    _RECIPE_LOG_CONFIG = args.log_config
    if args.log_config is not None:
        # Reuse the existing process/subprocess logging configuration path.
        os.environ["FL_LOG_LEVEL"] = args.log_config
    _CONSUMED = True
    return args.export, export_dir


# Intentional import-time sys.argv mutation: strip shared recipe arguments before
# any ArgumentParser.parse_args() call in job.py runs. Doing this lazily (e.g. inside
# execute()) would be too late if the caller calls parse_args() first, which is the
# common pattern. The mutation is safe because job.py is always the process entry point.
_RECIPE_EXPORT, _RECIPE_EXPORT_DIR = _consume_recipe_args()


def _peek_recipe_args() -> tuple:
    """Return the export flags consumed at import time."""
    return _RECIPE_EXPORT, _RECIPE_EXPORT_DIR
