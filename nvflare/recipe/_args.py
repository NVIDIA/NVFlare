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

# Single source of truth for the default export dir. Kept in the current working
# directory on purpose: an export is a user-requested artifact, so it lands next to
# where the script ran (discoverable and predictable), and a relative path stays
# portable across OSes -- unlike a fixed /tmp path, which is non-portable on Windows
# and collision-prone on shared hosts.
DEFAULT_EXPORT_DIR = "./fl_job"

_CONSUMED = False
_RECIPE_LOG_CONFIG = None


def _consume_recipe_args() -> tuple:
    """Consume shared export/logging arguments and return (export, export_dir).

    Called once at module import time so that the caller's argparse never sees
    these flags regardless of the order in which parse_args() and execute() appear
    in job.py.

    Transactional: sys.argv is only mutated if the parse is clean. A malformed
    (dangling) --export-dir aborts the pass without mutating sys.argv and without
    enabling export, so a malformed import can neither raise nor silently export.
    The decision is frozen after the first call so repeated direct calls return the
    recorded import-time result rather than re-scanning a since-mutated sys.argv.
    """
    global _CONSUMED, _RECIPE_LOG_CONFIG
    if _CONSUMED:
        return _RECIPE_EXPORT, _RECIPE_EXPORT_DIR

    argv = sys.argv[1:]
    export = False
    export_dir = DEFAULT_EXPORT_DIR
    export_dir_seen = False
    log_config = None
    logging_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    logging_parser.add_argument("--log_config", metavar="CONFIG")
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] == "--":
            remaining.extend(argv[i:])
            break
        elif argv[i] == "--log_config" or argv[i].startswith("--log_config="):
            count = 2 if argv[i] == "--log_config" else 1
            log_config = logging_parser.parse_args(argv[i : i + count]).log_config
            if not log_config.strip():
                logging_parser.error("--log_config requires a non-empty configuration")
            i += count
        elif argv[i] == "--export":
            export = True
            i += 1
        elif argv[i] == "--export-dir":
            if i + 1 >= len(argv):
                # Dangling --export-dir with no value: abort the entire pass. Do not
                # mutate sys.argv and do not enable export. Leaving argv intact lets the
                # caller's own parser surface the leftover flags; enabling export here
                # could export to the default dir against the user's intent (e.g. under
                # parse_known_args()). Freeze the decision so a later direct call returns
                # it instead of re-scanning a since-mutated sys.argv.
                _CONSUMED = True
                return False, DEFAULT_EXPORT_DIR
            export_dir = argv[i + 1]
            export_dir_seen = True
            i += 2
        elif argv[i].startswith("--export-dir="):
            export_dir = argv[i].split("=", 1)[1]
            export_dir_seen = True
            i += 1
        else:
            remaining.append(argv[i])
            i += 1

    if export or export_dir_seen:
        message = (
            "NVFlare reserves '--export' and '--export-dir' as system-level recipe arguments and consumes them "
            "before the script's argument parser runs. Rename any script-defined arguments that use these names."
        )
        if export_dir_seen and not export:
            message += " '--export-dir' was provided without '--export', so the directory will not be used."
        warnings.warn(
            message,
            UserWarning,
            stacklevel=2,
        )

    sys.argv[1:] = remaining
    _RECIPE_LOG_CONFIG = log_config
    if log_config is not None:
        # Reuse the existing process/subprocess logging configuration path.
        os.environ["FL_LOG_LEVEL"] = log_config
    _CONSUMED = True
    return export, export_dir


# Intentional import-time sys.argv mutation: strip shared recipe arguments before
# any ArgumentParser.parse_args() call in job.py runs. Doing this lazily (e.g. inside
# execute()) would be too late if the caller calls parse_args() first, which is the
# common pattern. The mutation is safe because job.py is always the process entry point.
_RECIPE_EXPORT, _RECIPE_EXPORT_DIR = _consume_recipe_args()


def _peek_recipe_args() -> tuple:
    """Return the export flags consumed at import time."""
    return _RECIPE_EXPORT, _RECIPE_EXPORT_DIR
