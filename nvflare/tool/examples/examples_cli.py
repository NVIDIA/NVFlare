# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Retrieve editable example source without Git or a full repository checkout."""

import shlex
import signal
import sys

from nvflare.tool.cli_output import is_json_mode, output_error_message, output_ok, print_human
from nvflare.tool.cli_schema import handle_schema_flag
from nvflare.tool.examples.source import ExampleError

_parsers = {}
_EXAMPLES = [
    "nvflare examples get hello-pt",
    "nvflare examples get hello-pt --dest ./my-hello-pt --refresh",
    "nvflare examples get hello-pt --ref main --format json",
    "nvflare examples cache clear",
]


def def_examples_parser(sub_cmd):
    parser = sub_cmd.add_parser("examples", help="download release-matched runnable examples")
    children = parser.add_subparsers(dest="examples_sub_cmd")
    get = children.add_parser("get", help="download one example into a new directory")
    get.add_argument("name", help="catalog short name, e.g. hello-pt")
    get.add_argument("--ref", help="explicit tag, branch, or commit; default: installed version's source")
    get.add_argument("--dest", help="new destination directory; default: catalog destination in current directory")
    get.add_argument("--refresh", action="store_true", help="download again even if a validated cache entry exists")
    cache = children.add_parser("cache", help="manage the local example download cache")
    cache_children = cache.add_subparsers(dest="examples_cache_cmd")
    clear = cache_children.add_parser("clear", help="remove all cached example downloads; keep delivered workspaces")
    _parsers.clear()
    _parsers.update({None: parser, "get": get, "cache": cache, "cache clear": clear})
    for value in _parsers.values():
        value.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    return {"examples": parser}


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def handle_examples_cmd(args):
    sub = getattr(args, "examples_sub_cmd", None)
    action = getattr(args, "examples_cache_cmd", None)
    key = f"cache {action}" if sub == "cache" and action is not None else sub
    if key not in _parsers:
        output_error_message(
            "INVALID_ARGS", f"Unknown examples subcommand: {key}.", "Run nvflare examples --help.", exit_code=4
        )
    handle_schema_flag(
        _parsers[key],
        "nvflare examples" + (f" {key}" if key else ""),
        _EXAMPLES,
        sys.argv[1:],
        streaming=False,
        output_modes=["json"],
        mutating=True,
        idempotent=key == "cache clear",
        retry_token={"supported": False},
    )
    if key not in {"get", "cache clear"}:
        output_error_message(
            "INVALID_ARGS", "An examples subcommand is required.", "Run nvflare examples --help.", exit_code=4
        )

    from nvflare import _version
    from nvflare.tool.examples.store import ExampleStore

    store = ExampleStore()
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        if key == "cache clear":
            output_ok(store.clear())
            return
        result = store.get(
            _version.get_versions(),
            name=args.name,
            ref=args.ref,
            destination=args.dest,
            refresh=args.refresh,
        )
        if is_json_mode():
            output_ok(result)
        else:
            print_human(f"{result['example']}: {result['cache_status']}")
            print_human(f"Source: {result['source_url']}")
            print_human(f"Commit: {result['commit']}")
            print_human(f"Created: {result['directory']}\n")
            print_human("Next:")
            print_human(f"  cd {shlex.quote(result['directory'])}")
            print_human("  " + shlex.join(result["next_command"]))
            print_human("\nSee README.md for dependencies, customization, and further examples.")
    except ExampleError as error:
        output_error_message(error.code, str(error), error.hint, exit_code=4 if error.code == "INVALID_ARGS" else 1)
    except KeyboardInterrupt:
        output_error_message(
            "EXAMPLE_INTERRUPTED",
            "Example retrieval interrupted.",
            "Retry the command; an incomplete download was not delivered as the destination.",
            exit_code=130,
        )
    except OSError as error:
        output_error_message(
            "EXAMPLE_IO_ERROR",
            f"Cannot write or read the example files: {error}",
            "Check free disk space and permissions for the cache and destination, then retry.",
            exit_code=1,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
        store.source.close()
