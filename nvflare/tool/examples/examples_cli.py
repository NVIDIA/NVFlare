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

"""Copy release-matched example source bundled with NVFlare."""

import json
import shlex
import shutil
import sys
from importlib import resources
from pathlib import Path

from nvflare.tool.cli_output import is_json_mode, output_error_message, output_ok, print_human
from nvflare.tool.cli_schema import handle_schema_flag
from nvflare.tool.examples import HELLO_PT_FILES

PROVENANCE_FILE = ".nvflare-example.json"
EXAMPLES = {
    "hello-pt": {
        "source_path": "examples/hello-world/hello-pt",
        "destination": "hello-pt",
        "required_extra": "PT",
        "next_command": ["python", "job.py"],
        "files": HELLO_PT_FILES,
    }
}
_parsers = {}
_EXAMPLE_COMMANDS = ["nvflare examples get hello-pt", "nvflare examples get hello-pt --dest ./my-hello-pt"]


class ExampleError(Exception):
    def __init__(self, code, message, hint):
        super().__init__(message)
        self.code = code
        self.hint = hint


def def_examples_parser(sub_cmd):
    parser = sub_cmd.add_parser("examples", help="copy a runnable example bundled with NVFlare")
    children = parser.add_subparsers(dest="examples_sub_cmd")
    get = children.add_parser("get", help="copy one example into a new directory")
    get.add_argument("name", help="bundled example short name, e.g. hello-pt")
    get.add_argument("--dest", help="new destination directory; default: example name in the current directory")
    _parsers.clear()
    _parsers.update({None: parser, "get": get})
    for value in _parsers.values():
        value.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    return {"examples": parser}


def _example_source(name):
    bundled = resources.files("nvflare.tool.examples").joinpath("data", name)
    if bundled.is_dir():
        return bundled

    # Editable installs load this module from the checkout, where setup.py does
    # not retain its temporary package-data copy.
    checkout_root = Path(__file__).resolve().parents[3]
    checkout = checkout_root / EXAMPLES[name]["source_path"]
    if (checkout_root / "setup.py").is_file() and checkout.is_dir():
        return checkout
    raise OSError(f"The installed NVFlare package does not contain the bundled {name} example")


def _destination_exists(destination):
    raise ExampleError(
        "EXAMPLE_DESTINATION_EXISTS",
        f"Destination already exists: {destination}",
        "Use --dest <new-directory>, or move the existing directory before retrying.",
    )


def get_example(version_info, *, name, destination=None):
    if name not in EXAMPLES:
        raise ExampleError(
            "EXAMPLE_UNKNOWN",
            f"Unknown bundled example: {name}.",
            "Choose a bundled example: " + ", ".join(sorted(EXAMPLES)),
        )
    entry = EXAMPLES[name]
    destination = Path(destination or entry["destination"]).expanduser().absolute()
    if not destination.parent.is_dir():
        raise ExampleError(
            "EXAMPLE_DESTINATION_INVALID",
            f"Destination parent does not exist: {destination.parent}",
            "Create the parent directory or choose --dest under an existing directory.",
        )

    source = _example_source(name)
    provenance = {
        "schema_version": 1,
        "example": name,
        "source": "bundled",
        "source_path": entry["source_path"],
        "nvflare_version": version_info["version"],
    }
    allowed = set(entry["files"])
    try:
        shutil.copytree(source, destination, dirs_exist_ok=False, ignore=lambda _, names: set(names) - allowed)
    except FileExistsError:
        _destination_exists(destination)
    (destination / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        **provenance,
        "directory": str(destination),
        "required_extra": entry["required_extra"],
        "next_command": entry["next_command"],
        "readme": str(destination / "README.md"),
    }


def handle_examples_cmd(args):
    key = getattr(args, "examples_sub_cmd", None)
    if key not in _parsers:
        output_error_message(
            "INVALID_ARGS", f"Unknown examples subcommand: {key}.", "Run nvflare examples --help.", exit_code=4
        )
    handle_schema_flag(
        _parsers[key],
        "nvflare examples" + (f" {key}" if key else ""),
        _EXAMPLE_COMMANDS,
        sys.argv[1:],
        streaming=False,
        output_modes=["json"],
        mutating=True,
        idempotent=False,
        retry_token={"supported": False},
    )
    if key != "get":
        output_error_message(
            "INVALID_ARGS", "The examples get subcommand is required.", "Run nvflare examples --help.", exit_code=4
        )

    from nvflare import _version

    try:
        result = get_example(_version.get_versions(), name=args.name, destination=args.dest)
        if is_json_mode():
            output_ok(result)
        else:
            print_human(f"Created bundled example: {result['directory']}\n")
            print_human("Next:")
            print_human(f"  cd {shlex.quote(result['directory'])}")
            print_human("  " + shlex.join(result["next_command"]))
            print_human("\nSee README.md for dependencies and customization.")
    except ExampleError as error:
        output_error_message(error.code, str(error), error.hint, exit_code=4 if error.code == "INVALID_ARGS" else 1)
    except KeyboardInterrupt:
        output_error_message(
            "EXAMPLE_INTERRUPTED",
            "Example copy interrupted.",
            "Remove any incomplete destination, then retry the command.",
            exit_code=130,
        )
    except (OSError, RuntimeError) as error:
        output_error_message(
            "EXAMPLE_IO_ERROR",
            f"Cannot copy the bundled example: {error}",
            "Remove any incomplete destination, then check the path, permissions, free space, and installation.",
            exit_code=1,
        )
