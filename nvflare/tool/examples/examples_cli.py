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

"""Download examples selected by the installed NVFlare catalog."""

import json
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import quote

import requests

from nvflare.tool.cli_output import get_connect_timeout, is_json_mode, output_error_message, output_ok, print_human
from nvflare.tool.cli_schema import handle_schema_flag
from nvflare.tool.examples.catalog import load_catalog

EXAMPLE_CATALOG = load_catalog()
PROVENANCE_FILE = ".nvflare-example.json"
REPOSITORY = "NVIDIA/NVFlare"
MAX_EXAMPLE_FILES = 5000
MAX_EXAMPLE_BYTES = 128 * 1024 * 1024
_REVISION = re.compile(r"[0-9a-f]{40}")
_parsers = {}
_EXAMPLE_COMMANDS = [
    "nvflare examples list",
    "nvflare examples get hello-pt",
    "nvflare examples get hello-numpy --dest ./numpy-demo",
]


class ExampleError(Exception):
    def __init__(self, code, message, hint):
        super().__init__(message)
        self.code = code
        self.hint = hint


def def_examples_parser(sub_cmd):
    parser = sub_cmd.add_parser("examples", help="download an NVFlare example from GitHub")
    children = parser.add_subparsers(dest="examples_sub_cmd")
    list_parser = children.add_parser("list", help="list available example short names")
    get = children.add_parser("get", help="download one example into a new directory")
    get.add_argument("name", choices=sorted(EXAMPLE_CATALOG), help="example catalog short name")
    get.add_argument("--dest", help="new destination directory; default: example name in the current directory")
    _parsers.clear()
    _parsers.update({None: parser, "list": list_parser, "get": get})
    for value in _parsers.values():
        value.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    return {"examples": parser}


def _destination_exists(destination):
    raise ExampleError(
        "EXAMPLE_DESTINATION_EXISTS",
        f"Destination already exists: {destination}",
        "Use --dest <new-directory>, or move the existing directory before retrying.",
    )


def _source_revision(version_info):
    revision = version_info.get("full-revisionid")
    if isinstance(revision, str) and _REVISION.fullmatch(revision):
        return revision
    raise ExampleError(
        "EXAMPLE_VERSION_UNKNOWN",
        "This NVFlare installation does not identify its source revision.",
        "Install an official NVFlare wheel or an editable checkout with Git metadata.",
    )


def _download_example(revision, source_path, destination):
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{revision}?recursive=1"
    timeout = (get_connect_timeout(), 30)
    try:
        with requests.Session() as session:
            with session.get(tree_url, timeout=timeout) as response:
                response.raise_for_status()
                try:
                    metadata = response.json()
                except ValueError:
                    raise ExampleError(
                        "EXAMPLE_CONTENT_INVALID",
                        "GitHub returned invalid source metadata.",
                        "Retry later or use the example directly from the NVIDIA/NVFlare repository.",
                    ) from None
            if (
                not isinstance(metadata, dict)
                or metadata.get("truncated")
                or not isinstance(metadata.get("tree"), list)
            ):
                raise ExampleError(
                    "EXAMPLE_CONTENT_INVALID",
                    "GitHub returned an incomplete source tree.",
                    "Retry later or use the example directly from the NVIDIA/NVFlare repository.",
                )
            prefix = source_path + "/"
            files = [
                entry
                for entry in metadata["tree"]
                if entry.get("type") == "blob"
                and isinstance(entry.get("path"), str)
                and entry["path"].startswith(prefix)
            ]
            if not files:
                raise ExampleError(
                    "EXAMPLE_SOURCE_NOT_FOUND",
                    f"The release does not contain the catalog path: {source_path}",
                    "Use an example listed by this NVFlare installation.",
                )
            if any(not isinstance(entry.get("size"), int) or entry["size"] < 0 for entry in files):
                raise ExampleError(
                    "EXAMPLE_CONTENT_INVALID",
                    "GitHub returned invalid file metadata.",
                    "Retry later or use the example directly from the NVIDIA/NVFlare repository.",
                )
            total_size = sum(entry["size"] for entry in files)
            if len(files) > MAX_EXAMPLE_FILES or total_size > MAX_EXAMPLE_BYTES:
                raise ExampleError(
                    "EXAMPLE_DOWNLOAD_TOO_LARGE",
                    "The selected example exceeds the supported download size.",
                    "Use the example directly from the NVIDIA/NVFlare GitHub repository.",
                )
            for entry in files:
                relative = entry["path"][len(prefix) :]
                relative_parts = relative.split("/")
                if any(part in {"", ".", ".."} for part in relative_parts):
                    raise ExampleError(
                        "EXAMPLE_CONTENT_INVALID",
                        f"GitHub returned an invalid path for {source_path}.",
                        "Retry later or use the example directly from the NVIDIA/NVFlare repository.",
                    )
                target = destination.joinpath(*relative_parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                raw_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{quote(entry['path'], safe='/')}"
                with session.get(raw_url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    actual_size = 0
                    with target.open("wb") as target_file:
                        for chunk in response.iter_content(1024 * 1024):
                            actual_size += len(chunk)
                            if actual_size > entry["size"]:
                                raise ExampleError(
                                    "EXAMPLE_CONTENT_INVALID",
                                    f"GitHub returned too much data for {entry['path']}.",
                                    "Retry later or use the example directly from the NVIDIA/NVFlare repository.",
                                )
                            target_file.write(chunk)
                    if actual_size != entry["size"]:
                        raise ExampleError(
                            "EXAMPLE_CONTENT_INVALID",
                            f"GitHub returned incomplete data for {entry['path']}.",
                            "Retry later or use the example directly from the NVIDIA/NVFlare repository.",
                        )
                if entry.get("mode") == "100755":
                    target.chmod(0o755)
    except requests.RequestException as error:
        raise ExampleError(
            "EXAMPLE_NETWORK_ERROR",
            f"Could not download the NVFlare example: {error}",
            "Check GitHub access, your network connection, and proxy settings, then retry.",
        ) from None
    return tree_url


def get_example(version_info, *, name, destination=None):
    if name not in EXAMPLE_CATALOG:
        raise ExampleError(
            "EXAMPLE_UNKNOWN",
            f"Unknown example: {name}.",
            "Choose an example: " + ", ".join(sorted(EXAMPLE_CATALOG)),
        )
    destination = Path(destination or name).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        _destination_exists(destination)
    if not destination.parent.is_dir():
        raise ExampleError(
            "EXAMPLE_DESTINATION_INVALID",
            f"Destination parent does not exist: {destination.parent}",
            "Create the parent directory or choose --dest under an existing directory.",
        )

    revision = _source_revision(version_info)
    entry = EXAMPLE_CATALOG[name]
    try:
        destination.mkdir()
    except FileExistsError:
        _destination_exists(destination)
    tree_url = _download_example(revision, entry["source_path"], destination)

    requirements = destination / "requirements.txt"
    if not requirements.exists():
        requirements.write_text("", encoding="utf-8")
    readme = next(
        (candidate for candidate in (destination / "README.md", destination / "README.rst") if candidate.is_file()),
        None,
    )
    if readme is None:
        raise ExampleError(
            "EXAMPLE_CONTENT_INVALID",
            "The downloaded example does not contain a README.",
            "Use the example directly from the NVIDIA/NVFlare GitHub repository.",
        )
    provenance = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": revision,
        "example": name,
        "source_path": entry["source_path"],
        "source_url": f"https://github.com/{REPOSITORY}/tree/{revision}/{entry['source_path']}",
        "nvflare_version": version_info["version"],
    }
    (destination / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        **provenance,
        "tree_url": tree_url,
        "directory": str(destination),
        "setup_commands": [["pip", "install", "-r", "requirements.txt"]],
        "readme": str(readme),
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
        mutating=key == "get",
        idempotent=key == "list",
        retry_token={"supported": False},
    )
    if key not in {"list", "get"}:
        output_error_message(
            "INVALID_ARGS", "An examples subcommand is required.", "Run nvflare examples --help.", exit_code=4
        )

    if key == "list":
        examples = [
            {"name": name, "source_path": entry["source_path"]} for name, entry in sorted(EXAMPLE_CATALOG.items())
        ]
        if is_json_mode():
            output_ok({"examples": examples})
        else:
            name_width = max(len("SHORT NAME"), *(len(example["name"]) for example in examples))
            print_human(f"{'SHORT NAME':<{name_width}}  SOURCE PATH")
            for example in examples:
                print_human(f"{example['name']:<{name_width}}  {example['source_path']}")
        return

    from nvflare import _version

    try:
        result = get_example(_version.get_versions(), name=args.name, destination=args.dest)
        if is_json_mode():
            output_ok(result)
        else:
            print_human(f"Downloaded example: {result['directory']}")
            print_human(f"Source: {result['source_url']}\n")
            print_human("Next:")
            print_human(f"  cd {shlex.quote(result['directory'])}")
            print_human("  pip install -r requirements.txt")
            print_human(f"\nFollow {Path(result['readme']).name} for preparation and run instructions.")
    except ExampleError as error:
        output_error_message(error.code, str(error), error.hint, exit_code=4 if error.code == "INVALID_ARGS" else 1)
    except KeyboardInterrupt:
        output_error_message(
            "EXAMPLE_INTERRUPTED",
            "Example download interrupted.",
            "Remove any incomplete destination, then retry the command.",
            exit_code=130,
        )
    except (OSError, RuntimeError) as error:
        output_error_message(
            "EXAMPLE_IO_ERROR",
            f"Cannot download the example: {error}",
            "Remove any incomplete destination, then check the path, permissions, free space, and installation.",
            exit_code=1,
        )
