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
import unicodedata
from pathlib import Path
from urllib.parse import quote

import requests

from nvflare.tool.cli_output import get_connect_timeout, is_json_mode, output_error_message, output_ok, print_human
from nvflare.tool.cli_schema import handle_schema_flag
from nvflare.tool.examples.catalog import load_catalog

PROVENANCE_FILE = ".nvflare-example.json"
REPOSITORY = "NVIDIA/NVFlare"
_REVISION = re.compile(r"[0-9a-f]{40}")
_parsers = {}
_EXAMPLE_COMMANDS = [
    "nvflare examples list",
    "nvflare examples get hello-pt",
    "nvflare examples get hello-numpy --dest ./numpy-demo",
    "nvflare examples revision",
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
    get.add_argument("name", metavar="NAME", help="example short name from 'nvflare examples list'")
    get.add_argument("--dest", help="new destination directory; default: example name in the current directory")
    revision = children.add_parser("revision", help="print the source revision recorded by a downloaded example")
    revision.add_argument("--dir", default=".", help="downloaded example directory; default: current directory")
    _parsers.clear()
    _parsers.update({None: parser, "list": list_parser, "get": get, "revision": revision})
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


def _example_revision(directory):
    provenance_file = Path(directory).expanduser().absolute() / PROVENANCE_FILE
    try:
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        revision = provenance.get("revision")
    except (OSError, ValueError, AttributeError):
        revision = None
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ExampleError(
            "EXAMPLE_PROVENANCE_INVALID",
            f"Cannot read an NVFlare example revision from {provenance_file}.",
            "Run this command inside a directory created by 'nvflare examples get', or pass --dir <directory>.",
        )
    return {"revision": revision, "provenance_file": str(provenance_file)}


def _load_example_catalog():
    try:
        catalog, errors = load_catalog()
    except (OSError, ValueError) as error:
        raise ExampleError(
            "EXAMPLE_CATALOG_INVALID",
            f"Cannot load the installed example catalog: {error}",
            "Reinstall NVFlare, or use examples directly from the NVIDIA/NVFlare GitHub repository.",
        ) from None
    if not catalog:
        raise ExampleError(
            "EXAMPLE_CATALOG_INVALID",
            "The installed example catalog contains no valid entries.",
            "Reinstall NVFlare, or use examples directly from the NVIDIA/NVFlare GitHub repository.",
        )
    return catalog, errors


def _content_error(
    message="GitHub returned invalid source metadata.", hint="Retry or use the example directly from GitHub."
):
    return ExampleError("EXAMPLE_CONTENT_INVALID", message, hint)


def _validate_tree_entries(entries, source_path):
    files = []
    entry_keys = set()
    directory_paths = {}
    file_keys = set()

    def path_key(parts):
        return tuple(unicodedata.normalize("NFC", unicodedata.normalize("NFC", part).casefold()) for part in parts)

    def collision(path):
        raise _content_error(
            f"The example contains colliding paths: {path}.",
            "Use the example directly from a Git checkout.",
        )

    def add_directory(parts, path):
        key = path_key(parts)
        original = tuple(parts)
        if key in file_keys or (key in directory_paths and directory_paths[key] != original):
            collision(path)
        directory_paths[key] = original

    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("type"), str)
            or not isinstance(entry.get("path"), str)
        ):
            raise _content_error()
        entry_type = entry["type"]
        entry_mode = entry.get("mode")
        if entry_type == "commit" or entry_mode in {"120000", "160000"}:
            raise _content_error(
                f"The example contains an unsupported symlink or submodule: {entry['path']}.",
                "Use the example directly from a Git checkout.",
            )
        if entry_type not in {"blob", "tree"}:
            raise _content_error()
        relative_parts = entry["path"].split("/")
        if any(part in {"", ".", ".."} for part in relative_parts):
            raise _content_error(f"GitHub returned an invalid path for {source_path}.")
        key = path_key(relative_parts)
        if key in entry_keys:
            collision(entry["path"])
        entry_keys.add(key)
        for length in range(1, len(relative_parts)):
            add_directory(relative_parts[:length], entry["path"])
        if entry_type == "tree":
            add_directory(relative_parts, entry["path"])
            continue
        if key in directory_paths:
            collision(entry["path"])
        file_keys.add(key)
        files.append((entry, relative_parts))
    if not files:
        raise _content_error("GitHub returned no files for the example.")
    return files


def _download_example(revision, source_path, destination):
    encoded_source_path = quote(source_path, safe="/")
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{revision}:{encoded_source_path}?recursive=1"
    timeout = (get_connect_timeout(), 30)
    destination_created = False
    try:
        with requests.Session() as session:
            with session.get(tree_url, timeout=timeout) as response:
                if response.status_code == 404:
                    raise ExampleError(
                        "EXAMPLE_SOURCE_NOT_FOUND",
                        f"GitHub does not contain this installation's source revision or catalog path: "
                        f"{revision}:{source_path}",
                        "For an editable install, push the commit or check out a revision available on GitHub; "
                        "otherwise reinstall NVFlare.",
                    )
                response.raise_for_status()
                try:
                    metadata = response.json()
                    entries = metadata["tree"]
                except (ValueError, KeyError, TypeError):
                    metadata = {}
                    entries = None
                if metadata.get("truncated") or not isinstance(entries, list) or not entries:
                    raise _content_error() from None
            files = _validate_tree_entries(entries, source_path)
            try:
                destination.mkdir()
            except FileExistsError:
                _destination_exists(destination)
            destination_created = True
            for entry, relative_parts in files:
                relative = entry["path"]
                target = destination.joinpath(*relative_parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                repository_path = f"{source_path}/{relative}"
                raw_url = (
                    f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{quote(repository_path, safe='/')}"
                )
                with session.get(raw_url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with target.open("wb") as target_file:
                        for chunk in response.iter_content(1024 * 1024):
                            target_file.write(chunk)
                if entry.get("mode") == "100755":
                    target.chmod(0o755)
    except requests.RequestException as error:
        cleanup = f"Remove the incomplete destination directory at {destination}. " if destination_created else ""
        raise ExampleError(
            "EXAMPLE_NETWORK_ERROR",
            f"Could not download the NVFlare example: {error}",
            f"{cleanup}Check GitHub access and your network settings, then retry.",
        ) from None


def get_example(version_info, catalog, *, name, destination=None):
    if name not in catalog:
        raise ExampleError(
            "EXAMPLE_UNKNOWN",
            f"Unknown example: {name}.",
            "Run 'nvflare examples list' to choose an available short name.",
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
    entry = catalog[name]
    _download_example(revision, entry["source_path"], destination)
    warnings = [
        {
            "code": "EXAMPLE_DEPENDENCY_GUIDANCE",
            "message": "Preserve the installed NVFlare distribution when setting up this example.",
            "hint": (
                "Skip any README or dependency-file instruction that installs nvflare or nvflare-nightly. "
                "Add required extras to the same stable, nightly, or editable distribution, then install only "
                "the remaining dependencies."
            ),
        }
    ]

    readme = next(
        (candidate for candidate in (destination / "README.md", destination / "README.rst") if candidate.is_file()),
        None,
    )
    if readme is None:
        warnings.append(
            {
                "code": "EXAMPLE_README_MISSING",
                "message": "The downloaded example does not contain a root README.",
                "hint": "Inspect the downloaded files for dependency, preparation, and run instructions.",
            }
        )
    provenance = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": revision,
        "example": name,
        "source_path": entry["source_path"],
        "source_url": f"https://github.com/{REPOSITORY}/tree/{revision}/{entry['source_path']}",
        "nvflare_version": version_info.get("version"),
    }
    (destination / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        **provenance,
        "directory": str(destination),
        "readme": str(readme) if readme else None,
        "warnings": warnings,
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
        idempotent=key in {"list", "revision"},
        retry_token={"supported": False},
    )
    if key not in {"list", "get", "revision"}:
        output_error_message(
            "INVALID_ARGS", "An examples subcommand is required.", "Run nvflare examples --help.", exit_code=4
        )

    try:
        if key == "revision":
            result = _example_revision(args.dir)
            if is_json_mode():
                output_ok(result)
            else:
                print_human(result["revision"])
            return

        catalog, catalog_errors = _load_example_catalog()
        if key == "list":
            examples = [
                {"name": name, "category": entry["category"], "source_path": entry["source_path"]}
                for name, entry in sorted(catalog.items(), key=lambda item: (item[1]["category"], item[0]))
            ]
            if is_json_mode():
                output_ok({"examples": examples, "catalog_errors": catalog_errors})
            else:
                name_width = max(len("SHORT NAME"), *(len(example["name"]) for example in examples))
                category = None
                for example in examples:
                    if example["category"] != category:
                        if category is not None:
                            print_human("")
                        category = example["category"]
                        print_human(category.replace("-", " ").upper())
                        print_human(f"  {'SHORT NAME':<{name_width}}  SOURCE PATH")
                    print_human(f"  {example['name']:<{name_width}}  {example['source_path']}")
                if catalog_errors:
                    print_human("\nSkipped invalid catalog entries:")
                    for error in catalog_errors:
                        print_human(f"  {error['name']}: {error['error']}")
            return

        invalid_entry = next((error for error in catalog_errors if error["name"] == args.name), None)
        if invalid_entry:
            raise ExampleError(
                "EXAMPLE_CATALOG_INVALID",
                f"The installed catalog entry '{args.name}' is invalid: {invalid_entry['error']}.",
                "Reinstall NVFlare, or use the example directly from the NVIDIA/NVFlare GitHub repository.",
            )

        from nvflare import _version

        result = get_example(_version.get_versions(), catalog, name=args.name, destination=args.dest)
        if is_json_mode():
            output_ok(result)
        else:
            print_human(f"Downloaded example: {result['directory']}")
            print_human(f"Source: {result['source_url']}\n")
            for warning in result["warnings"]:
                print_human(f"Warning: {warning['message']}")
                print_human(f"{warning['hint']}\n")
            print_human("Next:")
            print_human(f"  cd {shlex.quote(result['directory'])}")
            if result["readme"]:
                print_human(
                    f"\nFollow {Path(result['readme']).name} for dependency, preparation, and run instructions."
                )
    except ExampleError as error:
        output_error_message(error.code, str(error), error.hint, exit_code=1)
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
