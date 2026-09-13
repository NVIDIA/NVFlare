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

"""Bounded, revision-pinned retrieval of catalogued NVIDIA/NVFlare examples."""

import hashlib
import json
import re
import unicodedata
from pathlib import PurePosixPath
from urllib.parse import quote

import requests
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

REPOSITORY = "NVIDIA/NVFlare"
CATALOG_PATH = "examples/catalog.json"
PROVENANCE_FILE = ".nvflare-example.json"
MAX_FILES = 256
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{40}")
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
FALLBACK = "Full-repository alternative: https://github.com/NVIDIA/NVFlare#readme"


class ExampleError(Exception):
    def __init__(self, code, message, hint):
        super().__init__(message)
        self.code = code
        self.hint = hint


def reject(message):
    raise ExampleError("EXAMPLE_CONTENT_REJECTED", message, "Choose a catalogued example from a trusted release.")


def safe_path(value):
    """Accept portable relative paths, including on case-insensitive filesystems."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        reject("Invalid example path.")
    for part in value.split("/"):
        if (
            part in {"", ".", ".."}
            or part.lower() == ".git"
            or part.endswith((".", " "))
            or re.search(r'[\x00-\x1f\x7f\\:*?"<>|]', part)
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?", part)
        ):
            reject(f"Unsafe example path: {value!r}.")
    return value


def blob_sha(data):
    # Git's SHA-1 object ID is a content-integrity check, not an authentication primitive.
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False).hexdigest()


def _path_key(value):
    """Compare portable paths without changing the original Git path spelling."""
    # Case folding can decompose characters, so normalize its result as well.
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", value).casefold())


def selected_ref(version_info, explicit_ref=None):
    if explicit_ref:
        if len(explicit_ref) > 256 or re.search(r"[\x00-\x20\x7f]", explicit_ref):
            reject("Invalid Git reference.")
        return explicit_ref
    try:
        version = Version(version_info["version"])
    except (KeyError, InvalidVersion, TypeError):
        version = None
    revision = version_info.get("full-revisionid")
    if version_info.get("dirty"):
        raise ExampleError(
            "EXAMPLE_VERSION_UNKNOWN",
            "The installed source checkout has uncommitted changes.",
            "Commit the changes, or select the intended remote revision with --ref <commit>.",
        )
    if version and not version_info.get("error") and not version.is_devrelease and version.local is None:
        return f"refs/tags/{version}"
    if isinstance(revision, str) and SHA.fullmatch(revision):
        return revision
    raise ExampleError(
        "EXAMPLE_VERSION_UNKNOWN",
        "This development build has no usable source revision.",
        "Use --ref <tag-or-commit>; no default revision was guessed.",
    )


def validate_catalog(catalog):
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        reject("Unsupported example catalog schema.")
    entries = catalog.get("examples")
    if not isinstance(entries, dict) or not entries or len(entries) > 100:
        reject("Invalid example catalog entries.")
    paths = set()
    for name, entry in entries.items():
        if not NAME.fullmatch(name) or not isinstance(entry, dict):
            reject("Invalid catalog example name or entry.")
        path = safe_path(entry.get("path"))
        destination = entry.get("destination")
        if not path.startswith("examples/") or _path_key(path) in paths:
            reject("Catalog paths must be distinct directories under examples/.")
        paths.add(_path_key(path))
        if not isinstance(destination, str) or not NAME.fullmatch(destination):
            reject("Invalid default destination.")
        if not isinstance(entry.get("extra"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["extra"]):
            reject("Invalid dependency extra.")
        # This initial schema deliberately supports runnable Python entry points only.
        if entry.get("next_command") != ["python", "job.py"]:
            reject("Unsupported next command in example catalog.")
        try:
            specifier = entry["nvflare"]
            if not isinstance(specifier, str) or not specifier:
                reject("A version compatibility constraint is required.")
            SpecifierSet(specifier)
        except (KeyError, InvalidSpecifier):
            reject("Invalid NVFlare version constraint.")
    return entries


def check_compatibility(entry, version_info):
    try:
        installed = Version(version_info["version"])
    except (KeyError, InvalidVersion, TypeError):
        raise ExampleError(
            "EXAMPLE_VERSION_UNKNOWN",
            "Cannot check compatibility without an installed NVFlare version.",
            "Install a versioned NVFlare distribution and retry.",
        ) from None
    if not SpecifierSet(entry["nvflare"]).contains(installed, prereleases=True):
        raise ExampleError(
            "EXAMPLE_VERSION_INCOMPATIBLE",
            f"This example requires nvflare{entry['nvflare']}; installed version is {installed}.",
            "Install the matching NVFlare release, or select a compatible source with --ref.",
        )


class GitHubSource:
    def __init__(self):
        self.session = requests.Session()
        # Do not implicitly load credentials from .netrc. Keep normal TLS/proxy settings.
        self.session.auth = lambda request: request

    def close(self):
        self.session.close()

    def read(self, path, *, raw=False, limit=MAX_METADATA_BYTES, missing="EXAMPLE_SOURCE_NOT_FOUND", accept=None):
        host = "https://raw.githubusercontent.com" if raw else "https://api.github.com/repos"
        url = f"{host}/{REPOSITORY}/{path}"
        headers = {"Accept": accept or "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        try:
            with self.session.get(
                url, headers=headers, timeout=(10, 30), stream=True, allow_redirects=False
            ) as response:
                if response.status_code == 404:
                    raise ExampleError(
                        missing, f"Source unavailable: {url}", "Check the exact ref and catalog path. " + FALLBACK
                    )
                if response.status_code in {403, 429}:
                    raise ExampleError(
                        "EXAMPLE_ACCESS_LIMITED",
                        "GitHub denied the request or its public API rate limit was reached.",
                        "Retry after the GitHub rate limit resets. " + FALLBACK,
                    )
                if response.status_code != 200:
                    raise ExampleError(
                        "EXAMPLE_HTTP_ERROR",
                        f"GitHub returned HTTP {response.status_code}.",
                        "Check GitHub availability and retry. " + FALLBACK,
                    )
                data = bytearray()
                for chunk in response.iter_content(65536):
                    if len(data) + len(chunk) > limit:
                        raise ExampleError(
                            "EXAMPLE_LIMIT_EXCEEDED",
                            "Download exceeds the configured size limit.",
                            "Use a smaller catalogued example. " + FALLBACK,
                        )
                    data.extend(chunk)
                return bytes(data)
        except requests.RequestException:
            raise ExampleError(
                "EXAMPLE_NETWORK_ERROR",
                "Could not download from GitHub over HTTPS.",
                "Check network access, TLS certificates, and proxy settings, then retry. " + FALLBACK,
            ) from None

    def json(self, path):
        try:
            result = json.loads(self.read(path))
        except (ValueError, UnicodeError):
            reject("GitHub returned invalid JSON metadata.")
        if not isinstance(result, dict):
            reject("GitHub returned invalid metadata.")
        return result

    def resolve(self, ref):
        data = self.read(
            f"commits/{quote(ref, safe='')}",
            accept="application/vnd.github.sha",
            missing="EXAMPLE_REF_NOT_FOUND",
            limit=128,
        )
        commit = data.decode("ascii", errors="replace").strip()
        if not SHA.fullmatch(commit):
            reject("GitHub did not resolve the reference to a commit.")
        return commit

    def tree(self, revision, recursive=False):
        result = self.json(f"git/trees/{revision}" + ("?recursive=1" if recursive else ""))
        if result.get("truncated") is not False or not isinstance(result.get("tree"), list):
            reject("GitHub returned an incomplete tree; nothing was delivered.")
        return result["tree"]

    def locate(self, commit, path):
        revision = commit
        components = safe_path(path).split("/")
        for index, component in enumerate(components):
            matches = [item for item in self.tree(revision) if isinstance(item, dict) and item.get("path") == component]
            if len(matches) != 1:
                raise ExampleError(
                    "EXAMPLE_PATH_NOT_FOUND",
                    f"Path not found at {commit}: {path}",
                    "Select a revision containing this catalogued example. " + FALLBACK,
                )
            item = matches[0]
            if not SHA.fullmatch(str(item.get("sha", ""))):
                reject("Invalid object ID in GitHub tree.")
            if index < len(components) - 1 and (item.get("type"), item.get("mode")) != ("tree", "040000"):
                reject("The example path traverses an unsupported object.")
            revision = item["sha"]
        return item

    def file(self, commit, path, item, limit=MAX_FILE_BYTES):
        if item.get("type") != "blob" or item.get("mode") not in {"100644", "100755"}:
            reject("Only ordinary files are supported; symlinks and submodules are rejected.")
        data = self.read(f"{commit}/{quote(path, safe='/')}", raw=True, limit=limit)
        if len(data) != item.get("size") or blob_sha(data) != item.get("sha"):
            reject(f"Downloaded file does not match its Git object: {path}.")
        if data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            reject("Git LFS pointers are not supported in downloadable examples.")
        return data

    def catalog(self, commit):
        try:
            item = self.locate(commit, CATALOG_PATH)
        except ExampleError as error:
            if error.code != "EXAMPLE_PATH_NOT_FOUND":
                raise
            raise ExampleError(
                "EXAMPLE_CATALOG_UNAVAILABLE",
                f"No example catalog exists at {commit}.",
                "Select a release containing the examples command and catalog. " + FALLBACK,
            ) from None
        try:
            catalog = json.loads(self.file(commit, CATALOG_PATH, item, MAX_METADATA_BYTES))
        except (ValueError, UnicodeError):
            reject("Invalid example catalog JSON.")
        return validate_catalog(catalog)

    def inventory(self, commit, path):
        item = self.locate(commit, path)
        if (item.get("type"), item.get("mode")) != ("tree", "040000"):
            reject("The catalogued example must be a directory.")
        items = self.tree(item["sha"], recursive=True)
        return validate_inventory(items)


def validate_inventory(items):
    if not isinstance(items, list) or not items or len(items) > MAX_FILES:
        raise ExampleError("EXAMPLE_LIMIT_EXCEEDED", "Invalid or excessive example file count.", FALLBACK)
    files = []
    seen = set()
    spellings = {}
    total = 0
    for item in items:
        if not isinstance(item, dict):
            reject("Invalid example tree entry.")
        path = safe_path(item.get("path"))
        if _path_key(path) in seen or _path_key(path.split("/")[0]) == PROVENANCE_FILE:
            reject("Duplicate, Unicode-equivalent, case-colliding, or reserved example path.")
        seen.add(_path_key(path))
        parts = path.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            previous = spellings.setdefault(_path_key(prefix), prefix)
            if previous != prefix:
                reject("Unicode-equivalent or case-colliding file or directory names are not portable.")
        if not SHA.fullmatch(str(item.get("sha", ""))):
            reject("Invalid Git object ID.")
        kind = (item.get("type"), item.get("mode"))
        if kind == ("tree", "040000"):
            continue
        if kind not in {("blob", "100644"), ("blob", "100755")}:
            reject("Only ordinary files and directories are supported; symlinks and submodules are rejected.")
        size = item.get("size")
        if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
            raise ExampleError("EXAMPLE_LIMIT_EXCEEDED", f"File exceeds its size limit: {path}.", FALLBACK)
        total += size
        files.append({key: item[key] for key in ("path", "sha", "size", "type", "mode")})
    if total > MAX_TOTAL_BYTES:
        raise ExampleError("EXAMPLE_LIMIT_EXCEEDED", "Example exceeds the total size limit.", FALLBACK)
    names = {_path_key(item["path"]) for item in files}
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            reject("An example file is also used as a directory.")
    if not {"job.py", "README.md"} <= {item["path"] for item in files}:
        reject("A runnable example must contain job.py and README.md.")
    return files
