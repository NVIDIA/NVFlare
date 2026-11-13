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

"""Build and publish the open-source Dashboard frontend without Python build hooks."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

PREFIX = "/nvflare-dashboard"
MANIFEST_NAME = "dashboard-build.json"
MANIFEST_SCHEMA = 1
EXPECTED_PAGES = (
    "index.html",
    "404.html",
    "downloads.html",
    "logout.html",
    "project-admin-dashboard.html",
    "project-configuration.html",
    "registration-form.html",
    "server-config.html",
    "site-dashboard.html",
    "user-dashboard.html",
)
REQUIRED_INPUTS = ("package.json", "package-lock.json", "next.config.js", "siteparams.json", "tsconfig.json")
OPTIONAL_INPUTS = (
    ".npmrc",
    ".nvmrc",
    ".babelrc",
    ".babelrc.json",
    ".babelrc.js",
    "babel.config.js",
    "postcss.config.js",
    "postcss.config.mjs",
    "postcss.config.cjs",
    "tailwind.config.js",
    "tailwind.config.ts",
)
INPUT_DIRECTORIES = ("src", "styles", "public", "scripts")
CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)


class BuildError(RuntimeError):
    """A build input, generated asset, or publication step is invalid."""


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.references = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in {"script", "img", "source"} and attributes.get("src"):
            self.references.append(attributes["src"])
        elif (
            tag == "link"
            and attributes.get("href")
            and set(attributes.get("rel", "").split()) & {"stylesheet", "preload", "modulepreload", "icon", "manifest"}
        ):
            self.references.append(attributes["href"])


def source_fingerprint(ui_directory):
    """Hash build inputs, excluding tests, docs, generated files, and caches."""
    ui_directory = Path(ui_directory).resolve()
    paths = []
    for name in REQUIRED_INPUTS:
        path = ui_directory / name
        if not path.is_file():
            raise BuildError(f"Missing Dashboard build input: {path}")
        paths.append(path)
    paths.extend(
        ui_directory / name
        for name in OPTIONAL_INPUTS
        if (ui_directory / name).exists() or (ui_directory / name).is_symlink()
    )
    for name in INPUT_DIRECTORIES:
        directory = ui_directory / name
        if directory.is_symlink():
            raise BuildError(f"Dashboard build inputs must not be symbolic links: {directory}")
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_symlink():
                    raise BuildError(f"Dashboard build inputs must not be symbolic links: {path}")
                if path.is_file():
                    paths.append(path)
    digest = hashlib.sha256(b"nvflare-dashboard-source-v1\0")
    for path in sorted(set(paths), key=lambda item: item.relative_to(ui_directory).as_posix()):
        if path.is_symlink():
            raise BuildError(f"Dashboard build inputs must not be symbolic links: {path}")
        relative = path.relative_to(ui_directory).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _asset_path(reference, document, export_directory):
    reference = reference.strip()
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    if path.startswith(PREFIX + "/"):
        candidate = export_directory / path[len(PREFIX) + 1 :]
    elif path.startswith("/"):
        raise BuildError(f"Asset URL must use {PREFIX}/: {reference} in {document.name}")
    else:
        candidate = document.parent / path
    candidate = candidate.resolve()
    if not candidate.is_relative_to(export_directory):
        raise BuildError(f"Asset reference escapes the exported Dashboard: {reference}")
    if not candidate.is_file():
        raise BuildError(f"Missing exported asset: {reference} in {document.relative_to(export_directory)}")
    return candidate


def validate_export(export_directory):
    """Check Flask's expected pages, notices, and local HTML/CSS asset URLs."""
    export_directory = Path(export_directory).resolve()
    if not export_directory.is_dir():
        raise BuildError(f"Dashboard export is missing: {export_directory}")
    for name in (*EXPECTED_PAGES, "THIRD_PARTY_NOTICES.txt"):
        path = export_directory / name
        if not path.is_file() or not path.stat().st_size:
            raise BuildError(f"Dashboard export is missing required file: {name}")
    references = 0
    for document in export_directory.rglob("*"):
        if document.is_symlink():
            raise BuildError(f"Dashboard export must not contain symbolic links: {document}")
        if not document.is_file():
            continue
        if document.suffix == ".map":
            raise BuildError(f"Private browser source map must not be published: {document}")
        if document.suffix == ".html":
            parser = AssetReferences()
            parser.feed(document.read_text(encoding="utf-8"))
            urls = parser.references
        elif document.suffix == ".css":
            urls = [match.group(2) for match in CSS_URL.finditer(document.read_text(encoding="utf-8"))]
        else:
            continue
        for reference in urls:
            if _asset_path(reference, document, export_directory) is not None:
                references += 1
    if not references:
        raise BuildError("Dashboard export does not reference any local assets.")
    return references


def _export_files(export_directory):
    export_directory = Path(export_directory)
    return {
        path.relative_to(export_directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(export_directory.rglob("*"))
        if path.is_file() and path.relative_to(export_directory).as_posix() != MANIFEST_NAME
    }


def write_build_manifest(export_directory, fingerprint):
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_sha256": fingerprint,
        "build_id": fingerprint[:20],
        "files": _export_files(export_directory),
    }
    path = Path(export_directory) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def check_build(ui_directory, destination):
    """Check source freshness and published-file integrity without Node.js."""
    destination = Path(destination)
    path = destination / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BuildError(f"Dashboard build manifest is missing or invalid: {path}") from error
    if not isinstance(manifest, dict):
        raise BuildError(f"Dashboard build manifest is missing or invalid: {path}")
    fingerprint = source_fingerprint(ui_directory)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise BuildError("Dashboard build manifest uses an unsupported schema.")
    if manifest.get("source_sha256") != fingerprint or manifest.get("build_id") != fingerprint[:20]:
        raise BuildError("Dashboard assets are stale. Run python dashboard/build_ui.py --install.")
    validate_export(destination)
    if manifest.get("files") != _export_files(destination):
        raise BuildError("Dashboard asset contents differ from the build manifest. Rebuild the frontend.")
    return manifest


def check_toolchain():
    tools = {}
    for name, expected in (("node", 24), ("npm", 11)):
        executable = shutil.which(name)
        if not executable:
            raise BuildError(f"{name} is unavailable. Install Node 24 LTS with npm 11.")
        result = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True)
        version = result.stdout.strip()
        if not re.match(rf"^v?{expected}\.", version):
            raise BuildError(f"{name} {expected}.x is required; found {version}.")
        tools[name] = executable
    return tools


def publish_export(export_directory, destination):
    """Replace generated assets, restoring the previous tree on rename failure."""
    export_directory = Path(export_directory).resolve()
    destination = Path(destination).absolute()
    validate_export(export_directory)
    if destination.is_symlink():
        raise BuildError(f"Refusing to replace a symbolic-link asset directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent.resolve() / destination.name
    if (
        export_directory == destination
        or export_directory.is_relative_to(destination)
        or destination.is_relative_to(export_directory)
    ):
        raise BuildError("Export and publication directories must be separate trees.")
    if destination.exists() and not destination.is_dir():
        raise BuildError(f"Dashboard asset destination is not a directory: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    backup = destination.parent / f".{destination.name}.backup-{uuid4().hex}"
    previous_moved = False
    published = False
    try:
        shutil.copytree(export_directory, staging, dirs_exist_ok=True)
        if destination.exists():
            os.replace(destination, backup)
            previous_moved = True
        try:
            os.replace(staging, destination)
            published = True
        except OSError as error:
            if previous_moved:
                try:
                    os.replace(backup, destination)
                    previous_moved = False
                except OSError as rollback_error:
                    raise BuildError(
                        f"Asset publication and rollback failed; previous assets remain at {backup}"
                    ) from rollback_error
            raise BuildError("Asset publication failed; previous assets were restored.") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if published and previous_moved:
        try:
            shutil.rmtree(backup)
        except OSError:
            print(f"Warning: previous asset backup could not be removed: {backup}", file=sys.stderr)


def main(argv=None):
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Install locked npm dependencies before building")
    parser.add_argument(
        "--check", action="store_true", help="Check published source freshness and asset integrity without Node"
    )
    parser.add_argument("--ui-dir", type=Path, default=repository / "dashboard" / "ui", help=argparse.SUPPRESS)
    parser.add_argument(
        "--destination",
        type=Path,
        default=repository / "nvflare" / "dashboard" / "application" / "static" / "nvflare-dashboard",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.check and args.install:
        parser.error("--check and --install cannot be combined")
    try:
        if args.check:
            check_build(args.ui_dir, args.destination)
            print("Dashboard assets match the source and build manifest.")
            return 0
        tools = check_toolchain()
        if args.install:
            subprocess.run(
                [tools["npm"], "ci", "--ignore-scripts", "--strict-peer-deps", "--no-audit", "--no-fund"],
                cwd=args.ui_dir,
                check=True,
            )
        fingerprint = source_fingerprint(args.ui_dir)
        environment = dict(os.environ, NVFLARE_DASHBOARD_BUILD_ID=fingerprint[:20], NEXT_TELEMETRY_DISABLED="1")
        subprocess.run([tools["npm"], "run", "build"], cwd=args.ui_dir, env=environment, check=True)
        if source_fingerprint(args.ui_dir) != fingerprint:
            raise BuildError("Dashboard source changed during the build. Rebuild before publishing.")
        export_directory = args.ui_dir / "out"
        references = validate_export(export_directory)
        write_build_manifest(export_directory, fingerprint)
        publish_export(export_directory, args.destination)
        print(f"Published Dashboard assets to {args.destination} ({references} local asset references checked).")
        return 0
    except (BuildError, OSError, subprocess.CalledProcessError) as error:
        print(f"Dashboard build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
