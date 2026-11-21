# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Build a self-contained plugin without changing the existing signed skills.

Resource hashes are checked for integrity only. This is not cryptographic
signature verification, plugin evaluation, or plugin signing.
"""

import argparse
import base64
import hashlib
import json
import re
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

SKILLS = (
    "nvflare-autofl",
    "nvflare-autofl-report",
    "nvflare-convert-huggingface",
    "nvflare-convert-lightning",
    "nvflare-convert-pytorch",
    "nvflare-diagnose-job",
    "nvflare-fed-stats",
    "nvflare-orient",
    "nvflare-shared",
)
MANIFESTS = ("plugin.json", ".claude-plugin/plugin.json", ".codex-plugin/plugin.json", ".cursor-plugin/plugin.json")
IDENTITY_FIELDS = ("name", "version", "description", "author", "homepage", "repository", "license", "keywords")
BUNDLE_PATH = Path("plugins/nvidia-nvflare")
LIGHTNING_EVAL_CHECKPOINT = "evals/files/lightning-eval-only/litnet-eval.pt"
DICOM_EVAL_DIR = "evals/files/fedstats-image-dicom-dependency"
DICOM_EVAL_FILES = tuple(f"{DICOM_EVAL_DIR}/site-{site}/synthetic.dcm" for site in range(1, 4))
# CLI exports and verification-service outputs are retained, never generated here.
VERIFICATION_FILES = {
    "eval-result.json",
    "handoff-manifest.json",
    "overall-score.json",
    "BENCHMARK.md",
    "plugin-card.json",
    "plugin-card.md",
    "plugin.sig.oms",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def read_file(root: Path, relative: str) -> tuple[bytes, int]:
    path = root
    for part in PurePosixPath(relative).parts:
        if part in ("..", "/"):
            raise ValueError(f"Path escapes the package: {relative}")
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlinks are not packaged: {relative}")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"Not a regular file: {relative}")
    return path.read_bytes(), 0o755 if mode & 0o111 else 0o644


def verify_resources(files: dict[str, tuple[bytes, int]]) -> int:
    """Compare every bundled skill file with its retained manifest resource list."""
    count = 0
    for skill in SKILLS:
        prefix = f"skills/{skill}/"
        signature = prefix + "skill.oms.sig"
        envelope = json.loads(files[signature][0])["dsseEnvelope"]
        statement = json.loads(base64.b64decode(envelope["payload"], validate=True))
        recorded = set()
        for resource in statement["predicate"]["resources"]:
            name = resource["name"]
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
                raise ValueError(f"Invalid resource path in {signature}: {name}")
            key = prefix + name
            if key in recorded or key == signature:
                raise ValueError(f"Duplicate or self-referencing resource: {key}")
            recorded.add(key)
            if resource["algorithm"] != "sha256":
                raise ValueError(f"Unsupported resource hash algorithm: {key}")
            if key not in files or hashlib.sha256(files[key][0]).hexdigest() != resource["digest"]:
                raise ValueError(f"Signed resource hash mismatch or missing file: {key}")
        actual = {key for key in files if key.startswith(prefix) and key != signature}
        if actual != recorded:
            raise ValueError(f"Skill files absent from the resource list: {sorted(actual - recorded)}")
        if prefix + "SKILL.md" not in recorded:
            raise ValueError(f"Missing SKILL.md resource: {skill}")
        count += len(recorded)
    return count


def collect_files(root: Path) -> tuple[dict[str, tuple[bytes, int]], dict]:
    # Only tracked skill files enter the bundle; local logs, keys and caches do not.
    tracked = git(root, "ls-files", "-z", "--", "skills").strip("\0").split("\0")
    inventory = {PurePosixPath(p).parts[1] for p in tracked if p.endswith("/SKILL.md")}
    if inventory != set(SKILLS):
        raise ValueError(f"Expected the nine catalog skill directories; found: {sorted(inventory)}")
    files = {p: read_file(root, p) for p in tracked}
    for path in MANIFESTS:
        files[path] = read_file(root, path)
    files["README.md"] = read_file(root, "PLUGIN.md")
    files["evals.json"] = read_file(root, "evals.json")
    files["evals/config.json"] = read_file(root, "evals/config.json")
    for path in (LIGHTNING_EVAL_CHECKPOINT, *DICOM_EVAL_FILES):
        files[path] = read_file(root, path)
    files["LICENSE"] = read_file(root, "LICENSE")
    files["LICENSE-CC-BY-4.0"] = read_file(root, "LICENSE-CC-BY-4.0")
    files["SKILLS-LICENSE.md"] = read_file(root, "SKILLS-LICENSE.md")
    files["CONTRIBUTING.md"] = read_file(root, "CONTRIBUTING.md")
    canonical = json.loads(files["plugin.json"][0])
    if canonical["name"] != "nvidia-nvflare" or not re.fullmatch(r"\d+\.\d+\.\d+", canonical["version"]):
        raise ValueError("Expected nvidia-nvflare with a three-part release version")
    for path in MANIFESTS[1:]:
        manifest = json.loads(files[path][0])
        if any(manifest.get(field) != canonical.get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"Shared plugin metadata differs: {path}")
    count = verify_resources(files)
    packaging_paths = [
        *MANIFESTS,
        "PLUGIN.md",
        "evals.json",
        "evals/config.json",
        LIGHTNING_EVAL_CHECKPOINT,
        *DICOM_EVAL_FILES,
        "LICENSE",
        "LICENSE-CC-BY-4.0",
        "SKILLS-LICENSE.md",
        "CONTRIBUTING.md",
        "dev_tools/agent/package_plugin.py",
    ]
    committed_packaging = set(git(root, "ls-tree", "-r", "--name-only", "HEAD", "--", *packaging_paths).splitlines())
    metadata = {
        "plugin": canonical["name"],
        "version": canonical["version"],
        "source_commit": git(root, "rev-parse", "HEAD").strip(),
        "source_skills_tree": git(root, "rev-parse", "HEAD:skills").strip(),
        "skills_match_source_commit": not bool(git(root, "diff", "HEAD", "--", "skills")),
        "packaging_matches_source_commit": committed_packaging == set(packaging_paths)
        and not bool(git(root, "diff", "HEAD", "--", *packaging_paths)),
        "skill_directories": list(SKILLS),
        "matched_resource_hashes": count,
        "cryptographic_signatures_verified": False,
        "plugin_signature_verified": False,
        "files": {path: hashlib.sha256(data).hexdigest() for path, (data, _) in sorted(files.items())},
    }
    if not metadata["skills_match_source_commit"]:
        raise ValueError("Commit the evaluated, signed skill changes before building a release candidate")
    return files, metadata


def bundle_path(root: Path) -> Path:
    path = root
    for part in BUNDLE_PATH.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"Symlinks are not packaged: {path}")
    return path


def read_bundle(package: Path) -> dict[str, tuple[bytes, int]]:
    if package.is_symlink() or not package.is_dir():
        raise ValueError(f"Expected a real plugin directory: {package}")
    files = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symlinks are not packaged: {relative}")
        if not path.is_dir():
            files[relative] = read_file(package, relative)
    return files


def check_bundle(package: Path, expected: dict[str, tuple[bytes, int]]) -> dict[str, tuple[bytes, int]]:
    """Check source parity; verification of retained evidence belongs to NVCARPS."""
    actual = read_bundle(package)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys() - VERIFICATION_FILES)
    changed = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
    if missing or extra or changed:
        raise ValueError(
            f"Plugin bundle drift: missing={missing}, extra={extra}, changed={changed}. Rebuild with --sync."
        )
    return actual


def write_package(package: Path, files: dict[str, tuple[bytes, int]]) -> None:
    for relative, (data, mode) in sorted(files.items()):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)


def sync_bundle(package: Path, files: dict[str, tuple[bytes, int]]) -> None:
    """Refresh generated inputs without silently discarding or retaining stale evidence."""
    actual = read_bundle(package) if package.exists() or package.is_symlink() else {}
    extra = sorted(actual.keys() - files.keys() - VERIFICATION_FILES)
    if extra:
        raise ValueError(f"Review and remove obsolete or unexpected bundle files before rebuilding: {extra}")
    if actual.keys() & VERIFICATION_FILES and any(actual.get(name) != value for name, value in files.items()):
        raise ValueError(
            "Retire the prior evaluation/signing outputs before rebuilding; fresh evaluation and signing required"
        )
    write_package(package, files)
    check_bundle(package, files)


def write_bundle(output: Path, files: dict[str, tuple[bytes, int]], metadata: dict) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    package = output / metadata["plugin"]
    archive = output / f"{metadata['plugin']}-{metadata['version']}.zip"
    metadata["files"] = {path: hashlib.sha256(data).hexdigest() for path, (data, _) in sorted(files.items())}
    metadata["plugin_signature_present"] = "plugin.sig.oms" in files
    write_package(package, files)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative, (data, mode) in sorted(files.items()):
            info = zipfile.ZipInfo(f"{package.name}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
    metadata["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "build-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "SHA256SUMS").write_text(f"{metadata['archive_sha256']}  {archive.name}\n")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--output", type=Path, help="archive the checked bundle into a new directory with build metadata"
    )
    action.add_argument(
        "--sync", action="store_true", help="regenerate plugins/nvidia-nvflare from the canonical inputs"
    )
    action.add_argument("--check", action="store_true", help="fail if plugins/nvidia-nvflare differs from its sources")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        files, metadata = collect_files(root)
        package = bundle_path(root)
        if args.sync:
            sync_bundle(package, files)
            print(f"Updated {BUNDLE_PATH}; commit the bundle before evaluating this exact directory.")
            return
        files = check_bundle(package, files)
        if args.check:
            print(f"Bundle matches source: {len(files)} files, {metadata['matched_resource_hashes']} resource hashes")
            return
        archive = write_bundle(args.output, files, metadata)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Plugin packaging failed: {exc}\n")
    print(f"Built {archive}; {len(files)} files, {metadata['matched_resource_hashes']} matching resource hashes")
    print("Packaging checked; evaluation, signature trust and publication require separate verification.")


if __name__ == "__main__":
    main()
