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
    files["plugin-card.md"] = read_file(root, "plugin-card.md")
    files["LICENSE"] = read_file(root, "LICENSE")
    canonical = json.loads(files["plugin.json"][0])
    if canonical["name"] != "nvidia-nvflare" or not re.fullmatch(r"\d+\.\d+\.\d+", canonical["version"]):
        raise ValueError("Expected nvidia-nvflare with a three-part release version")
    for path in MANIFESTS[1:]:
        manifest = json.loads(files[path][0])
        if any(manifest.get(field) != canonical.get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"Shared plugin metadata differs: {path}")
    count = verify_resources(files)
    packaging_paths = [*MANIFESTS, "PLUGIN.md", "plugin-card.md", "LICENSE", "dev_tools/agent/package_plugin.py"]
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
        "plugin_signed": False,
        "files": {path: hashlib.sha256(data).hexdigest() for path, (data, _) in sorted(files.items())},
    }
    if not metadata["skills_match_source_commit"]:
        raise ValueError("Commit the evaluated, signed skill changes before building a release candidate")
    return files, metadata


def write_bundle(output: Path, files: dict[str, tuple[bytes, int]], metadata: dict) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    package = output / metadata["plugin"]
    archive = output / f"{metadata['plugin']}-{metadata['version']}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative, (data, mode) in sorted(files.items()):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(mode)
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
    parser.add_argument("--output", type=Path, required=True, help="new directory for the bundle and build metadata")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        files, metadata = collect_files(root)
        archive = write_bundle(args.output, files, metadata)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Plugin packaging failed: {exc}\n")
    print(f"Built {archive}; {len(files)} files, {metadata['matched_resource_hashes']} matching resource hashes")
    print("Unsigned plugin candidate; evaluation, signature trust and publication require separate verification.")


if __name__ == "__main__":
    main()
