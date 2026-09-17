#!/usr/bin/env python3
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

"""Assemble public role kits from a clean Git checkout and shared sources.

Never executes installation scripts, reads private configuration, or downloads.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLES = ("admin", "service", "coco", "trusted_system")
GENERATED = {f"{role}/lib/validate-config.sh": "shared/validate-config.sh" for role in ROLES}
for role in ("admin", "trusted_system", "coco"):
    GENERATED[f"{role}/lib/workload-security-context.py"] = "shared/workload-security-context.py"
for role in ("coco", "trusted_system"):
    GENERATED[f"{role}/lib/kata-runtime-profile.py"] = "shared/kata-runtime-profile.py"
    for name in ("lib/common.sh", "templates/kubeadm.yaml.in", "10-install-kubernetes.sh"):
        GENERATED[f"{role}/bootstrap/{name}"] = f"shared/bootstrap/{name}"


def package_files(root, assembled=False):
    if assembled:
        paths = list(root.rglob("*"))
        if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in paths):
            raise ValueError("Assembled package must contain only regular files and directories")
        return sorted(path.relative_to(root).as_posix() for path in paths if path.is_file())

    # Git's index defines the reviewed source set, not a recursive working-tree copy.
    # No fallback to scanning is safe when Git metadata is unavailable.
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z", "--", "."], cwd=root, text=True)
    names = []
    for entry in entries.split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, _, stage = metadata.split()
        if mode not in ("100644", "100755") or stage != "0":
            raise ValueError(f"Source must contain regular, non-conflicted tracked files: {name}")
        names.append(name)
    if not names:
        raise ValueError("No tracked package sources found; use the NVFlare Git checkout")
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Unsafe tracked source path")
        if any((root / part).is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"Symlink is not publishable: {name}")
        if not (root / path).is_file():
            raise ValueError(f"Tracked source file is missing: {name}")
    return sorted(names)


def validate_layout(root, assembled=False):
    for target, source in GENERATED.items():
        if assembled:
            if (root / target).read_bytes() != (root / source).read_bytes():
                raise ValueError(f"Generated role file differs from shared source: {target}")
        elif "/templates/" in target:
            if (root / target).exists():
                raise ValueError(f"Keep only the shared template in source: {target}")
        else:
            text = (root / target).read_text()
            code_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if "/shared/" not in text or len(code_lines) > 8:
                raise ValueError(f"Source role entry point must be a thin shared-code wrapper: {target}")


def assemble(root, output):
    root = root.resolve()
    output = output.absolute()
    if output.is_symlink() or output.exists():
        raise ValueError("Output must be a new directory; existing output is never overwritten")
    if output.resolve().is_relative_to(root):
        raise ValueError("Output must be outside the public source package")
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "."], cwd=root, text=True
    )
    if changes:
        raise ValueError("Commit reviewed package changes before assembly; tracked sources must be clean")
    names = package_files(root)
    validate_layout(root)
    subprocess.run([sys.executable, str(root / "validate-package.py")], check=True)
    output.mkdir(mode=0o755)
    for name in names:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        target.chmod(0o755 if name.endswith(".sh") else 0o644)
    for target, source in GENERATED.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source, destination)
        destination.chmod(0o755 if target.endswith(".sh") else 0o644)
    validate_layout(output, assembled=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory outside this source package")
    args = parser.parse_args()
    output = assemble(ROOT, args.output)
    print(f"Assembled public role kits: {output}")
    print("Validate with: python3 OUTPUT/validate-package.py --assembled")
    print("Transfer only the intended role directory; supply deployment inputs separately.")


if __name__ == "__main__":
    main()
