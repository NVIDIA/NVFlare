#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Assemble public role kits from an explicit inventory and shared sources.

Never executes installation scripts, reads private configuration, or downloads.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLES = ("admin", "service", "coco", "trusted_system")
GENERATED = {f"{role}/lib/validate-config.sh": "shared/validate-config.sh" for role in ROLES}
for role in ("coco", "trusted_system"):
    for name in ("lib/common.sh", "templates/kubeadm.yaml.in", "10-install-kubernetes.sh"):
        GENERATED[f"{role}/bootstrap/{name}"] = f"shared/bootstrap/{name}"


def inventory(root):
    names = (root / "PACKAGE-FILES.txt").read_text().splitlines()
    if names != sorted(set(names)):
        raise ValueError("Public allowlist must be sorted and unique")
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Unsafe public allowlist entry")
    found = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink is not publishable: {path.relative_to(root)}")
        if path.is_file():
            found.add(path.relative_to(root).as_posix())
    if found != set(names):
        raise ValueError(f"Public inventory mismatch: missing={set(names) - found}, extra={found - set(names)}")
    return names


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
            if "/shared/" not in text or len(text.splitlines()) > 8:
                raise ValueError(f"Source role entry point must be a thin shared-code wrapper: {target}")


def assemble(root, output):
    root = root.resolve()
    output = output.absolute()
    if output.is_symlink() or output.exists():
        raise ValueError("Output must be a new directory; existing output is never overwritten")
    if output.resolve().is_relative_to(root):
        raise ValueError("Output must be outside the public source package")
    names = inventory(root)
    validate_layout(root, assembled=(root / "coco/bootstrap/templates/kubeadm.yaml.in").exists())
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
    (output / "PACKAGE-FILES.txt").write_text("\n".join(sorted(set(names) | set(GENERATED))) + "\n")
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
