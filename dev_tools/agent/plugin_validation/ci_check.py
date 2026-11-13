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

"""Check the plugin archive and its contained resources without running agents.

Run from the source root with python -m dev_tools.agent.plugin_validation.ci_check.
These checks do not establish security-scan completeness or publication approval.
"""

import argparse
import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

from jsonschema import Draft202012Validator

from ci.check_relative_doc_links import check_relative_doc_links
from dev_tools.agent import package_plugin

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_URL = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
SCHEMA_SHA256 = "0a4aad95ce337878ad38802ebf0daa3fde76abe3f65400c86bcbb1ec0b3ab883"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_contained_package(package, schema):
    """Check discovery, signed resources and links inside the extracted boundary."""
    manifests = {path: json.loads((package / path).read_text()) for path in package_plugin.MANIFESTS}
    canonical = manifests["plugin.json"]
    Draft202012Validator(schema).validate(canonical)
    for path, manifest in manifests.items():
        require(
            all(manifest.get(field) == canonical.get(field) for field in package_plugin.IDENTITY_FIELDS),
            f"Plugin identity differs in {path}",
        )
        require(not manifest.get("mcpServers") and not manifest.get("hooks"), f"Unexpected MCP or hooks in {path}")
    codex = manifests[".codex-plugin/plugin.json"]
    require(codex.get("requires_local_executor") is True, "Codex local executor requirement is missing")
    require(codex.get("skills") == "./skills/", "Codex skill discovery path differs")
    require(
        manifests[".cursor-plugin/plugin.json"].get("skills", "./skills/") == "./skills/", "Cursor skill path differs"
    )
    require(
        manifests[".claude-plugin/plugin.json"].get("skills", "./skills/") in ("./skills/", ["./skills/"]),
        "Claude skill discovery path differs",
    )
    paths = sorted(package.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "Extracted plugin contains a symlink")
    inventory = {path.parent.name for path in (package / "skills").glob("*/SKILL.md")}
    require(inventory == set(package_plugin.SKILLS), "Incomplete or unexpected component inventory")
    files = {
        path.relative_to(package).as_posix(): package_plugin.read_file(package, path.relative_to(package).as_posix())
        for path in paths
        if path.is_file()
    }
    count = package_plugin.verify_resources(files)
    require(bool(files.get("plugin-card.md", (b"", 0))[0].strip()), "Plugin card is missing or empty")
    docs = [path for path in paths if path.suffix in (".md", ".html") and path.is_file()]
    problems = check_relative_doc_links(docs, package)
    require(not problems, "Broken package links: " + "; ".join(f"{p.file_path}:{p.line}: {p.target}" for p in problems))
    return {"component_count": len(inventory), "matched_resource_hashes": count, "documents_checked": len(docs)}


def run(output, schema_path):
    schema_bytes = schema_path.read_bytes()
    require(hashlib.sha256(schema_bytes).hexdigest() == SCHEMA_SHA256, "Agent Plugins schema checksum differs")
    schema = json.loads(schema_bytes)
    require(schema.get("$id") == SCHEMA_URL, "Unexpected Agent Plugins schema identity")
    Draft202012Validator.check_schema(schema)
    files, metadata = package_plugin.collect_files(ROOT)
    archive = package_plugin.write_bundle(output / "candidate", files, metadata)
    second_files, second_metadata = package_plugin.collect_files(ROOT)
    repeat = package_plugin.write_bundle(output / "repeat", second_files, second_metadata)
    require(archive.read_bytes() == repeat.read_bytes(), "Plugin archive is not reproducible")
    with zipfile.ZipFile(archive) as zipped:
        expected = {f"{metadata['plugin']}/{name}" for name in files}
        require(
            set(zipped.namelist()) == expected and len(zipped.namelist()) == len(expected), "Archive inventory differs"
        )
        for entry in zipped.infolist():
            name = entry.filename.removeprefix(metadata["plugin"] + "/")
            require(zipped.read(entry) == files[name][0], f"Archive bytes differ: {name}")
            mode = entry.external_attr >> 16
            require(stat.S_ISREG(mode) and stat.S_IMODE(mode) == files[name][1], f"Archive permissions differ: {name}")
        # Only extract this builder's archive after checking every entry name.
        zipped.extractall(output / "extracted")
    package = output / "extracted" / metadata["plugin"]
    details = validate_contained_package(package, schema)
    command = [sys.executable, "-m", "dev_tools.agent.skills.checks", "--skills-root", str(package / "skills")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    (output / "skill-lint.log").write_text(result.stdout + result.stderr)
    require(result.returncode == 0, "Packaged skill lint failed; see skill-lint.log")
    return {
        "passed": True,
        **details,
        "archive_sha256": metadata["archive_sha256"],
        "source_commit": metadata["source_commit"],
        "source_skills_tree": metadata["source_skills_tree"],
        "packaging_matches_source_commit": metadata["packaging_matches_source_commit"],
        "schema_sha256": SCHEMA_SHA256,
        "reproducible_archive": True,
        "archive_bytes_and_modes_verified": True,
        "skill_lint": "passed",
        "publication_eligibility": "not assessed",
        "cryptographic_signature_trust": "not assessed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory outside the checkout")
    parser.add_argument("--schema", type=Path, required=True, help="downloaded Agent Plugins 1.0.0 schema")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        parser.error("Output must be outside the source checkout")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.exit(1, f"Cannot create check output: {exc}\n")
    try:
        status = run(output, args.schema)
    except Exception as exc:
        status = {"passed": False, "error": str(exc), "publication_eligibility": "not assessed"}
    (output / "results.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
