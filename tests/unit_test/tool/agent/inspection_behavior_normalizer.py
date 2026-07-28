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

"""Normalize inspector output for the behavior contract.

This module intentionally imports NVFLARE only inside ``inspect_corpus``. Use
``--source-root`` to generate the immutable golden from a detached reference
tree. The command removes NVFLARE editable-install finders and verifies the
resolved package before inspection, so ``PYTHONPATH`` cannot silently select
the working tree:

    python tests/unit_test/tool/agent/inspection_behavior_normalizer.py \
        --source-root /path/to/reference \
        --output tests/unit_test/tool/agent/fixtures/inspection_behavior/expected.json
"""

import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

CORPUS_ROOT = Path(__file__).parent / "fixtures" / "inspection_behavior"
CASES_FILE = CORPUS_ROOT / "cases.json"
EXPECTED_FILE = CORPUS_ROOT / "expected.json"
ENVIRONMENT_NORMALIZATIONS = (
    "path",
    "nvflare_version",
    "installed_skills",
)


def normalizer_sha256() -> str:
    """Return the digest that pins this normalizer and its exclusion rules."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_case_definitions(corpus_root: Path = CORPUS_ROOT) -> list[dict]:
    """Load ordered corpus metadata."""
    payload = json.loads((corpus_root / CASES_FILE.name).read_text(encoding="utf-8"))
    return payload["cases"]


def normalize_inspection(result: dict) -> dict:
    """Normalize only fields whose values depend on the host environment."""
    normalized = copy.deepcopy(result)
    normalized["path"] = "<CASE_ROOT>"
    normalized["nvflare_version"] = "<NVFLARE_VERSION>"

    normalized["installed_skills"] = []
    return normalized


def _prepare_source_root(source_root: Path) -> Path:
    """Make one explicit source tree authoritative for subsequent NVFLARE imports."""
    source_root = source_root.resolve()
    package_root = source_root / "nvflare"
    if not (package_root / "__init__.py").is_file():
        raise ValueError(f"source root does not contain nvflare/__init__.py: {source_root}")

    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not getattr(finder, "__module__", "").startswith("__editable___nvflare")
    ]
    for module_name in tuple(sys.modules):
        if module_name == "nvflare" or module_name.startswith("nvflare."):
            del sys.modules[module_name]
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    return source_root


def _assert_nvflare_source(source_root: Path) -> None:
    import nvflare

    package_file = Path(nvflare.__file__).resolve()
    expected_package_root = (source_root / "nvflare").resolve()
    if not package_file.is_relative_to(expected_package_root):
        raise RuntimeError(
            f"resolved NVFLARE package is outside --source-root: {package_file} (expected under {expected_package_root})"
        )


def inspect_corpus(corpus_root: Path = CORPUS_ROOT, source_root: Optional[Path] = None) -> dict:
    """Inspect and normalize every case using the selected NVFLARE source."""
    if source_root is not None:
        source_root = _prepare_source_root(source_root)
        _assert_nvflare_source(source_root)

    from nvflare.tool.agent.inspector import inspect_path

    cases = {}
    for case in load_case_definitions(corpus_root):
        case_id = case["id"]
        cases[case_id] = normalize_inspection(inspect_path(corpus_root / case["path"]))
    return {
        "contract_version": "1",
        "normalizer_sha256": normalizer_sha256(),
        "environment_normalizations": list(ENVIRONMENT_NORMALIZATIONS),
        "cases": cases,
    }


def canonical_json(payload: dict) -> str:
    """Serialize a contract payload deterministically, one line per case.

    The contract compares parsed objects, so indentation carries no meaning for
    the gate. Emitting each case on its own line keeps the golden small while
    still naming the case that moved in a failing diff.
    """
    compact = {"sort_keys": True, "ensure_ascii": True, "separators": (",", ":")}
    header = {key: value for key, value in payload.items() if key != "cases"}
    lines = ["{"]
    for key in sorted(header):
        lines.append(f"  {json.dumps(key)}:{json.dumps(header[key], **compact)},")
    lines.append('  "cases":{')
    case_ids = list(payload["cases"])
    for index, case_id in enumerate(case_ids):
        trailer = "," if index < len(case_ids) - 1 else ""
        lines.append(f"    {json.dumps(case_id)}:{json.dumps(payload['cases'][case_id], **compact)}{trailer}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate normalized agent-inspector characterization output.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="detached NVFLARE source root whose inspector produces the output",
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_ROOT, help="inspection behavior corpus root")
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    rendered = canonical_json(inspect_corpus(args.corpus, source_root=args.source_root))
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
