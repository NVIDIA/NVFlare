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

"""Reject known sources of unit-test timing instability; this is not a concurrency proof."""

import ast
from pathlib import Path

CLOCK_FUNCTIONS = {"time", "monotonic", "perf_counter", "sleep"}


def check_source(source, path):
    tree = ast.parse(source, filename=str(path))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def name(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{name(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ""

    def clock_isolation(scope):
        isolated = set()
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            target = name(node.func)
            if target.endswith(".isolate_time") or target == "isolate_time":
                isolated.update(name(arg) for arg in node.args[1:])
            elif target.endswith(".setattr") and len(node.args) >= 3 and name(node.args[1]) == "time":
                isolated.add(name(node.args[0]))
        return isolated

    def is_fixture(node):
        return any(
            name(decorator.func if isinstance(decorator, ast.Call) else decorator).endswith(".fixture")
            for decorator in node.decorator_list
        )

    # Only module-level autouse fixtures apply to every test; a local isolation
    # in one test must not excuse another test's shared-clock mutation.
    module_isolation = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(
            isinstance(decorator, ast.Call)
            and name(decorator.func).endswith(".fixture")
            and any(
                keyword.arg == "autouse" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in decorator.keywords
            )
            for decorator in node.decorator_list
        ):
            module_isolation.update(clock_isolation(node))

    findings = []
    for test in ast.walk(tree):
        if not isinstance(test, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not test.name.startswith("test_") and not is_fixture(test):
            continue
        # This test intentionally patches stdlib to verify the helper's real guard.
        helper_probe = str(path).replace("\\", "/").endswith("tests/unit_test/timing_utils_test.py") and (
            test.name == "test_wait_guard_is_not_frozen_by_a_shared_time_patch"
        )
        isolated = module_isolation | clock_isolation(test)
        for node in ast.walk(test):
            if not isinstance(node, ast.Call):
                continue
            target = name(node.func)
            if target == "threading.Timer":
                findings.append((node.lineno, "timer-controlled interleaving: use an observed readiness event"))
            patch_target = ""
            if target.endswith(".setattr") or target.endswith(".patch.object"):
                if len(node.args) == 2 and isinstance(node.args[0], ast.Constant):
                    patch_target = name(node.args[0])
                elif len(node.args) >= 2:
                    patch_target = f"{name(node.args[0])}.{name(node.args[1])}"
            elif target.endswith(".patch") and node.args:
                patch_target = name(node.args[0])
            parts = patch_target.split(".")
            if len(parts) >= 2 and parts[-2] == "time" and parts[-1] in CLOCK_FUNCTIONS:
                module = ".".join(parts[:-2])
                if module not in isolated and not helper_probe:
                    findings.append((node.lineno, "shared time-module patch: isolate the subject module's time first"))
    return sorted(set(findings))


def main():
    root = Path(__file__).resolve().parents[1]
    findings = []
    # This standalone unittest tree is excluded by its conftest.py and executed
    # in a separate Linux process by cvm_builder_test.py, outside pytest workers.
    standalone = root / "tests/unit_test/lighter/cc/image_builder"
    for path in sorted((root / "tests/unit_test").rglob("*.py")):
        if path.is_relative_to(standalone):
            continue
        relative = path.relative_to(root)
        findings.extend(f"{relative}:{line}: {message}" for line, message in check_source(path.read_text(), relative))
    if findings:
        print("\n".join(findings))
        return 1
    print("Unit-test timing contracts passed (timer interleavings and shared clock patches).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
