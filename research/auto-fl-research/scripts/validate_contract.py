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

"""Static validator for the Auto-FL client contract."""

import argparse
import ast
import py_compile
from pathlib import Path

REQUIRED_CHECKS = [
    "flare.init()",
    "flare.receive()",
    "flare.send(...)",
    "model.load_state_dict(..., strict=True)",
    "compute_model_diff(...)",
    "FLModel(..., params_type=ParamsType.DIFF, ...)",
    'meta includes "NUM_STEPS_CURRENT_ROUND"',
    "flare.is_evaluate() branch sends ParamsType.DIFF",
]


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def is_name(node, dotted_name):
    return call_name(node) == dotted_name


def is_true_constant(node):
    return isinstance(node, ast.Constant) and node.value is True


def is_params_type_diff(node):
    return isinstance(node, ast.Attribute) and node.attr == "DIFF" and is_name(node.value, "ParamsType")


def call_has_keyword(call, keyword_name, predicate):
    for keyword in call.keywords:
        if keyword.arg == keyword_name and predicate(keyword.value):
            return True
    return False


def contains_call(node, dotted_name):
    return any(isinstance(child, ast.Call) and is_name(child.func, dotted_name) for child in ast.walk(node))


def contains_num_steps_key(node):
    return any(isinstance(child, ast.Constant) and child.value == "NUM_STEPS_CURRENT_ROUND" for child in ast.walk(node))


def contains_diff_flmodel(node):
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if call_name(child.func) not in {"flare.FLModel", "FLModel"}:
            continue
        if call_has_keyword(child, "params_type", is_params_type_diff):
            return True
    return False


class ClientContractVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_flare_init = False
        self.has_flare_receive = False
        self.has_flare_send = False
        self.has_strict_load = False
        self.has_compute_model_diff = False
        self.has_diff_flmodel = False
        self.has_num_steps = False
        self.has_eval_diff_send = False

    def visit_Call(self, node):
        name = call_name(node.func)
        if name == "flare.init":
            self.has_flare_init = True
        elif name == "flare.receive":
            self.has_flare_receive = True
        elif name == "flare.send":
            self.has_flare_send = True
        elif name and name.endswith(".load_state_dict"):
            self.has_strict_load = self.has_strict_load or call_has_keyword(node, "strict", is_true_constant)
        elif name == "compute_model_diff":
            self.has_compute_model_diff = True
        elif name in {"flare.FLModel", "FLModel"}:
            self.has_diff_flmodel = self.has_diff_flmodel or call_has_keyword(node, "params_type", is_params_type_diff)

        if contains_num_steps_key(node):
            self.has_num_steps = True
        self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == "NUM_STEPS_CURRENT_ROUND":
            self.has_num_steps = True

    def visit_If(self, node):
        if contains_call(node.test, "flare.is_evaluate"):
            self.has_eval_diff_send = self.has_eval_diff_send or contains_diff_flmodel(
                ast.Module(body=node.body, type_ignores=[])
            )
        self.generic_visit(node)

    def missing_checks(self):
        checks = [
            (self.has_flare_init, REQUIRED_CHECKS[0]),
            (self.has_flare_receive, REQUIRED_CHECKS[1]),
            (self.has_flare_send, REQUIRED_CHECKS[2]),
            (self.has_strict_load, REQUIRED_CHECKS[3]),
            (self.has_compute_model_diff, REQUIRED_CHECKS[4]),
            (self.has_diff_flmodel, REQUIRED_CHECKS[5]),
            (self.has_num_steps, REQUIRED_CHECKS[6]),
            (self.has_eval_diff_send, REQUIRED_CHECKS[7]),
        ]
        return [label for ok, label in checks if not ok]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="client.py")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    visitor = ClientContractVisitor()
    visitor.visit(tree)
    missing = visitor.missing_checks()
    if missing:
        print("ERROR: client contract validation failed. Missing:")
        for item in missing:
            print(f"  - {item}")
        return 2

    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"ERROR: syntax validation failed: {exc}")
        return 3

    print(f"OK: static client contract checks passed for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
