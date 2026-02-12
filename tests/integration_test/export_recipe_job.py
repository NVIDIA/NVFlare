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

"""
Export a recipe-based job.py to a traditional job folder for integration testing.

This script imports the actual job.py from examples and patches the recipe's
execute() method to export instead, ensuring consistency with the examples.

Usage:
    python export_recipe_job.py --recipe_dir <path_to_recipe_dir> --output_dir <output_job_dir> [options]

Example:
    python export_recipe_job.py \
        --recipe_dir ../../examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedavg \
        --output_dir ./exported_jobs/cifar10_fedavg_test \
        --recipe_args "--n_clients 2 --num_rounds 2"
"""

import argparse
import importlib.util
import os
import shlex
import shutil
import sys
from typing import Optional
from unittest.mock import patch


def ensure_nvflare_on_path():
    """Ensure the NVFlare package root is on sys.path so the recipe patch target can be imported."""
    # Infer repo root from this script's location: .../tests/integration_test/export_recipe_job.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    nvflare_dir = os.path.join(repo_root, "nvflare")
    if os.path.isdir(nvflare_dir) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def find_src_parent(start_path: str, max_levels: int = 5) -> Optional[str]:
    """Find nearest ancestor that contains a src/ directory.

    Returns the parent directory that owns ``src/``; returns None if not found.
    """
    parent = os.path.dirname(start_path)
    for _ in range(max_levels):
        if not parent or parent == os.path.dirname(parent):
            break
        src_path = os.path.join(parent, "src")
        if os.path.isdir(src_path):
            return parent
        parent = os.path.dirname(parent)
    return None


def add_paths_for_recipe(recipe_dir: str):
    """Add necessary paths for importing recipe modules.

    Adds the recipe directory itself and, if found, the nearest ancestor ``src/``
    directory (used by CIFAR-10 examples for ``from data.xxx`` / ``from model`` imports).
    Only directories that contain a ``src/`` sub-folder are added — intermediate
    ancestors are *not* added so we don't pollute ``sys.path``.
    """
    recipe_abs_path = os.path.abspath(recipe_dir)

    # Add recipe directory itself
    if recipe_abs_path not in sys.path:
        sys.path.insert(0, recipe_abs_path)

    src_parent = find_src_parent(recipe_abs_path, max_levels=5)
    if src_parent:
        src_path = os.path.join(src_parent, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if src_parent not in sys.path:
            sys.path.insert(0, src_parent)


def export_recipe_from_job_py(recipe_dir: str, output_dir: str, recipe_args: Optional[list[str]] = None):
    """
    Import and run job.py, but patch recipe.execute() to export instead.

    This ensures we test the exact same configuration as the example.
    """
    recipe_abs_path = os.path.abspath(recipe_dir)
    job_py_path = os.path.join(recipe_abs_path, "job.py")

    # Convert output_dir to absolute path BEFORE changing directories
    # Otherwise, relative paths like "./data/jobs" would be resolved relative to the recipe dir
    output_abs_path = os.path.abspath(output_dir)

    if not os.path.exists(job_py_path):
        raise FileNotFoundError(f"job.py not found in {recipe_dir}")

    # Capture state to restore on exit (including on exception)
    original_path = list(sys.path)
    original_cwd = os.getcwd()
    original_argv = sys.argv

    # Ensure nvflare is importable so the patch target can be resolved
    ensure_nvflare_on_path()

    # Add paths for imports
    add_paths_for_recipe(recipe_dir)

    # Change to recipe directory so relative imports work
    os.chdir(recipe_abs_path)

    # Storage for captured recipe
    captured_recipe = {"recipe": None}

    def patched_execute(self, env, server_exec_params=None, client_exec_params=None):
        """Capture the recipe instead of executing.

        Applies server/client exec params exactly as the real Recipe.execute does
        so the exported job matches what a real run would produce.
        """
        if server_exec_params:
            self.job.to_server(server_exec_params)
        if client_exec_params:
            self.job.to_clients(client_exec_params)
        # Match real Recipe.execute behavior so export matches runtime configuration.
        self.process_env(env)

        captured_recipe["recipe"] = self

        class MockRun:
            """Mimics the real Run interface so post-execute code in job.py doesn't break."""

            def __init__(self, job_id):
                self.job_id = job_id

            def get_job_id(self):
                return self.job_id

            def get_status(self):
                return "EXPORTED"

            def get_result(self, timeout: float = 0.0):
                return output_abs_path

            def abort(self):
                pass

        return MockRun(self.name if hasattr(self, "name") else "exported_job")

    try:
        sys.argv = ["job.py"] + (recipe_args or [])

        with patch("nvflare.recipe.spec.Recipe.execute", patched_execute):
            spec = importlib.util.spec_from_file_location("__main__", job_py_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load recipe module from: {job_py_path}")
            module = importlib.util.module_from_spec(spec)
            module.__name__ = "__main__"
            try:
                spec.loader.exec_module(module)
            except SystemExit as e:
                # Keep successful exits quiet but preserve true failures.
                if e.code not in (None, 0):
                    raise

        recipe = captured_recipe["recipe"]
        if recipe is None:
            raise RuntimeError("Failed to capture recipe - job.py did not call recipe.execute()")

        # Export the recipe to job folder
        os.makedirs(output_abs_path, exist_ok=True)
        recipe.export(job_dir=output_abs_path)
        print(f"Exported recipe to: {output_abs_path}")

        # Copy the src/ folder if it exists (for model imports).
        src_parent = find_src_parent(recipe_abs_path, max_levels=5)
        src_dir = os.path.join(src_parent, "src") if src_parent else None

        if src_dir is not None:
            # Copy CONTENTS of src/ directly into app/custom/ (not as a subdirectory)
            # This way imports like "from data.xxx" work with just /local/custom in PYTHONPATH
            job_name = recipe.job.name
            job_custom_dir = os.path.join(output_abs_path, job_name, "app", "custom")
            if os.path.exists(job_custom_dir):
                for item in os.listdir(src_dir):
                    # Skip __pycache__ and hidden (dot) files only; keep __init__.py etc.
                    if item == "__pycache__" or item.startswith("."):
                        continue
                    src_item = os.path.join(src_dir, item)
                    dest_item = os.path.join(job_custom_dir, item)
                    if not os.path.exists(dest_item):
                        if os.path.isdir(src_item):
                            shutil.copytree(src_item, dest_item)
                        else:
                            shutil.copy2(src_item, dest_item)
                        print(f"Copied {item} to: {job_custom_dir}")

    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        sys.path = original_path


def main():
    parser = argparse.ArgumentParser(description="Export a recipe to a job folder for testing")
    parser.add_argument("--recipe_dir", required=True, type=str, help="Path to recipe directory containing job.py")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for exported job")
    parser.add_argument(
        "--recipe_args",
        type=str,
        default="",
        help="Arguments to pass to job.py (e.g., '--n_clients 2 --num_rounds 2')",
    )

    args = parser.parse_args()

    # Parse recipe_args string into list (shell-style, handles quotes and escapes)
    recipe_args = shlex.split(args.recipe_args) if args.recipe_args else []

    # Extract job name from recipe_args if provided (--name <job_name>)
    # We need this to clean only the specific job subfolder, not the entire output_dir
    job_name = None
    for i, arg in enumerate(recipe_args):
        if arg == "--name" and i + 1 < len(recipe_args):
            job_name = recipe_args[i + 1]
            break

    # Clean only the specific job subfolder if it exists
    # IMPORTANT: Do NOT delete the entire output_dir as it may contain other jobs
    if job_name:
        job_subfolder = os.path.join(args.output_dir, job_name)
        if os.path.exists(job_subfolder):
            shutil.rmtree(job_subfolder)
            print(f"Cleaned existing job folder: {job_subfolder}")

    export_recipe_from_job_py(args.recipe_dir, args.output_dir, recipe_args)


if __name__ == "__main__":
    main()
