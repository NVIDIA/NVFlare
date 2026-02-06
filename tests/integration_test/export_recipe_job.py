# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import sys
from unittest.mock import patch


def add_paths_for_recipe(recipe_dir: str):
    """Add necessary paths for importing recipe modules."""
    recipe_abs_path = os.path.abspath(recipe_dir)

    # Add recipe directory itself
    if recipe_abs_path not in sys.path:
        sys.path.insert(0, recipe_abs_path)

    # Walk up and find src/ directory - add it to path for "from data.xxx" style imports
    parent = os.path.dirname(recipe_abs_path)
    while parent and parent != "/":
        if parent not in sys.path:
            sys.path.insert(0, parent)

        # Check for src/ directory at this level
        # This is for CIFAR10 examples where imports are "from data.xxx" and "from model"
        # and the actual files are in pt/src/data/ and pt/src/model.py
        src_path = os.path.join(parent, "src")
        if os.path.exists(src_path) and os.path.isdir(src_path):
            # Add src/ itself so "from data.xxx" and "from model" work
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

        # Check if we've reached the examples root
        if os.path.basename(parent) in ["examples", "cifar10"]:
            break
        parent = os.path.dirname(parent)


def export_recipe_from_job_py(recipe_dir: str, output_dir: str, recipe_args: list = None):
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

    # Add paths for imports
    add_paths_for_recipe(recipe_dir)

    # Change to recipe directory so relative imports work
    original_cwd = os.getcwd()
    os.chdir(recipe_abs_path)

    # Storage for captured recipe
    captured_recipe = {"recipe": None}

    def patched_execute(self, env, server_exec_params=None, client_exec_params=None):
        """Capture the recipe instead of executing."""
        captured_recipe["recipe"] = self
        # Return a mock Run object
        class MockRun:
            def get_job_id(self):
                return self.job_id

            def get_status(self):
                return "EXPORTED"

            def get_result(self):
                return output_abs_path

            def __init__(self, job_id):
                self.job_id = job_id

        return MockRun(self.name if hasattr(self, "name") else "exported_job")

    try:
        # Patch sys.argv to pass recipe_args
        original_argv = sys.argv
        sys.argv = ["job.py"] + (recipe_args or [])

        # Patch recipe execute method to capture instead of run
        with patch("nvflare.recipe.spec.Recipe.execute", patched_execute):
            # Load and run the job.py module
            spec = importlib.util.spec_from_file_location("__main__", job_py_path)
            module = importlib.util.module_from_spec(spec)
            # Set __name__ to __main__ so the if __name__ == "__main__" block runs
            module.__name__ = "__main__"

            # Execute the module (this runs the main() which creates and "executes" the recipe)
            try:
                spec.loader.exec_module(module)
            except SystemExit:
                pass  # Some scripts call sys.exit()

        # Get the captured recipe
        recipe = captured_recipe["recipe"]
        if recipe is None:
            raise RuntimeError("Failed to capture recipe - job.py may not have called recipe.execute()")

        # Export the recipe to job folder
        os.makedirs(output_abs_path, exist_ok=True)
        recipe.export(job_dir=output_abs_path)
        print(f"Exported recipe to: {output_abs_path}")

        # Copy the src/ folder if it exists (for model imports)
        src_dir = os.path.join(os.path.dirname(recipe_abs_path), "src")
        if not os.path.exists(src_dir):
            # Try going up one more level
            src_dir = os.path.join(os.path.dirname(os.path.dirname(recipe_abs_path)), "src")

        if os.path.exists(src_dir):
            # Copy CONTENTS of src/ directly into app/custom/ (not as a subdirectory)
            # This way imports like "from data.xxx" work with just /local/custom in PYTHONPATH
            job_name = recipe.job.name
            job_custom_dir = os.path.join(output_abs_path, job_name, "app", "custom")
            if os.path.exists(job_custom_dir):
                for item in os.listdir(src_dir):
                    # Skip __pycache__ and hidden files
                    if item.startswith("__") or item.startswith("."):
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

    # Parse recipe_args string into list
    recipe_args = args.recipe_args.split() if args.recipe_args else []

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
