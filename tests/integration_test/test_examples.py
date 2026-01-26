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
Auto-discovers and tests all examples with job.py files.
Each test runs in a fresh virtual environment.

Usage:
    pytest test_examples.py -v -k "hello-numpy"
    pytest test_examples.py -v -n 4  # parallel execution
"""

import fcntl
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest

EXAMPLES_ROOT = Path(__file__).parent.parent.parent / "examples"
HELLO_WORLD_ROOT = EXAMPLES_ROOT / "hello-world"
ADVANCED_ROOT = EXAMPLES_ROOT / "advanced"
NVFLARE_ROOT = Path(__file__).parent.parent.parent

# Directory to store the built wheel (survives test runs for caching)
WHEEL_DIR = Path(tempfile.gettempdir()) / "nvflare_test_wheel"


def _get_wheel_path():
    """Build nvflare wheel once, return path. Uses file lock for parallel safety."""
    WHEEL_DIR.mkdir(exist_ok=True)
    lock_file = WHEEL_DIR / ".lock"
    path_file = WHEEL_DIR / ".wheel_path"
    
    with open(lock_file, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Check if wheel already built
            if path_file.exists():
                wheel_path = path_file.read_text().strip()
                if Path(wheel_path).exists():
                    return wheel_path
            
            # Build wheel
            for old in WHEEL_DIR.glob("*.whl"):
                old.unlink()
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(WHEEL_DIR), str(NVFLARE_ROOT)],
                check=True, timeout=300
            )
            wheel_path = str(next(WHEEL_DIR.glob("*.whl")))
            path_file.write_text(wheel_path)
            return wheel_path
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# Build wheel at module load (will be cached for parallel workers)
WHEEL_PATH = _get_wheel_path()

# Examples that require special handling
EXCLUDED_EXAMPLES = [
    "hello-flower",  # Requires --job_name and --content_dir arguments
    # Heavy external dependencies
    "bionemo",
    "llm_hf",
    "codon-fm",
    "wandb",
    "mlflow",
    "multi-gpu",
    "tensor-stream",  # Downloads large GPT-2 model
    # No argparse / hardcoded values - can't parametrize
    "experiment-tracking/tensorboard",
    "fedavg-with-early-stopping",
    # Complex setup / missing data prep
    "kaplan-meier-he",
]


def discover_examples():
    """Find all directories with job.py in hello-world and advanced."""
    examples = []
    for root in [HELLO_WORLD_ROOT, ADVANCED_ROOT]:
        for job_file in root.rglob("job.py"):
            if "__pycache__" in str(job_file):
                continue
            example_dir = job_file.parent
            # Skip excluded examples
            if any(excl in str(example_dir) for excl in EXCLUDED_EXAMPLES):
                continue
            # Skip nested examples (no requirements.txt in same directory)
            if root == ADVANCED_ROOT and not (example_dir / "requirements.txt").exists():
                continue
            rel_path = example_dir.relative_to(EXAMPLES_ROOT)
            examples.append((str(rel_path), example_dir))
    return sorted(examples, key=lambda x: x[0])


EXAMPLES = discover_examples()


@pytest.mark.parametrize("name,path", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_example(name, path):
    """Run an example's job.py in a fresh venv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        venv_dir = tmpdir / "venv"
        
        # Create isolated venv
        venv.create(venv_dir, with_pip=True)
        py = str(venv_dir / "bin" / "python")
        
        # Set up environment
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["NVFLARE_WORKSPACE_ROOT"] = str(tmpdir / "workspace")  # Unique per test
        
        # Install nvflare from pre-built wheel (avoids parallel build conflicts)
        subprocess.run([py, "-m", "pip", "install", "-q", WHEEL_PATH],
                       env=env, check=True, timeout=300)
        
        # Install example requirements (filtering out nvflare to keep local version)
        req_file = path / "requirements.txt"
        if req_file.exists():
            reqs = [line.strip() for line in req_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#") and not line.lower().startswith("nvflare")]
            if reqs:
                subprocess.run([py, "-m", "pip", "install", "-q"] + reqs,
                               env=env, check=False, timeout=900)
        
        # Run data prep scripts
        for script in path.glob("*.py"):
            if script.stem in ("download_data", "prepare_data", "generate_pretrain_models"):
                subprocess.run([py, str(script)], cwd=path, env=env, check=False, timeout=900)
        for script in path.glob("*.sh"):
            if script.stem in ("download_data", "prepare_data", "generate_pretrain_models"):
                subprocess.run(["bash", str(script)], cwd=path, env=env, check=False, timeout=900)
        
        # Run job.py
        result = subprocess.run([py, "job.py"], cwd=path, env=env, timeout=3600)
        assert result.returncode == 0, f"job.py failed: {result.returncode}"


if __name__ == "__main__":
    print(f"Discovered {len(EXAMPLES)} examples:")
    for name, path in EXAMPLES:
        print(f"  - {name}")
