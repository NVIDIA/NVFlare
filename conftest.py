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
import shutil
import subprocess
from pathlib import Path

try:
    import nbformat
except Exception:
    nbformat = None

_executed_notebooks = set()
_passed_notebooks = set()
_backup_notebooks = set()  # Track notebooks with backup files


def pytest_collection_modifyitems(config, items):
    """Filter tagged cells from notebooks before nbmake runs"""
    # Only run if --nbmake flag is present
    if not config.getoption('--nbmake', default=False):
        return

    kernel_name = config.getoption('--kernel', default="")

    for item in items:
        if _is_notebook_item(item):
            notebook_path = item.path if hasattr(item, 'path') else item.fspath
            filter_notebook(notebook_path, kernel_name)


def filter_notebook(notebook_path, kernel_name):
    """Remove cells tagged with 'skip-execution' and handle kernel specifications"""
    nb = nbformat.read(notebook_path, as_version=4)
    
    kernel_spec_updated = False
    current_spec = nb.metadata.get("kernelspec", {})
    spec_kernel_name = current_spec.get("name", "")
    
    # Determine which kernel to use
    if kernel_name:
        # User provided --kernel flag, use that
        target_kernel = kernel_name
    else:
        # No --kernel flag, use notebook's existing kernel
        target_kernel = spec_kernel_name
    
    # Handle kernel specification
    if current_spec:
        # Check if target kernel exists and differs from current spec
        if target_kernel:
            is_registered = is_kernel_registered(target_kernel)
            
            if is_registered and target_kernel != spec_kernel_name:
                # Update to the registered target kernel
                nb.metadata["kernelspec"] = {
                    "display_name": target_kernel,
                    "language": "python",
                    "name": target_kernel
                }
                print(f"Updated kernel: '{spec_kernel_name}' → '{target_kernel}'")
                kernel_spec_updated = True
            elif not is_registered:
                # Target kernel not registered, remove kernelspec
                del nb.metadata["kernelspec"]
                print(f"Warning: Kernel '{target_kernel}' not registered. Removed kernelspec - nbmake will use current interpreter.")
                kernel_spec_updated = True
            # else: kernel is registered and matches current spec, no change needed
    
    # Filter cells
    filtered_cells = []
    for cell in nb.cells:
        tags = cell.get('metadata', {}).get('tags', [])
        if any(tag in ['skip-execution', 'skip', 'colab'] for tag in tags):
            continue
        filtered_cells.append(cell)
    
    cell_skipped = len(filtered_cells) != len(nb.cells)
    if cell_skipped:
        nb.cells = filtered_cells
    
    # Write modified notebook to disk if anything changed
    # (nbmake reads from disk, so changes must be persisted)
    if cell_skipped or kernel_spec_updated:
        # Create backup of original notebook before modifying
        notebook_path = Path(notebook_path)
        backup_path = notebook_path.with_suffix(notebook_path.suffix + '.backup')
        
        if backup_path.exists():
            # Existing backup found - assume it's the true original
            print(f"Found existing backup: {backup_path.name}")
        else:
            shutil.copy(notebook_path, backup_path)
            print(f"Created backup: {backup_path.name}")
        _backup_notebooks.add(notebook_path)
        
        # Write modified notebook
        nbformat.write(nb, notebook_path)
        if cell_skipped:
            print(f"Filtered {len(nb.cells)} → {len(filtered_cells)} cells in {notebook_path.name}")
        if kernel_spec_updated:
            print(f"Updated kernel spec in {notebook_path.name}")


def is_kernel_registered(kernel_name):
    """Check if a kernel is registered in Jupyter"""
    if not kernel_name:
        return False
    
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        ksm = KernelSpecManager()
        return kernel_name in ksm.get_all_specs()
    except Exception:
        # If jupyter_client isn't available or fails, assume kernel doesn't exist
        return False

def restore_notebooks():
    """Restore original notebooks from .backup files"""
    if not _backup_notebooks:
        return
    
    restored = []
    for notebook_path in _backup_notebooks:
        backup_path = notebook_path.with_suffix(Path(notebook_path).suffix + '.backup')
        if backup_path.exists():
            shutil.move(backup_path, notebook_path)
            restored.append(str(notebook_path))
    
    if restored:
        print(f"\n[notebook-restore] Restored {len(restored)} original notebook(s):")
        for p in restored:
            print("  -", p)

def pytest_addoption(parser):
    parser.addoption(
        "--nbmake-clean",
        action="store",
        default="on-success",
        choices=["always", "on-success", "never"],
        help="When to clear outputs from executed notebooks: always, on-success (default), never",
    )
    parser.addoption(
        "--kernel",
        action="store",
        default="",
        help="specify the kernel name",
    )


def _is_notebook_item(item):
    """Check if item is a notebook, compatible with old and new pytest versions"""
    try:
        # Try newer pytest attribute first (pytest 7+)
        if hasattr(item, 'path'):
            return str(item.path).endswith('.ipynb')
        # Fall back to older pytest attribute
        return Path(str(item.fspath)).suffix == ".ipynb"
    except Exception:
        return False

def pytest_runtest_setup(item):
    # record that this notebook is being executed
    if _is_notebook_item(item):
        notebook_path = Path(str(item.path)) if hasattr(item, 'path') else Path(str(item.fspath))
        _executed_notebooks.add(notebook_path)

def pytest_runtest_makereport(item, call):
    # called for setup/call/teardown — only consider the 'call' phase
    if call.when != "call":
        return
    if not _is_notebook_item(item):
        return
    # success if no exception info
    if call.excinfo is None:
        notebook_path = Path(str(item.path)) if hasattr(item, 'path') else Path(str(item.fspath))
        _passed_notebooks.add(notebook_path)

def _clear_outputs_with_nbformat(nb_path: Path):
    if nbformat is None:
        return False
    nb = nbformat.read(str(nb_path), as_version=nbformat.NO_CONVERT)
    changed = False
    for cell in nb.cells:
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if "execution_count" in cell and cell["execution_count"] is not None:
            cell["execution_count"] = None
            changed = True
    if changed:
        nbformat.write(nb, str(nb_path))
    return changed

def _clear_outputs_with_nbconvert(nb_path: Path):
    # fallback if nbformat isn't available or you prefer nbconvert
    subprocess.run([
        "jupyter", "nbconvert",
        "--ClearOutputPreprocessor.enabled=True",
        "--inplace", str(nb_path)
    ], check=False)

def pytest_sessionfinish(session, exitstatus):
    # First, restore original notebooks (before cleaning outputs)
    restore_notebooks()
    
    mode = session.config.getoption("--nbmake-clean")
    if mode == "never":
        return

    if mode == "always":
        to_clean = _executed_notebooks
    else:  # on-success
        to_clean = _passed_notebooks

    if not to_clean:
        return

    cleaned = []
    for nb in sorted(to_clean):
        # prefer programmatic clearing; fallback to nbconvert
        ok = False
        if nbformat is not None:
            try:
                ok = _clear_outputs_with_nbformat(nb)
            except Exception:
                ok = False
        if not ok:
            _clear_outputs_with_nbconvert(nb)
        cleaned.append(str(nb))

    print(f"\n[nbmake-clean] cleaned outputs from {len(cleaned)} executed notebook(s):")
    for p in cleaned:
        print("  -", p)
