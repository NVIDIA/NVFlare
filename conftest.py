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

import pytest
import nbformat
from pathlib import Path


def pytest_collection_modifyitems(config, items):
    """Filter tagged cells from notebooks before nbmake runs"""
    # Only run if --nbmake flag is present
    if not config.getoption('--nbmake', default=False):
        return

    for item in items:
        if hasattr(item, 'path') and str(item.path).endswith('.ipynb'):
            filter_notebook(item.path)


def filter_notebook(notebook_path):
    """Remove cells tagged with 'skip-execution'"""
    nb = nbformat.read(notebook_path, as_version=4)

    filtered_cells = []
    for cell in nb.cells:
        tags = cell.get('metadata', {}).get('tags', [])
        if any(tag in ['skip-execution', 'skip', 'colab'] for tag in tags):
            continue
        filtered_cells.append(cell)

    if len(filtered_cells) != len(nb.cells):
        nb.cells = filtered_cells
        nbformat.write(nb, notebook_path)