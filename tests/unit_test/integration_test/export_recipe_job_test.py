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

import pytest

from tests.integration_test.tools.export_recipe_job import export_recipe_from_job_py


@pytest.mark.parametrize("job_file", ("../outside.py", "nested/../../outside.py"))
def test_export_recipe_rejects_job_file_path_traversal(tmp_path, job_file):
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    (tmp_path / "outside.py").touch()

    with pytest.raises(ValueError, match="job_file must resolve within recipe_dir"):
        export_recipe_from_job_py(str(recipe_dir), str(tmp_path / "output"), job_file=job_file)


def test_export_recipe_rejects_absolute_job_file_path(tmp_path):
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    outside_job = tmp_path / "outside.py"
    outside_job.touch()

    with pytest.raises(ValueError, match="job_file must resolve within recipe_dir"):
        export_recipe_from_job_py(str(recipe_dir), str(tmp_path / "output"), job_file=str(outside_job))


def test_export_recipe_rejects_job_file_symlink_outside_recipe_dir(tmp_path):
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    outside_job = tmp_path / "outside.py"
    outside_job.touch()
    (recipe_dir / "job.py").symlink_to(outside_job)

    with pytest.raises(ValueError, match="job_file must resolve within recipe_dir"):
        export_recipe_from_job_py(str(recipe_dir), str(tmp_path / "output"))
