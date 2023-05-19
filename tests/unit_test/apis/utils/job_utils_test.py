# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from nvflare.apis.utils.job_utils import convert_legacy_zipped_app_to_job
from nvflare.fuel.utils.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes


def create_fake_app(app_root: Path):
    os.makedirs(app_root)
    os.mkdir(app_root / "config")
    open(app_root / "config" / "config_fed_server.json", "w").close()
    open(app_root / "config" / "config_fed_client.json", "w").close()
    os.mkdir(app_root / "custom")
    open(app_root / "custom" / "c1.py", "w").close()
    open(app_root / "custom" / "c2.py", "w").close()


def create_fake_job(temp_dir, job_name, app_name):
    root_dir = Path(temp_dir) / job_name
    os.makedirs(root_dir)
    create_fake_app(root_dir / app_name)
    with open(root_dir / "meta.json", "w") as f:
        f.write("{}")


@pytest.fixture()
def create_fake_app_dir():
    """
    app/
        config/
            config_fed_server.json
            config_fed_client.json
        custom/
            c1.py
            c2.py
    expect result:
        app/
            app/
                config/
                    config_fed_server.json
                    config_fed_client.json
                custom/
                    c1.py
                    c2.py
            meta.json
    """
    temp_dir = tempfile.mkdtemp()
    app_name = "app"
    root_dir = Path(temp_dir) / app_name
    create_fake_app(root_dir)

    temp_dir2 = tempfile.mkdtemp()
    create_fake_job(temp_dir2, app_name, app_name)
    yield temp_dir, app_name, temp_dir2
    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_dir2)


@pytest.fixture()
def create_fake_job_dir():
    """
    fed_avg/
        app/
            config/
                config_fed_server.json
                config_fed_client.json
            custom/
                c1.py
                c2.py
        meta.json
    """
    temp_dir = tempfile.mkdtemp()
    job_name = "fed_avg"
    app_name = "app"
    create_fake_job(temp_dir, job_name, app_name)
    yield temp_dir, job_name
    shutil.rmtree(temp_dir)


class TestJobUtils:
    def test_convert_legacy_zip_job(self, create_fake_job_dir):
        tmp_dir, job_name = create_fake_job_dir
        zip_data = zip_directory_to_bytes(root_dir=tmp_dir, folder_name=job_name)
        new_bytes = convert_legacy_zipped_app_to_job(zip_data)

        output_tmp_dir = tempfile.mkdtemp()
        unzip_all_from_bytes(new_bytes, output_dir_name=output_tmp_dir)
        # stays the same
        for i, j in zip(os.walk(tmp_dir), os.walk(output_tmp_dir)):
            assert i[1] == j[1]
            assert i[2] == j[2]

        shutil.rmtree(output_tmp_dir)

    def test_convert_legacy_zip_app(self, create_fake_app_dir):
        tmp_dir, app_name, tmp_dir_with_job = create_fake_app_dir
        zip_data = zip_directory_to_bytes(root_dir=tmp_dir, folder_name=app_name)
        new_bytes = convert_legacy_zipped_app_to_job(zip_data)

        output_tmp_dir = tempfile.mkdtemp()
        unzip_all_from_bytes(new_bytes, output_dir_name=output_tmp_dir)
        for i, j in zip(os.walk(tmp_dir_with_job), os.walk(output_tmp_dir)):
            assert i[1] == j[1]
            assert i[2] == j[2]

        shutil.rmtree(output_tmp_dir)
