# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.hci.zip_utils import (
    convert_legacy_zip,
    get_all_file_paths,
    normpath_for_zip,
    remove_leading_dotdot,
    split_path,
    unzip_all_from_bytes,
    zip_directory_to_bytes,
)


def create_fake_job(temp_dir, job_name):
    root_dir = Path(temp_dir) / job_name
    os.makedirs(root_dir)
    os.mkdir(root_dir / "fed_avg_app")
    os.mkdir(root_dir / "fed_avg_app" / "config")
    open(root_dir / "fed_avg_app" / "config" / "config_fed_server.json", "w").close()
    open(root_dir / "fed_avg_app" / "config" / "config_fed_client.json", "w").close()
    os.mkdir(root_dir / "fed_avg_app" / "custom")
    open(root_dir / "fed_avg_app" / "custom" / "c1.py", "w").close()
    open(root_dir / "fed_avg_app" / "custom" / "c2.py", "w").close()
    with open(root_dir / "meta.json", "w") as f:
        f.write("{}")


@pytest.fixture()
def create_fake_app_dir():
    """
    fed_avg_app/
        config/
            config_fed_server.json
            config_fed_client.json
        custom/
            c1.py
            c2.py
    expect result:
        fed_avg_app/
            fed_avg_app/
                config/
                    config_fed_server.json
                    config_fed_client.json
                custom/
                    c1.py
                    c2.py
            meta.json
    """
    temp_dir = tempfile.mkdtemp()
    app_name = "fed_avg_app"
    root_dir = Path(temp_dir) / app_name
    os.makedirs(root_dir)
    os.mkdir(root_dir / "config")
    open(root_dir / "config" / "config_fed_server.json", "w").close()
    open(root_dir / "config" / "config_fed_client.json", "w").close()
    os.mkdir(root_dir / "custom")
    open(root_dir / "custom" / "c1.py", "w").close()
    open(root_dir / "custom" / "c2.py", "w").close()

    temp_dir2 = tempfile.mkdtemp()
    create_fake_job(temp_dir2, "fed_avg_app")
    yield temp_dir, app_name, temp_dir2
    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_dir2)


@pytest.fixture()
def create_fake_job_dir():
    """
    fed_avg/
        fed_avg_app/
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
    create_fake_job(temp_dir, job_name)
    yield temp_dir, job_name
    shutil.rmtree(temp_dir)


@pytest.fixture()
def create_fake_dir():
    """
    /a/b/c/
      folder1/
        file1
        file2
      folder2/
        file3
        file4
      folder3/
        folder4/
          file5
        folder5/
      folder6/
      file6
      file7
    """
    temp_dir = Path(tempfile.mkdtemp())
    prefix = os.path.sep.join(["a", "b", "c"])
    root_dir = temp_dir / prefix
    os.makedirs(root_dir)
    os.mkdir(root_dir / "folder1")
    open(root_dir / "folder1" / "file1", "w").close()
    open(root_dir / "folder1" / "file2", "w").close()
    os.mkdir(root_dir / "folder2")
    open(root_dir / "folder2" / "file3", "w").close()
    open(root_dir / "folder2" / "file4", "w").close()
    os.mkdir(root_dir / "folder3")
    os.mkdir(root_dir / "folder3" / "folder4")
    open(root_dir / "folder3" / "folder4" / "file5", "w").close()
    os.mkdir(root_dir / "folder3" / "folder5")
    os.mkdir(root_dir / "folder6")
    open(root_dir / "file6", "w").close()
    open(root_dir / "file7", "w").close()
    yield temp_dir, prefix
    shutil.rmtree(temp_dir)


class TestSecurityUtils:
    @pytest.mark.parametrize(
        "path, output",
        [
            (os.path.sep.join(["..", "..", "..", "hello"]), "hello"),
            (f".{os.path.sep}hello", "hello"),
            (os.path.sep.join(["..", "..", "..", "hello", "motor"]), f"hello{os.path.sep}motor"),
            (os.path.sep.join(["..", "..", "..", "hello", "..", "motor"]), os.path.sep.join(["hello", "..", "motor"])),
            (f"{os.path.abspath(os.path.sep)}hello", f"{os.path.abspath(os.path.sep)}hello"),
        ],
    )
    def test_remove_leading_dotdot(self, path, output):
        assert remove_leading_dotdot(path) == output

    @pytest.mark.parametrize(
        "path, output",
        [
            ("hello", ("", "hello")),
            (f".{os.path.sep}hello", ("", "hello")),
            (
                os.path.sep.join(["..", "..", "..", "hello", "..", "motor"]),
                (os.path.sep.join(["..", "..", "..", "hello", ".."]), "motor"),
            ),
            (f"{os.path.abspath(os.path.sep)}hello", (os.path.abspath(os.path.sep), "hello")),
            (f"hello{os.path.sep}", ("", "hello")),
            (
                os.path.sep.join(["..", "hello", "..", "motor", ""]),
                (os.path.sep.join(["..", "hello", ".."]), "motor"),
            ),
        ],
    )
    def test_split_path(self, path, output):
        assert split_path(path) == output

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("hello", "hello"),
            ("PPAP\\ABCD", "PPAP/ABCD"),
            ("/home/random_dir/something.txt", "/home/random_dir/something.txt"),
        ],
    )
    def test_normpath_for_zip(self, path, expected):
        assert normpath_for_zip(path) == expected

    def test_get_all_file_paths(self, create_fake_dir):
        tmp_dir, prefix = create_fake_dir
        test_path = os.path.join(tmp_dir, prefix)
        assert len(get_all_file_paths(test_path)) == 13

    def test_zip_unzip(self, create_fake_dir):
        tmp_dir, prefix = create_fake_dir
        first, second = os.path.split(prefix)
        root_dir = os.path.join(tmp_dir, first)
        zip_data = zip_directory_to_bytes(root_dir=root_dir, folder_name=second)

        temp_dir = tempfile.mkdtemp()
        unzip_all_from_bytes(zip_data, output_dir_name=temp_dir)
        for i, j in zip(os.walk(root_dir), os.walk(temp_dir)):
            assert i[1] == j[1]
            assert i[2] == j[2]
        shutil.rmtree(temp_dir)

    def test_convert_legacy_zip_job(self, create_fake_job_dir):
        tmp_dir, job_name = create_fake_job_dir
        zip_data = zip_directory_to_bytes(root_dir=tmp_dir, folder_name=job_name)
        new_bytes = convert_legacy_zip(zip_data)

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
        new_bytes = convert_legacy_zip(zip_data)

        output_tmp_dir = tempfile.mkdtemp()
        unzip_all_from_bytes(new_bytes, output_dir_name=output_tmp_dir)
        for i, j in zip(os.walk(tmp_dir_with_job), os.walk(output_tmp_dir)):
            assert i[1] == j[1]
            assert i[2] == j[2]

        shutil.rmtree(output_tmp_dir)
