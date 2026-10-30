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

import io
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from nvflare.fuel.utils.zip_utils import (
    get_all_file_paths,
    ls_zip_from_bytes,
    normpath_for_zip,
    remove_leading_dotdot,
    split_path,
    unzip_all_from_bytes,
    unzip_single_file_from_bytes,
    zip_directory_to_bytes,
)


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


class TestZipUtils:
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

    @pytest.mark.parametrize(
        "cwd, root_dir, folder_name, expected_base",
        [
            (["a", "b"], "", "c", ["a", "b"]),
            (["a", "b"], "", "./c", ["a", "b"]),
            (["a", "b"], ".", "c", ["a", "b"]),
            (["a", "b"], "c", "", ["a", "b", "c"]),
            (["a", "b", "c"], ".", "", ["a", "b", "c"]),
            (["a", "b", "c"], "", "", ["a", "b", "c"]),
        ],
    )
    def test_zip_relative_root(self, create_fake_dir, monkeypatch, cwd, root_dir, folder_name, expected_base):
        tmp_dir, _ = create_fake_dir
        expected_folder = os.path.basename(os.path.normpath(folder_name)) if folder_name else ""
        expected = zip_directory_to_bytes(os.path.join(tmp_dir, *expected_base), expected_folder)

        monkeypatch.chdir(os.path.join(tmp_dir, *cwd))
        zip_data = zip_directory_to_bytes(root_dir, folder_name)

        names = sorted(i.filename for i in ls_zip_from_bytes(zip_data))
        assert names == sorted(i.filename for i in ls_zip_from_bytes(expected))
        if folder_name:
            assert all(n.startswith("c/") for n in names)

    @pytest.mark.parametrize("file_path", ["file.txt", "nested/file.txt", "nested/deep/file.txt"])
    @pytest.mark.parametrize("output_exists", [False, True])
    def test_unzip_single_file_keeps_relative_path(self, tmp_path, file_path, output_exists):
        archive_bytes = io.BytesIO()
        with ZipFile(archive_bytes, "w") as archive:
            archive.writestr(file_path, b"payload")

        output_dir = tmp_path / "output"
        if output_exists:
            output_dir.mkdir()

        unzip_single_file_from_bytes(archive_bytes.getvalue(), str(output_dir), file_path)

        assert (output_dir / file_path).read_bytes() == b"payload"
        extracted = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file())
        assert extracted == [file_path]

    def test_unzip_single_file_can_extract_again(self, tmp_path):
        archive_bytes = io.BytesIO()
        with ZipFile(archive_bytes, "w") as archive:
            archive.writestr("nested/file.txt", b"payload")

        output_dir = tmp_path / "output"
        for _ in range(2):
            unzip_single_file_from_bytes(archive_bytes.getvalue(), str(output_dir), "nested/file.txt")

        assert (output_dir / "nested" / "file.txt").read_bytes() == b"payload"

    def test_unzip_single_file_rejects_unsafe_path_before_creating_output(self, tmp_path):
        output_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="unsafe"):
            unzip_single_file_from_bytes(b"not a zip", str(output_dir), "../file.txt")
        assert not output_dir.exists()
