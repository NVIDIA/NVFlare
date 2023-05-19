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

import ast
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from nvflare.apis.storage import StorageException
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage


def random_string(length):
    s = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = "".join(random.sample(s, length))
    return p


def random_path(depth):
    path = os.sep.join([random_string(4) for _ in range(depth)])
    return path


def random_data():
    return bytes(bytearray(random.getrandbits(8) for _ in range(16384)))


def random_meta():
    return {random.getrandbits(8): random.getrandbits(8) for _ in range(32)}


ROOT_DIR = os.path.abspath(os.sep)


# TODO:: Add S3Storage test
@pytest.fixture(name="storage", params=["FilesystemStorage"])
def setup_and_teardown(request):
    print(f"setup {request.param}")
    if request.param == "FilesystemStorage":
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = FilesystemStorage(root_dir=os.path.join(tmp_dir, "filesystem-storage"))
    else:
        raise StorageException(f"Storage type {request.param} is not supported.")
    yield storage
    print("teardown")


class TestStorage:
    @pytest.mark.parametrize("n_files", [20, 100])
    @pytest.mark.parametrize("n_folders", [5, 20])
    @pytest.mark.parametrize("path_depth", [3, 10])
    def test_large_storage(self, storage, n_folders, n_files, path_depth):
        test_tmp_dir = tempfile.TemporaryDirectory()
        test_tmp_dir_name = test_tmp_dir.name
        dir_to_files = defaultdict(list)
        print(f"Prepare data {n_files} files for {n_folders} folders")

        for _ in range(n_folders):
            base_path = os.path.join(ROOT_DIR, random_path(path_depth))

            for i in range(round(n_files / n_folders)):

                # distribute files among path_depth levels of directory depth
                dir_path = base_path
                for _ in range(round(i / (n_files / path_depth))):
                    dir_path = os.path.split(dir_path)[0]

                filename = random_string(8)
                dir_to_files[dir_path].append(os.path.join(dir_path, filename))
                filepath = os.path.join(dir_path, filename)

                test_filepath = os.path.join(test_tmp_dir_name, filepath.lstrip("/"))
                Path(test_filepath).mkdir(parents=True, exist_ok=True)

                # use f.write() as reference to compare with storage implementation
                with open(os.path.join(test_filepath, "data"), "wb") as f:
                    data = random_data()
                    f.write(data)

                with open(os.path.join(test_filepath, "meta"), "wb") as f:
                    meta = random_meta()
                    f.write(json.dumps(str(meta)).encode("utf-8"))

                storage.create_object(filepath, data, meta, overwrite_existing=True)

        for test_dir_path, _, object_files in os.walk(test_tmp_dir_name):

            dir_path = "/" + test_dir_path[len(test_tmp_dir_name) :].lstrip("/")
            assert set(storage.list_objects(dir_path)) == set(dir_to_files[dir_path])

            # if dir_path is an object
            if object_files:
                with open(os.path.join(test_dir_path, "data"), "rb") as f:
                    data = f.read()
                with open(os.path.join(test_dir_path, "meta"), "rb") as f:
                    meta = ast.literal_eval(json.loads(f.read().decode("utf-8")))

                assert storage.get_data(dir_path) == data
                assert storage.get_detail(dir_path)[1] == data
                assert storage.get_meta(dir_path) == meta
                assert storage.get_detail(dir_path)[0] == meta

                storage.delete_object(dir_path)

        test_tmp_dir.cleanup()

    @pytest.mark.parametrize(
        "uri, data, meta, overwrite_existing",
        [
            (1234, b"c", {}, True),
            ("/test_dir/test_object", "not a byte string", {}, True),
            ("/test_dir/test_object", b"c", "not a dictionary", True),
            ("/test_dir/test_object", b"c", {}, "not a bool"),
        ],
    )
    def test_create_invalid_inputs(self, storage, uri, data, meta, overwrite_existing):
        with pytest.raises(TypeError):
            storage.create_object(uri, data, meta, overwrite_existing)

    def test_invalid_inputs(self, storage):
        uri = 1234
        with pytest.raises(TypeError):
            storage.list_objects(uri)
        with pytest.raises(TypeError):
            storage.get_meta(uri)
        with pytest.raises(TypeError):
            storage.get_data(uri)
        with pytest.raises(TypeError):
            storage.get_detail(uri)
        with pytest.raises(TypeError):
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, meta, overwrite_existing",
        [
            (1234, {}, True),
            ("/test_dir/test_object", "not a dictionary", True),
            ("/test_dir/test_object", {}, "not a bool"),
        ],
    )
    def test_update_meta_invalid_inputs(self, storage, uri, meta, overwrite_existing):
        with pytest.raises(TypeError):
            storage.update_meta(uri, meta, overwrite_existing)

    @pytest.mark.parametrize(
        "uri, data",
        [
            (1234, "not bytes"),
            ("/test_dir/test_object", "not bytes"),
        ],
    )
    def test_update_data_invalid_inputs(self, storage, uri, data):
        with pytest.raises(TypeError):
            storage.update_data(uri, data)

    @pytest.mark.parametrize(
        "uri",
        ["/test_dir/test_object"],
    )
    def test_create_read(self, storage, uri):
        data = random_data()
        meta = random_meta()
        storage.create_object(uri, data, meta, overwrite_existing=True)

        # get_data()
        assert storage.get_data(uri) == data
        assert storage.get_detail(uri)[1] == data

        # get_meta()
        assert storage.get_meta(uri) == meta
        assert storage.get_detail(uri)[0] == meta

        storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri",
        ["/test_dir/test_object"],
    )
    def test_create_overwrite(self, storage, uri):
        data = random_data()
        meta = random_meta()
        storage.create_object(uri, random_data(), random_meta(), overwrite_existing=True)
        storage.create_object(uri, data, meta, overwrite_existing=True)

        assert storage.get_data(uri) == data
        assert storage.get_meta(uri) == meta

        with pytest.raises(StorageException):
            storage.create_object(uri, data, meta, overwrite_existing=False)

        storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, test_uri",
        [("/test_dir/test_object", "/test_dir")],
    )
    def test_create_nonempty(self, storage, uri, test_uri):
        storage.create_object("/test_dir/test_object", random_data(), random_meta())

        # cannot create object at nonempty directory
        with pytest.raises(StorageException):
            storage.create_object(test_uri, random_data(), random_meta(), overwrite_existing=True)

        storage.delete_object(uri)

    @pytest.mark.parametrize(
        "dir_path, num",
        [("/test_dir/test_object", 10), ("/test_dir/test_happy/test_object", 20)],
    )
    def test_list(self, storage, dir_path, num):
        dir_to_files = defaultdict(list)
        for i in range(num):
            object_uri = os.path.join(dir_path, str(i))
            storage.create_object(object_uri, random_data(), random_meta())
            dir_to_files[dir_path].append(object_uri)

        assert set(storage.list_objects(dir_path)) == set(dir_to_files[dir_path])

        for i in range(num):
            object_uri = os.path.join(dir_path, str(i))
            storage.delete_object(object_uri)

    def test_delete(self, storage):
        uri = "/test_dir/test_object"
        storage.create_object(uri, random_data(), random_meta(), overwrite_existing=True)

        storage.delete_object(uri)

        # methods on non-existent object
        with pytest.raises(StorageException):
            data3 = random_data()
            storage.update_data(uri, data3)
        with pytest.raises(StorageException):
            meta4 = random_meta()
            storage.update_meta(uri, meta4, replace=True)
        with pytest.raises(StorageException):
            storage.get_data(uri)
        with pytest.raises(StorageException):
            storage.get_meta(uri)
        with pytest.raises(StorageException):
            storage.get_detail(uri)
        with pytest.raises(StorageException):
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri",
        ["/test_dir/test_object"],
    )
    def test_data_read_update(self, storage, uri):
        data = random_data()
        meta = random_meta()
        storage.create_object(uri, data, meta, overwrite_existing=True)

        # get_data()
        assert storage.get_data(uri) == data
        assert storage.get_detail(uri)[1] == data

        # update_data()
        data2 = random_data()
        storage.update_data(uri, data2)
        assert storage.get_data(uri) == data2
        assert storage.get_detail(uri)[1] == data2

        storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri",
        ["/test_dir/test_object"],
    )
    def test_meta_read_update(self, storage, uri):
        data = random_data()
        meta = random_meta()
        storage.create_object(uri, data, meta, overwrite_existing=True)

        # get_meta()
        assert storage.get_meta(uri) == meta
        assert storage.get_detail(uri)[0] == meta

        # update_meta() w/ replace
        meta2 = random_meta()
        storage.update_meta(uri, meta2, replace=True)
        assert storage.get_meta(uri) == meta2
        assert storage.get_detail(uri)[0] == meta2

        # update_meta() w/o replace
        meta3 = random_meta()
        meta2.update(meta3)
        storage.update_meta(uri, meta3, replace=False)
        assert storage.get_meta(uri) == meta2
        assert storage.get_detail(uri)[0] == meta2

        storage.delete_object(uri)
