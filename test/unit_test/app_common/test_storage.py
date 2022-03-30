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
import pickle
import random
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pytest

from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from nvflare.app_common.storages.s3_storage import S3Storage


def random_string(length):
    s = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = "".join(random.sample(s, length))
    return p


def random_path(depth):
    path = "/".join([random_string(4) for _ in range(depth)])
    return path


def random_data():
    return bytearray(random.getrandbits(8) for _ in range(16384))


def random_meta():
    return {random.getrandbits(8): random.getrandbits(8) for _ in range(32)}


@pytest.mark.xdist_group(name="storage_tests_group")
@pytest.mark.parametrize("storage", ["FilesystemStorage", "S3Storage"])
class TestStorage:

    tmp_dir = None
    storages = defaultdict()
    minio_ps = None

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        tmp_dir_name = cls.tmp_dir.name

        # start local minio server as a compatible S3 bucket for testing

        commands = [
            ["wget", "https://dl.min.io/server/minio/release/linux-amd64/minio"],
            ["chmod", "+x", "minio"],
            ["./minio", "server", os.path.join(tmp_dir_name, "s3-storage"), "--console-address", ":9001"]
        ]

        env = os.environ.update({"MINIO_ROOT_USER": "admin", "MINIO_ROOT_PASSWORD": "password"})
        for command in commands:
            cls.minio_ps = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            if command[0] != "./minio":
                cls.minio_ps.communicate()

        time.sleep(15)

        cls.storages["FilesystemStorage"] = FilesystemStorage(root_dir=os.path.join(tmp_dir_name, "filesystem-storage"))
        cls.storages["S3Storage"] = S3Storage(endpoint="localhost:9000", access_key="admin", secret_key="password", secure=False, bucket_name="nvflare-storage")

        print("Finished setup.")

    @classmethod
    def teardown_class(cls):
        os.remove("minio")
        if isinstance(cls.minio_ps, subprocess.Popen):
            cls.minio_ps.kill()
        cls.tmp_dir.cleanup()

        print("Finished teardown.")

    @pytest.mark.parametrize("n_files", [20, 100])
    @pytest.mark.parametrize("n_folders", [5, 20])
    @pytest.mark.parametrize("path_depth", [3, 10])
    def test_large_storage(self, n_folders, n_files, path_depth, storage):
        storage = self.storages[storage]
        test_tmp_dir = tempfile.TemporaryDirectory()
        test_tmp_dir_name = test_tmp_dir.name
        dir_to_files = defaultdict(list)
        print(f"Prepare data {n_files} files for {n_folders} folders")

        for _ in range(n_folders):
            basepath = "/" + random_path(path_depth)

            for i in range(round(n_files / n_folders)):

                # distribute files among path_depth levels of directory depth
                dirpath = basepath
                for _ in range(round(i / (n_files / path_depth))):
                    dirpath = os.path.split(dirpath)[0]

                filename = random_string(8)
                dir_to_files[dirpath].append(os.path.join(dirpath, filename))
                filepath = os.path.join(dirpath, filename)

                test_filepath = os.path.join(test_tmp_dir_name, filepath.lstrip("/"))
                Path(test_filepath).mkdir(parents=True, exist_ok=True)

                # use f.write() as reference to compare with storage implementation
                with open(os.path.join(test_filepath, "data"), "wb") as f:
                    data = random_data()
                    f.write(pickle.dumps(data))

                with open(os.path.join(test_filepath, "meta"), "wb") as f:
                    meta = random_meta()
                    f.write(pickle.dumps(meta))

                self.test_create_read(storage, filepath, data, meta, unittest=False)
                self.test_create_overwrite(storage, filepath, data, meta, unittest=False)

                self.test_create_nonempty(storage, filepath, random_data(), random_meta(), unittest=False)
                self.test_create_inside_prexisting(storage, filepath, random_data(), random_meta(), unittest=False)

        verified = 0
        for test_dirpath, _, object_files in os.walk(test_tmp_dir_name):

            dirpath = "/" + test_dirpath[len(test_tmp_dir_name) :].lstrip("/")
            self.test_list(storage, dirpath, dir_to_files, unittest=False)

            # if dirpath is an object
            if object_files:
                with open(os.path.join(test_dirpath, "data"), "rb") as f:
                    data = pickle.loads(f.read())
                with open(os.path.join(test_dirpath, "meta"), "rb") as f:
                    meta = pickle.loads(f.read())

                self.test_data_read_update(storage, dirpath, data, meta, unittest=False)
                self.test_meta_read_update(storage, dirpath, data, meta, unittest=False)

                self.test_delete(storage, dirpath, unittest=False)

                verified += 1

        if callable(getattr(storage, "finalize", None)):
            storage.finalize()

        test_tmp_dir.cleanup()

        print(f"Verified {verified} files")

    @pytest.mark.parametrize(
        "uri, data, meta, overwrite_existing",
        [
            (1234, random_data(), random_meta(), True),
            ("/test_dir/test_object", random_data(), "not a dictionary", True),
            ("/test_dir/test_object", random_data(), random_meta(), "not a bool"),
            ("/test_dir/test_object", "not a byte string", random_meta(), True),
        ],
    )
    def test_invalid_inputs(self, storage, uri, data, meta, overwrite_existing):
        storage = self.storages[storage]

        invalid_uri = 1234
        invalid_data = "not a byte string"
        invalid_meta = "not a dict"
        invalid_bool = "not a bool"

        with pytest.raises(TypeError):
            storage.create_object(uri, data, meta, overwrite_existing)

        if uri == invalid_uri or meta == invalid_meta or overwrite_existing == invalid_bool:
            with pytest.raises(TypeError):
                storage.update_meta(uri, meta, overwrite_existing)

        if uri == invalid_uri or data == invalid_data:
            with pytest.raises(TypeError):
                storage.update_data(uri, data)

        if uri == invalid_uri:
            with pytest.raises(TypeError):
                storage.list_objects(uri)
            with pytest.raises(TypeError):
                storage.get_meta(uri)
            with pytest.raises(TypeError):
                storage.get_full_meta(uri)
            with pytest.raises(TypeError):
                storage.get_data(uri)
            with pytest.raises(TypeError):
                storage.get_detail(uri)
            with pytest.raises(TypeError):
                storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, data, meta, unittest",
        [("/test_dir/test_object", random_data(), random_meta(), True)],
    )
    def test_create_read(self, storage, uri, data, meta, unittest):
        if unittest:
            storage = self.storages[storage]

        storage.create_object(uri, data, meta, overwrite_existing=True)

        # get_data()
        assert storage.get_data(uri) == data
        assert storage.get_detail(uri)[1] == data

        # get_meta()
        assert storage.get_meta(uri) == meta
        assert storage.get_detail(uri)[0] == meta

        if unittest:
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, data, meta, unittest",
        [("/test_dir/test_object", random_data(), random_meta(), True)],
    )
    def test_create_overwrite(self, storage, uri, data, meta, unittest):
        if unittest:
            storage = self.storages[storage]

        storage.create_object(uri, random_data(), random_meta(), overwrite_existing=True)
        storage.create_object(uri, data, meta, overwrite_existing=True)
        with pytest.raises(RuntimeError):
            storage.create_object(uri, data, meta, overwrite_existing=False)

        if unittest:
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, data, meta, unittest",
        [("/test_dir/test_object", random_data(), random_meta(), True)],
    )
    def test_create_nonempty(self, storage, uri, data, meta, unittest):
        if unittest:
            storage = self.storages[storage]
            storage.create_object(uri, data, meta, True)

        # cannot create object at nonempty directory
        with pytest.raises(Exception):
            storage.create_object(os.path.split(uri)[0], random_data(), random_meta(), overwrite_existing=True)

        if unittest:
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, data, meta, unittest",
        [("/test_dir/test_object", random_data(), random_meta(), True)],
    )
    def test_create_inside_prexisting(self, storage, uri, data, meta, unittest):
        if unittest:
            storage = self.storages[storage]
            storage.create_object(uri, data, meta, True)

        # cannot create object inside a prexisiting object
        with pytest.raises(Exception):
            storage.create_object(
                os.path.join(uri, random_path(3)), random_data(), random_meta(), overwrite_existing=True
            )

        if unittest:
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "dirpath, dir_to_files, unittest",
        [("/test_dir/test_object", defaultdict(list), True)],
    )
    def test_list(self, storage, dirpath, dir_to_files, unittest):
        if unittest:
            storage = self.storages[storage]
            dir_to_files = defaultdict(list)
            for i in range(10):
                object_uri = os.path.join(dirpath, str(i))
                storage.create_object(object_uri, random_data(), random_meta(), overwrite_existing=True)
                dir_to_files[dirpath].append(object_uri)

        assert set(storage.list_objects(dirpath)) == set(dir_to_files.get(dirpath, []))

        if unittest:
            for i in range(10):
                object_uri = os.path.join(dirpath, str(i))
                storage.delete_object(object_uri)

    @pytest.mark.parametrize(
        "uri, unittest",
        [("/test_dir/test_object", True)],
    )
    def test_delete(self, storage, uri, unittest):
        if unittest:
            storage = self.storages[storage]
            storage.create_object(uri, random_data(), random_meta(), overwrite_existing=True)

        storage.delete_object(uri)

        # methods on non-existent object
        with pytest.raises(RuntimeError):
            data3 = random_data()
            storage.update_data(uri, data3)
        with pytest.raises(RuntimeError):
            meta4 = random_meta()
            storage.update_meta(uri, meta4, replace=True)
        with pytest.raises(RuntimeError):
            storage.get_data(uri)
        with pytest.raises(RuntimeError):
            storage.get_meta(uri)
        with pytest.raises(RuntimeError):
            storage.get_detail(uri)
        with pytest.raises(RuntimeError):
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, data, meta, unittest",
        [("/test_dir/test_object", random_data(), random_meta(), True)],
    )
    def test_data_read_update(self, storage, uri, data, meta, unittest):
        if unittest:
            storage = self.storages[storage]
            storage.create_object(uri, data, meta, overwrite_existing=True)

        # get_data()
        assert storage.get_data(uri) == data
        assert storage.get_detail(uri)[1] == data

        # update_data()
        data2 = bytearray(random.getrandbits(8) for _ in range(16384))
        storage.update_data(uri, data2)
        assert storage.get_data(uri) == data2
        assert storage.get_detail(uri)[1] == data2

        if unittest:
            storage.delete_object(uri)

    @pytest.mark.parametrize(
        "uri, data, meta, unittest",
        [("/test_dir/test_object", random_data(), random_meta(), True)],
    )
    def test_meta_read_update(self, storage, uri, data, meta, unittest):
        if unittest:
            storage = self.storages[storage]
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

        if unittest:
            storage.delete_object(uri)
