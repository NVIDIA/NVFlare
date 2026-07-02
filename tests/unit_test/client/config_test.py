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

import json
import os
import stat

import pytest

from nvflare.client.config import CONFIG_FILE_PERMISSION, ClientConfig, from_file, write_config_to_file


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def restore_umask():
    """Pin umask so fresh-write permission assertions are deterministic."""
    old = os.umask(0o022)
    yield
    os.umask(old)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions required")
class TestConfigFilePermissions:
    def test_to_json_fresh_write_is_owner_only(self, tmp_path, restore_umask):
        config_file = str(tmp_path / "client_api_config.json")
        ClientConfig({"SITE_NAME": "site-1"}).to_json(config_file)

        assert _mode(config_file) == CONFIG_FILE_PERMISSION
        with open(config_file) as f:
            assert json.load(f) == {"SITE_NAME": "site-1"}

    def test_write_config_to_file_fresh_write_is_owner_only(self, tmp_path, restore_umask):
        config_file = str(tmp_path / "client_api_config.json")
        write_config_to_file(config_data={"AUTH_TOKEN": "secret"}, config_file_path=config_file)

        assert _mode(config_file) == CONFIG_FILE_PERMISSION

    def test_write_config_to_file_tightens_pre_existing_world_readable_file(self, tmp_path, restore_umask):
        config_file = str(tmp_path / "client_api_config.json")
        with open(config_file, "w") as f:
            json.dump({"SITE_NAME": "site-1"}, f)
        os.chmod(config_file, 0o644)
        assert _mode(config_file) == 0o644

        write_config_to_file(config_data={"AUTH_TOKEN": "secret"}, config_file_path=config_file)

        assert _mode(config_file) == CONFIG_FILE_PERMISSION
        # update-in-place must merge with the existing content
        config = from_file(config_file)
        assert config.config == {"SITE_NAME": "site-1", "AUTH_TOKEN": "secret"}

    def test_to_json_tightens_pre_existing_world_readable_file(self, tmp_path, restore_umask):
        config_file = str(tmp_path / "client_api_config.json")
        with open(config_file, "w") as f:
            f.write("{}")
        os.chmod(config_file, 0o644)

        ClientConfig({"AUTH_TOKEN": "secret"}).to_json(config_file)

        assert _mode(config_file) == CONFIG_FILE_PERMISSION

    def test_to_json_fails_closed_when_permission_cannot_be_set(self, tmp_path, monkeypatch):
        # If the file cannot be secured (e.g. pre-existing, owned by another user),
        # the write must be refused rather than writing the token into an exposed file.
        config_file = str(tmp_path / "client_api_config.json")
        with open(config_file, "w") as f:
            f.write('{"SITE_NAME": "site-1"}')
        os.chmod(config_file, 0o644)

        def deny_fchmod(fd, mode):
            raise PermissionError("not owner")

        monkeypatch.setattr(os, "fchmod", deny_fchmod)

        with pytest.raises(RuntimeError, match="owner-only"):
            ClientConfig({"AUTH_TOKEN": "secret"}).to_json(config_file)

        # the secret must not have been written
        with open(config_file) as f:
            assert "secret" not in f.read()

    @pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW required")
    def test_to_json_rejects_symlink_target(self, tmp_path, restore_umask):
        target = tmp_path / "victim.txt"
        target.write_text("important")
        link = str(tmp_path / "client_api_config.json")
        os.symlink(str(target), link)

        with pytest.raises(OSError):
            ClientConfig({"AUTH_TOKEN": "secret"}).to_json(link)

        # the symlink target must be untouched
        assert target.read_text() == "important"


class TestConfigFileContent:
    def test_write_config_to_file_round_trip(self, tmp_path):
        config_file = str(tmp_path / "client_api_config.json")
        write_config_to_file(config_data={"SITE_NAME": "site-1"}, config_file_path=config_file)
        write_config_to_file(config_data={"JOB_ID": "job-1"}, config_file_path=config_file)

        config = from_file(config_file)
        assert config.config == {"SITE_NAME": "site-1", "JOB_ID": "job-1"}
