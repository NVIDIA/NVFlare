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

from argparse import Namespace

from nvflare.tool.cli_session import new_cli_session


def _make_admin_startup_kit(parent, name):
    admin_dir = parent / name
    startup_dir = admin_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text(f'{{"admin": {{"username": "{name}"}}}}', encoding="utf-8")
    (startup_dir / "client.crt").write_text("cert", encoding="utf-8")
    (startup_dir / "rootCA.pem").write_text("root", encoding="utf-8")
    return admin_dir


def test_new_cli_session_delegates_to_secure_session_factory():
    from unittest.mock import MagicMock, patch

    fake_sess = MagicMock()
    with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_sess) as new_secure:
        returned = new_cli_session("user", "/tmp/startup", timeout=2.5, study="default")

    new_secure.assert_called_once_with(
        username="user",
        startup_kit_location="/tmp/startup",
        debug=False,
        study="default",
        timeout=2.5,
        auto_login_max_tries=1,
    )
    assert returned is fake_sess


def test_new_cli_session_for_args_uses_explicit_startup_kit(tmp_path):
    from unittest.mock import MagicMock, patch

    from nvflare.tool.cli_session import new_cli_session_for_args

    startup_kit = _make_admin_startup_kit(tmp_path, "explicit@nvidia.com")
    fake_sess = MagicMock()

    with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_sess) as new_secure:
        returned = new_cli_session_for_args(Namespace(startup_kit=str(startup_kit)), timeout=2.5, study="default")

    assert returned is fake_sess
    assert new_secure.call_args.kwargs["username"] == "explicit@nvidia.com"
    assert new_secure.call_args.kwargs["startup_kit_location"] == str(startup_kit)


def test_new_cli_session_for_args_uses_kit_id_without_mutating_active(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    from nvflare.tool.cli_session import new_cli_session_for_args

    active_kit = _make_admin_startup_kit(tmp_path, "active@nvidia.com")
    scoped_kit = _make_admin_startup_kit(tmp_path, "scoped@nvidia.com")
    home = tmp_path / "home"
    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.conf"
    config_path.write_text(
        f"""
        version = 2
        startup_kits {{
          active = "active"
          entries {{
            active = "{active_kit}"
            scoped = "{scoped_kit}"
          }}
        }}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
    fake_sess = MagicMock()

    with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_sess) as new_secure:
        returned = new_cli_session_for_args(Namespace(kit_id="scoped"), timeout=2.5, study="default")

    assert returned is fake_sess
    assert new_secure.call_args.kwargs["username"] == "scoped@nvidia.com"
    assert new_secure.call_args.kwargs["startup_kit_location"] == str(scoped_kit)
    assert 'active = "active"' in config_path.read_text(encoding="utf-8")
