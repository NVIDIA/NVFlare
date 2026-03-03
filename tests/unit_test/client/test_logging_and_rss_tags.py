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

"""
Unit tests for Fix 12a and Fix 12b.

Fix 12a: Harden subprocess logging config loading
  - Use ConfigFactory.load_config() so .json/.default variants are both found.
  - Keep only consoleHandler on the root logger to prevent duplicate writes when
    SubprocessLauncher captures subprocess stdout.

Fix 12b: Compact RSS role tags
  - Subprocess (CA): "CA s=<site> r=<round> recv" / "CA s=<site> r=<round> send"
  - Client job (CJ): "CJ s=<site> t=<task> r=<round> relay"

Tests verify:

  Fix 12a (_configure_subprocess_logging):
  1. ConfigFactory.load_config() is called with LOGGING_CONFIG and search_dirs=[local_dir].
  2. When ConfigFactory returns None (no file found), apply_log_config is NOT called.
  3. File handlers are stripped — only consoleHandler is kept in root logger handlers.
  4. apply_log_config IS called when a config is found.
  5. Exception is caught and logged as a warning — never propagated.
  6. Missing workspace_dir returns early without calling ConfigFactory.

  Fix 12b (RSS tag format):
  7.  ex_process receive() emits "CA s=... r=... recv" tag.
  8.  ex_process send() emits "CA s=... r=... send" tag.
  9.  in_process receive() emits "CA s=... r=... recv" tag.
  10. in_process send() emits "CA s=... r=... send" tag.
  11. client_api_launcher_executor check_output_shareable() emits "CJ s=... t=... r=... relay" tag.
  12. Old long-form tags ("after_receive", "after_send", "after_relay", "client_job") absent.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fix 12a helpers
# ---------------------------------------------------------------------------


def _make_client_config(workspace_dir: str = "/fake/workspace"):
    """Return a minimal ClientConfig dict with the pipe workspace_dir set."""
    from nvflare.client.config import ClientConfig, ConfigKey

    return ClientConfig(
        config={
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.PIPE: {
                    ConfigKey.ARG: {"workspace_dir": workspace_dir},
                }
            }
        }
    )


def _make_api():
    """Return an ExProcessClientAPI stub (bypass full init)."""
    from nvflare.client.ex_process.api import ExProcessClientAPI

    api = ExProcessClientAPI.__new__(ExProcessClientAPI)
    api.logger = MagicMock()
    return api


# ---------------------------------------------------------------------------
# Fix 12a: _configure_subprocess_logging()
# ---------------------------------------------------------------------------


class TestConfigureSubprocessLogging:

    def test_uses_config_factory_with_logging_config_constant(self):
        """ConfigFactory.load_config() is called with WorkspaceConstants.LOGGING_CONFIG and local_dir."""
        api = _make_api()
        client_config = _make_client_config("/ws")

        mock_conf = MagicMock()
        mock_conf.to_dict.return_value = {"loggers": {"root": {"handlers": ["consoleHandler"]}}}

        with (
            patch("nvflare.client.ex_process.api.ConfigFactory") as MockCF,
            patch("nvflare.client.ex_process.api.apply_log_config"),
        ):
            MockCF.load_config.return_value = mock_conf
            api._configure_subprocess_logging(client_config)

        import os

        from nvflare.apis.fl_constant import WorkspaceConstants

        expected_local_dir = os.path.join("/ws", "local")
        MockCF.load_config.assert_called_once_with(
            WorkspaceConstants.LOGGING_CONFIG,
            search_dirs=[expected_local_dir],
        )

    def test_no_call_to_apply_when_config_not_found(self):
        """When ConfigFactory returns None (no log config file), apply_log_config must not be called."""
        api = _make_api()
        client_config = _make_client_config("/ws")

        with (
            patch("nvflare.client.ex_process.api.ConfigFactory") as MockCF,
            patch("nvflare.client.ex_process.api.apply_log_config") as mock_apply,
        ):
            MockCF.load_config.return_value = None
            api._configure_subprocess_logging(client_config)

        mock_apply.assert_not_called()

    def test_file_handlers_stripped_keeps_only_console(self):
        """Only consoleHandler is kept; all file handlers are removed from root and named loggers."""
        api = _make_api()
        client_config = _make_client_config("/ws")

        original_handlers = ["consoleHandler", "logFileHandler", "errorFileHandler", "jsonFileHandler", "FLFileHandler"]
        dict_config = {
            "loggers": {
                "root": {"handlers": list(original_handlers)},
                "nvflare": {"handlers": list(original_handlers)},
            }
        }
        mock_conf = MagicMock()
        mock_conf.to_dict.return_value = dict_config

        captured = {}

        def capture_apply(cfg, *args, **kwargs):
            captured["cfg"] = cfg

        with (
            patch("nvflare.client.ex_process.api.ConfigFactory") as MockCF,
            patch("nvflare.client.ex_process.api.apply_log_config", side_effect=capture_apply),
        ):
            MockCF.load_config.return_value = mock_conf
            api._configure_subprocess_logging(client_config)

        assert "cfg" in captured, "apply_log_config must be called"
        kept_root = captured["cfg"]["loggers"]["root"]["handlers"]
        assert kept_root == ["consoleHandler"], f"Only consoleHandler must be kept on root; got {kept_root}"
        kept_named = captured["cfg"]["loggers"]["nvflare"]["handlers"]
        assert kept_named == ["consoleHandler"], f"Only consoleHandler must be kept on named logger; got {kept_named}"

    def test_apply_log_config_called_with_dict_and_workspace(self):
        """apply_log_config() is called with the filtered dict and workspace_dir."""
        api = _make_api()
        client_config = _make_client_config("/my/workspace")

        dict_config = {"loggers": {"root": {"handlers": ["consoleHandler"]}}}
        mock_conf = MagicMock()
        mock_conf.to_dict.return_value = dict_config

        with (
            patch("nvflare.client.ex_process.api.ConfigFactory") as MockCF,
            patch("nvflare.client.ex_process.api.apply_log_config") as mock_apply,
        ):
            MockCF.load_config.return_value = mock_conf
            api._configure_subprocess_logging(client_config)

        mock_apply.assert_called_once()
        args = mock_apply.call_args[0]
        assert args[1] == "/my/workspace", "workspace_dir must be passed to apply_log_config"

    def test_exception_is_caught_and_warned(self):
        """Any exception during logging setup is caught and logged as a warning — never propagated."""
        api = _make_api()
        client_config = _make_client_config("/ws")

        with patch("nvflare.client.ex_process.api.ConfigFactory") as MockCF:
            MockCF.load_config.side_effect = RuntimeError("boom")
            # Must not raise
            api._configure_subprocess_logging(client_config)

        api.logger.warning.assert_called_once()
        assert "boom" in api.logger.warning.call_args[0][0]

    def test_missing_workspace_dir_returns_early(self):
        """When workspace_dir is empty, ConfigFactory must not be called."""
        from nvflare.client.config import ClientConfig, ConfigKey

        api = _make_api()
        # Config with no workspace_dir
        client_config = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.PIPE: {ConfigKey.ARG: {}}}})

        with patch("nvflare.client.ex_process.api.ConfigFactory") as MockCF:
            api._configure_subprocess_logging(client_config)

        MockCF.load_config.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 12b helpers
# ---------------------------------------------------------------------------


def _rss_tags_in_source(module_path: str) -> list:
    """Return all log_rss() tag strings found in the source of the given module."""
    import ast
    import importlib
    import inspect

    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    tags = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "log_rss":
                if node.args:
                    # Grab the first arg as an AST dump (string or f-string)
                    tags.append(ast.dump(node.args[0]))
    return tags


# ---------------------------------------------------------------------------
# Fix 12b: compact RSS role tags
# ---------------------------------------------------------------------------


class TestRssTags:
    """RSS log tag format uses compact role-prefixed identifiers."""

    def _assert_no_old_format(self, tags: list, label: str):
        """Assert none of the old long-form markers appear in the tags."""
        for tag in tags:
            for old in ("after_receive", "after_send", "after_relay", "client_job"):
                assert old not in tag, (
                    f"{label}: old tag marker '{old}' still present in log_rss() call — "
                    f"should be replaced with compact CA/CJ format. Found in: {tag}"
                )

    def test_ex_process_recv_tag_starts_with_ca(self):
        """ex_process receive() must emit a tag starting with 'CA s='."""
        import inspect

        import nvflare.client.ex_process.api as mod

        src = inspect.getsource(mod.ExProcessClientAPI.receive)
        assert "CA s=" in src, "ex_process receive() must use 'CA s=...' RSS tag"

    def test_ex_process_send_tag_starts_with_ca(self):
        """ex_process send() must emit a tag starting with 'CA s='."""
        import inspect

        import nvflare.client.ex_process.api as mod

        src = inspect.getsource(mod.ExProcessClientAPI.send)
        assert "CA s=" in src, "ex_process send() must use 'CA s=...' RSS tag"

    def test_ex_process_recv_tag_contains_recv(self):
        """ex_process receive() tag must contain 'recv' phase token."""
        import inspect

        import nvflare.client.ex_process.api as mod

        src = inspect.getsource(mod.ExProcessClientAPI.receive)
        assert "recv" in src

    def test_ex_process_send_tag_contains_send(self):
        """ex_process send() tag must contain 'send' phase token."""
        import inspect

        import nvflare.client.ex_process.api as mod

        src = inspect.getsource(mod.ExProcessClientAPI.send)
        assert " send" in src or "'send'" in src or '"send"' in src

    def test_in_process_recv_tag_starts_with_ca(self):
        """in_process receive() must emit a tag starting with 'CA s='."""
        import inspect

        import nvflare.client.in_process.api as mod

        # InProcessClientAPI.receive is the public method
        src = inspect.getsource(mod.InProcessClientAPI.receive)
        assert "CA s=" in src, "in_process receive() must use 'CA s=...' RSS tag"

    def test_in_process_send_tag_starts_with_ca(self):
        """in_process send() must emit a tag starting with 'CA s='."""
        import inspect

        import nvflare.client.in_process.api as mod

        src = inspect.getsource(mod.InProcessClientAPI.send)
        assert "CA s=" in src, "in_process send() must use 'CA s=...' RSS tag"

    def test_cj_relay_tag_starts_with_cj(self):
        """client_api_launcher_executor check_output_shareable() must emit a 'CJ s=...' RSS tag."""
        import inspect

        import nvflare.app_common.executors.client_api_launcher_executor as mod

        src = inspect.getsource(mod.ClientAPILauncherExecutor.check_output_shareable)
        assert "CJ s=" in src, "check_output_shareable() must use 'CJ s=...' RSS tag"

    def test_cj_relay_tag_contains_relay(self):
        """client_api_launcher_executor RSS tag must contain 'relay' phase token."""
        import inspect

        import nvflare.app_common.executors.client_api_launcher_executor as mod

        src = inspect.getsource(mod.ClientAPILauncherExecutor.check_output_shareable)
        assert "relay" in src

    def test_no_old_format_in_ex_process(self):
        """Old long-form markers must not appear in ex_process RSS tags."""
        import inspect

        import nvflare.client.ex_process.api as mod

        src = inspect.getsource(mod.ExProcessClientAPI)
        for old in ("after_receive", "after_send", "client_job"):
            # Only check within log_rss calls, not in comments
            import re

            for m in re.finditer(r"log_rss\(([^)]+)\)", src):
                assert old not in m.group(1), f"Old RSS marker '{old}' still in ex_process log_rss call: {m.group(0)}"

    def test_no_old_format_in_cj_relay(self):
        """Old long-form markers must not appear in CJ relay RSS tag."""
        import inspect
        import re

        import nvflare.app_common.executors.client_api_launcher_executor as mod

        src = inspect.getsource(mod.ClientAPILauncherExecutor.check_output_shareable)
        for m in re.finditer(r"log_rss\(([^)]+)\)", src):
            for old in ("after_relay", "client_job"):
                assert old not in m.group(1), f"Old RSS marker '{old}' still in CJ relay log_rss call: {m.group(0)}"
