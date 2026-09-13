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
import signal
import time
from pathlib import Path

import pytest

from nvflare.tool.examples import examples_cli, store
from tests.unit_test.tool.examples.helpers import COMMIT, EXAMPLE_PATH, VERSION, Response


@pytest.mark.parametrize("command", [[], ["get"], ["cache"], ["cache", "clear"]])
def test_cli_schema_has_no_network_or_mutation(monkeypatch, capsys, command):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--schema"])
    monkeypatch.setattr(store, "ExampleStore", lambda: pytest.fail("schema must not create a cache"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == " ".join(["nvflare", "examples", *command])
    assert schema["mutating"] is True


@pytest.mark.parametrize("command", [["bogus"], ["cache", "bogus"]])
@pytest.mark.parametrize("output_format", ["txt", "json"])
def test_cli_unknown_schema_command_is_rejected(monkeypatch, capsys, command, output_format):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--schema", "--format", output_format])
    monkeypatch.setattr(store, "ExampleStore", lambda: pytest.fail("invalid schema must not create a cache"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 4
    captured = capsys.readouterr()
    if output_format == "json":
        result = json.loads(captured.out)
        assert result["status"] == "error"
        assert result["error_code"] == "INVALID_ARGS"
        assert "command" not in result
    else:
        assert captured.out == ""
        assert "INVALID_ARGS" in captured.err


def test_cli_download_json_and_cache_clear(monkeypatch, capsys, cache, tmp_path):
    from nvflare import cli

    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr(
        "sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(tmp_path / "out"), "--format", "json"]
    )
    cli.run("nvflare")
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["commit"] == COMMIT
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "cache", "clear", "--format", "json"])
    cli.run("nvflare")
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["entries_removed"] == 1
    assert (tmp_path / "out/job.py").is_file()


@pytest.mark.parametrize(
    "failure,code,exit_code",
    [(KeyboardInterrupt(), "EXAMPLE_INTERRUPTED", 130), (OSError("disk full"), "EXAMPLE_IO_ERROR", 1)],
)
def test_cli_failure_is_structured_and_restores_signal_handler(monkeypatch, capsys, cache, failure, code, exit_code):
    from nvflare import cli

    before = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr(cache, "get", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt", "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["error_code"] == code
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize(
    "command",
    [[], ["cache"], ["get"], ["get", "hello-pt", "--source", "https://github.com/NVIDIA/NVFlare/tree/main/examples/a"]],
)
def test_cli_missing_or_unsupported_arguments_fail(monkeypatch, capsys, cache, command):
    from nvflare import cli

    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"


@pytest.mark.parametrize("command", [["get", "hello-pt"], ["cache", "clear"]])
@pytest.mark.parametrize("failure", [OSError("cache unavailable"), RuntimeError("Could not determine home directory.")])
def test_cli_store_initialization_failure_is_structured(monkeypatch, capsys, command, failure):
    from nvflare import cli

    def fail():
        raise failure

    before = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(store, "default_cache_dir", fail)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert result["error_code"] == "EXAMPLE_IO_ERROR"
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize("operation", ["expanduser", "absolute", "resolve"])
def test_cli_destination_resolution_failure_is_structured(monkeypatch, capsys, cache, tmp_path, operation):
    from nvflare import cli

    destination = tmp_path / "out"
    original = getattr(Path, operation)

    def fail_for_destination(path, *args, **kwargs):
        if path == destination:
            raise RuntimeError("Cannot resolve destination")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, fail_for_destination)
    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr(
        "sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(destination), "--format", "json"]
    )
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "EXAMPLE_IO_ERROR"
    assert not destination.exists()
    assert not list(tmp_path.glob(".nvflare-example-*"))


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_cli_deadline_interrupts_trickling_response(monkeypatch, capsys, cache, remote, tmp_path, phase):
    from nvflare import cli

    payload = remote.files[f"{EXAMPLE_PATH}/job.py"]

    class Trickle(Response):
        def iter_content(self, chunk_size):
            for byte in self.data:
                time.sleep(0.01)
                yield bytes([byte])

    def headers():
        time.sleep(0.4)
        return Response(payload)

    previous_handler = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(examples_cli, "DOWNLOAD_TIMEOUT", 0.1, raising=False)
    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    remote.overrides[f"raw/{COMMIT}/{EXAMPLE_PATH}/job.py"] = headers if phase == "headers" else Trickle(payload)
    monkeypatch.setattr(
        "sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(tmp_path / "out"), "--format", "json"]
    )
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "EXAMPLE_TIMEOUT"
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    assert not (tmp_path / "out").exists()
    assert not list(cache.root.glob(".download-*"))
    assert cache._entries() == []
    # Cancellation releases the cache lock as well as staging files.
    assert cache.clear()["entries_removed"] == 0
