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
import shlex
import shutil
from pathlib import Path

import pytest

from nvflare.tool.cli_output import set_output_format
from nvflare.tool.examples import examples_cli

VERSION = {"version": "2.10.0", "full-revisionid": "a" * 40, "dirty": False, "error": None}
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "examples/hello-world/hello-pt"


@pytest.fixture(autouse=True)
def reset_output_mode():
    set_output_format("txt")
    yield
    set_output_format("txt")


def _files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("command", [[], ["get"]])
def test_cli_schema_does_not_copy(monkeypatch, capsys, command):
    from nvflare import cli

    monkeypatch.setattr(examples_cli, "get_example", lambda *args, **kwargs: pytest.fail("schema copied files"))
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--schema"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == " ".join(["nvflare", "examples", *command])
    assert schema["mutating"] is True


def test_cli_copies_exact_canonical_example_and_records_version(monkeypatch, capsys, tmp_path):
    from nvflare import cli

    destination = tmp_path / "copied"
    original = _files(SOURCE)
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr(
        "sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(destination), "--format", "json"]
    )
    cli.run("nvflare")

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ok"
    assert output["data"]["source"] == "bundled"
    assert output["data"]["nvflare_version"] == VERSION["version"]
    assert output["data"]["next_command"] == ["python", "job.py"]
    delivered = _files(destination)
    provenance = json.loads(delivered.pop(examples_cli.PROVENANCE_FILE))
    assert delivered == original
    assert provenance == {
        "schema_version": 1,
        "example": "hello-pt",
        "source": "bundled",
        "source_path": "examples/hello-world/hello-pt",
        "nvflare_version": VERSION["version"],
    }
    assert _files(SOURCE) == original


def test_copy_uses_example_allowlist(monkeypatch, tmp_path):
    source = tmp_path / "source"
    shutil.copytree(SOURCE, source)
    (source / "__pycache__").mkdir()
    (source / "__pycache__/client.cpython-313.pyc").write_bytes(b"generated")
    (source / "credentials.txt").write_text("local")
    destination = tmp_path / "copied"
    monkeypatch.setattr(examples_cli, "_example_source", lambda name: source)

    examples_cli.get_example(VERSION, name="hello-pt", destination=destination)

    delivered = _files(destination)
    delivered.pop(examples_cli.PROVENANCE_FILE)
    assert delivered == _files(SOURCE)


def test_default_destination_and_human_next_step(monkeypatch, capsys, tmp_path):
    from nvflare import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt"])
    cli.run("nvflare")
    output = capsys.readouterr().out
    assert f"Created bundled example: {tmp_path / 'hello-pt'}" in output
    assert "python job.py" in output
    assert (tmp_path / "hello-pt/job.py").is_file()


def test_human_next_step_quotes_destination(monkeypatch, capsys, tmp_path):
    from nvflare import cli

    destination = tmp_path / "copy with ' quote"
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(destination)])
    cli.run("nvflare")
    assert f"  cd {shlex.quote(str(destination))}" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["file", "directory", "nonempty", "dangling-symlink"])
def test_existing_destination_is_untouched(tmp_path, kind):
    destination = tmp_path / "copied"
    if kind == "file":
        destination.write_text("original")
    elif kind == "dangling-symlink":
        destination.symlink_to(tmp_path / "absent")
    else:
        destination.mkdir()
        if kind == "nonempty":
            (destination / "original").write_text("original")
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(VERSION, name="hello-pt", destination=destination)
    assert error.value.code == "EXAMPLE_DESTINATION_EXISTS"
    if kind == "file":
        assert destination.read_text() == "original"
    elif kind == "nonempty":
        assert (destination / "original").read_text() == "original"
    else:
        assert destination.exists() or destination.is_symlink()


def test_destination_created_during_copy_is_untouched(monkeypatch, tmp_path):
    destination = tmp_path / "copied"

    def race(name):
        destination.mkdir()
        (destination / "original").write_text("original")
        return SOURCE

    monkeypatch.setattr(examples_cli, "_example_source", race)
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(VERSION, name="hello-pt", destination=destination)
    assert error.value.code == "EXAMPLE_DESTINATION_EXISTS"
    assert (destination / "original").read_text() == "original"


def test_publish_does_not_replace_empty_destination_created_after_check(monkeypatch, tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new").write_text("new")
    destination = tmp_path / "copied"
    rename_noreplace = examples_cli._rename_noreplace

    def race(source, target):
        target.mkdir()
        rename_noreplace(source, target)

    monkeypatch.setattr(examples_cli, "_rename_noreplace", race)

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._publish(staged, destination)

    assert error.value.code == "EXAMPLE_DESTINATION_EXISTS"
    assert staged.is_dir()
    assert not list(destination.iterdir())


@pytest.mark.parametrize("failure", [OSError("disk full"), KeyboardInterrupt()])
def test_failed_copy_never_exposes_destination(monkeypatch, tmp_path, failure):
    destination = tmp_path / "copied"

    def fail(source, target, filenames):
        assert not destination.exists()
        assert target.name == "example"
        (target / "partial").write_text("partial")
        raise failure

    monkeypatch.setattr(examples_cli, "_copy_resource", fail)
    with pytest.raises(type(failure)):
        examples_cli.get_example(VERSION, name="hello-pt", destination=destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".nvflare-example-*"))


def test_missing_parent_and_unknown_example_are_structured(tmp_path):
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(VERSION, name="hello-pt", destination=tmp_path / "missing/out")
    assert error.value.code == "EXAMPLE_DESTINATION_INVALID"
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(VERSION, name="unknown", destination=tmp_path / "out")
    assert error.value.code == "EXAMPLE_UNKNOWN"


@pytest.mark.parametrize(
    "failure,code,exit_code",
    [(KeyboardInterrupt(), "EXAMPLE_INTERRUPTED", 130), (OSError("disk full"), "EXAMPLE_IO_ERROR", 1)],
)
def test_cli_failure_is_structured(monkeypatch, capsys, failure, code, exit_code):
    from nvflare import cli

    monkeypatch.setattr(examples_cli, "get_example", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt", "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["error_code"] == code


@pytest.mark.parametrize("command", [[], ["get"], ["cache", "clear"], ["get", "hello-pt", "--ref", "main"]])
def test_missing_or_removed_commands_fail(monkeypatch, capsys, command):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"
