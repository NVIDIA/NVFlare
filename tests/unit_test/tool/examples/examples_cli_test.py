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
from pathlib import Path

import pytest
import requests

from nvflare.tool.cli_output import set_output_format
from nvflare.tool.examples import examples_cli

REVISION = "a" * 40
VERSION = {"version": "2.10.0", "full-revisionid": REVISION, "dirty": False, "error": None}
CATALOG = examples_cli.EXAMPLE_CATALOG
SOURCE_PATH = "examples/hello-world/hello-pt"


@pytest.fixture(autouse=True)
def reset_output_mode():
    set_output_format("txt")
    yield
    set_output_format("txt")


def _mock_download(monkeypatch, *, requirements="nvflare[PT]~=2.9.0rc\ntorch\n", readme="README.md"):
    def download(revision, source_path, destination):
        assert revision == REVISION
        assert source_path == SOURCE_PATH
        (destination / readme).write_text("# Example\n")
        (destination / "job.py").write_text("print('example')\n")
        (destination / "nested").mkdir()
        (destination / "nested/client.py").write_text("# client\n")
        if requirements is not None:
            (destination / "requirements.txt").write_text(requirements)
        return "https://api.github.com/tree"

    monkeypatch.setattr(examples_cli, "_download_example", download)


def test_get_records_downloaded_source_and_requirements(monkeypatch, tmp_path):
    _mock_download(monkeypatch)
    destination = tmp_path / "hello-pt"

    result = examples_cli.get_example(VERSION, name="hello-pt", destination=destination)

    assert (destination / "job.py").read_text() == "print('example')\n"
    assert (destination / "nested/client.py").read_text() == "# client\n"
    assert (destination / "requirements.txt").read_text() == "torch\n"
    assert result["setup_commands"] == [["pip", "install", "-r", "requirements.txt"]]
    assert result["readme"] == str(destination / "README.md")
    assert result["source_url"] == f"https://github.com/NVIDIA/NVFlare/tree/{REVISION}/{SOURCE_PATH}"
    provenance = json.loads((destination / examples_cli.PROVENANCE_FILE).read_text())
    assert provenance["revision"] == REVISION
    assert provenance["example"] == "hello-pt"
    assert provenance["source_path"] == SOURCE_PATH
    assert provenance["nvflare_requirement_removed"] is True


def test_missing_requirements_becomes_empty_file(monkeypatch, tmp_path):
    _mock_download(monkeypatch, requirements=None)

    examples_cli.get_example(VERSION, name="hello-pt", destination=tmp_path / "hello-pt")

    assert (tmp_path / "hello-pt/requirements.txt").read_text() == ""


def test_only_nvflare_distribution_requirements_are_removed(tmp_path):
    destination = tmp_path / "example"
    destination.mkdir()
    requirements = destination / "requirements.txt"
    requirements.write_text(
        "NVFlare~=2.7.2rc\n"
        "nvflare[HE]>=2.7.2rc  # installed before retrieval\n"
        "nvflare-helper==1.0\n"
        "https://example.com/nvflare.whl\n"
    )

    changed = examples_cli._prepare_requirements(destination)

    assert changed is True
    assert requirements.read_text() == "nvflare-helper==1.0\nhttps://example.com/nvflare.whl\n"


def test_rst_readme_is_reported(monkeypatch, tmp_path):
    _mock_download(monkeypatch, readme="README.rst")

    result = examples_cli.get_example(VERSION, name="hello-pt", destination=tmp_path / "hello-pt")

    assert result["readme"].endswith("README.rst")


def test_download_fetches_only_files_below_catalog_path(monkeypatch, tmp_path):
    tree = {
        "truncated": False,
        "tree": [
            {"path": f"{SOURCE_PATH}/README.md", "type": "blob", "size": 10, "mode": "100644"},
            {"path": f"{SOURCE_PATH}/nested/run.sh", "type": "blob", "size": 4, "mode": "100755"},
            {"path": "examples/other/secret.txt", "type": "blob", "size": 6, "mode": "100644"},
        ],
    }
    payloads = {"README.md": b"# Example\n", "run.sh": b"run\n"}

    class Response:
        def __init__(self, *, metadata=None, data=None):
            self.metadata = metadata
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def json(self):
            return self.metadata

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            yield self.data

    class Session:
        def __init__(self):
            self.requested = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            self.requested.append(url)
            if "api.github.com" in url:
                return Response(metadata=tree)
            return Response(data=payloads[Path(url).name])

    session = Session()
    monkeypatch.setattr(requests, "Session", lambda: session)
    destination = tmp_path / "example"
    destination.mkdir()

    tree_url = examples_cli._download_example(REVISION, SOURCE_PATH, destination)

    assert tree_url.endswith(f"/{REVISION}?recursive=1")
    assert (destination / "README.md").read_bytes() == payloads["README.md"]
    assert (destination / "nested/run.sh").read_bytes() == payloads["run.sh"]
    assert (destination / "nested/run.sh").stat().st_mode & 0o111
    assert len(session.requested) == 3
    assert all("secret.txt" not in url for url in session.requested)


@pytest.mark.parametrize("tree", [{"truncated": True, "tree": []}, {"truncated": False, "tree": []}])
def test_invalid_or_missing_tree_is_rejected(monkeypatch, tmp_path, tree):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def json(self):
            return tree

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(requests, "Session", Session)

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._download_example(REVISION, SOURCE_PATH, tmp_path)

    expected = "EXAMPLE_CONTENT_INVALID" if tree["truncated"] else "EXAMPLE_SOURCE_NOT_FOUND"
    assert error.value.code == expected


def test_network_failure_is_structured(monkeypatch, tmp_path):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "Session", Session)

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._download_example(REVISION, SOURCE_PATH, tmp_path)

    assert error.value.code == "EXAMPLE_NETWORK_ERROR"


@pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
def test_existing_destination_is_rejected_before_download(monkeypatch, tmp_path, kind):
    destination = tmp_path / "hello-pt"
    if kind == "file":
        destination.write_text("original")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "absent")
    monkeypatch.setattr(examples_cli, "_download_example", lambda *args: pytest.fail("downloaded"))

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(VERSION, name="hello-pt", destination=destination)

    assert error.value.code == "EXAMPLE_DESTINATION_EXISTS"


def test_unknown_example_and_unknown_revision_are_structured(tmp_path):
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(VERSION, name="unknown", destination=tmp_path / "unknown")
    assert error.value.code == "EXAMPLE_UNKNOWN"

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(
            {**VERSION, "full-revisionid": None}, name="hello-pt", destination=tmp_path / "hello-pt"
        )
    assert error.value.code == "EXAMPLE_VERSION_UNKNOWN"


@pytest.mark.parametrize("command", [[], ["list"], ["get"]])
def test_cli_schema_does_not_download(monkeypatch, capsys, command):
    from nvflare import cli

    monkeypatch.setattr(examples_cli, "get_example", lambda *args, **kwargs: pytest.fail("downloaded"))
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--schema"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["streaming"] is False
    assert schema["output_modes"] == ["json"]
    assert schema["retry_token"] == {"supported": False}
    if command == ["list"]:
        assert schema["command"] == "nvflare examples list"
        assert schema["mutating"] is False
        assert schema["idempotent"] is True
    if command == ["get"]:
        assert schema["command"] == "nvflare examples get"
        assert schema["mutating"] is True
        assert schema["idempotent"] is False
        name_arg = next(argument for argument in schema["args"] if argument["name"] == "name")
        assert name_arg["choices"] == sorted(CATALOG)


def test_list_prints_short_names_and_source_paths(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "list"])
    cli.run("nvflare")

    output = capsys.readouterr().out
    assert "SHORT NAME" in output
    assert "hello-pt" in output
    assert "examples/hello-world/hello-pt" in output
    assert "collab-pt" in output
    assert "examples/advanced/collab/pt_cifar10" in output


def test_list_json_is_machine_readable(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "list", "--format", "json"])
    cli.run("nvflare")

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert {entry["name"]: entry["source_path"] for entry in result["data"]["examples"]} == {
        name: entry["source_path"] for name, entry in CATALOG.items()
    }


def test_unknown_subcommand_with_schema_is_rejected(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "--format", "json", "examples", "bogus", "--schema"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")

    assert error.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"


def test_human_output_points_to_requirements_and_readme(monkeypatch, capsys, tmp_path):
    from nvflare import cli

    _mock_download(monkeypatch)
    destination = tmp_path / "copy with ' quote"
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(destination)])

    cli.run("nvflare")

    output = capsys.readouterr().out
    assert f"Downloaded example: {destination}" in output
    assert f"  cd {shlex.quote(str(destination))}" in output
    assert "  pip install -r requirements.txt" in output
    assert "Follow README.md for preparation and run instructions." in output
    assert "python job.py" not in output


def test_get_json_is_machine_readable(monkeypatch, capsys, tmp_path):
    from nvflare import cli

    _mock_download(monkeypatch)
    destination = tmp_path / "hello-pt"
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr(
        "sys.argv", ["nvflare", "--format", "json", "examples", "get", "hello-pt", "--dest", str(destination)]
    )

    cli.run("nvflare")

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["example"] == "hello-pt"
    assert result["data"]["directory"] == str(destination)
    assert result["data"]["setup_commands"] == [["pip", "install", "-r", "requirements.txt"]]
    assert result["data"]["readme"] == str(destination / "README.md")


@pytest.mark.parametrize(
    "failure,code,exit_code",
    [
        (KeyboardInterrupt(), "EXAMPLE_INTERRUPTED", 130),
        (OSError("disk full"), "EXAMPLE_IO_ERROR", 1),
        (
            examples_cli.ExampleError("EXAMPLE_NETWORK_ERROR", "offline", "retry"),
            "EXAMPLE_NETWORK_ERROR",
            1,
        ),
    ],
)
def test_cli_failure_is_structured(monkeypatch, capsys, failure, code, exit_code):
    from nvflare import cli

    monkeypatch.setattr(examples_cli, "get_example", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt", "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["error_code"] == code
