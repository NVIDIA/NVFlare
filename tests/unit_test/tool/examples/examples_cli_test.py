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
from nvflare.tool.examples.catalog import load_catalog

REVISION = "a" * 40
VERSION = {"version": "2.10.0", "full-revisionid": REVISION, "dirty": False, "error": None}
CATALOG, CATALOG_ERRORS = load_catalog()
SOURCE_PATH = "examples/hello-world/hello-pt"
assert not CATALOG_ERRORS


@pytest.fixture(autouse=True)
def reset_output_mode():
    set_output_format("txt")
    yield
    set_output_format("txt")


class _Response:
    def __init__(self, *, status_code=200, metadata=None, data=b""):
        self.status_code = status_code
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


class _Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, url, **kwargs):
        self.requested.append(url)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _mock_session(monkeypatch, *responses):
    session = _Session(responses)
    monkeypatch.setattr(requests, "Session", lambda: session)
    return session


def _mock_download(
    monkeypatch,
    *,
    source_path=SOURCE_PATH,
    requirements="nvflare[PT]~=2.9.0rc\ntorch\n",
    nested_requirements=None,
    readme="README.md",
    readme_contents="# Example\n",
):
    def download(revision, requested_source_path, destination, destination_path=None):
        assert revision == REVISION
        assert requested_source_path == source_path
        destination.mkdir()
        content_directory = destination / destination_path if destination_path else destination
        if destination_path:
            content_directory.mkdir(parents=True)
        if readme is not None:
            (content_directory / readme).write_text(readme_contents)
        (content_directory / "job.py").write_text("print('example')\n")
        (content_directory / "nested").mkdir()
        (content_directory / "nested/client.py").write_text("# client\n")
        if nested_requirements is not None:
            (content_directory / "nested/requirements.txt").write_text(nested_requirements)
        if requirements is not None:
            (content_directory / "requirements.txt").write_text(requirements)

    monkeypatch.setattr(examples_cli, "_download_example", download)


def test_get_records_downloaded_source_and_requirements(monkeypatch, tmp_path):
    _mock_download(monkeypatch)
    destination = tmp_path / "hello-pt"

    result = examples_cli.get_example(VERSION, CATALOG, name="hello-pt", destination=destination)

    assert (destination / "job.py").read_text() == "print('example')\n"
    assert (destination / "nested/client.py").read_text() == "# client\n"
    assert (destination / "requirements.txt").read_text() == "nvflare[PT]~=2.9.0rc\ntorch\n"
    assert result["readme"] == str(destination / "README.md")
    assert result["source_url"] == f"https://github.com/NVIDIA/NVFlare/tree/{REVISION}/{SOURCE_PATH}"
    assert result["warnings"] == [
        {
            "code": "EXAMPLE_DEPENDENCY_GUIDANCE",
            "message": "Preserve the installed NVFlare distribution when setting up this example.",
            "hint": (
                "Skip any README or dependency-file instruction that installs nvflare or nvflare-nightly. "
                "Add required extras to the same stable, nightly, or editable distribution, then install only "
                "the remaining dependencies."
            ),
        }
    ]
    provenance = json.loads((destination / examples_cli.PROVENANCE_FILE).read_text())
    assert provenance["revision"] == REVISION
    assert provenance["example"] == "hello-pt"
    assert provenance["source_path"] == SOURCE_PATH


def test_missing_requirements_remains_missing(monkeypatch, tmp_path):
    _mock_download(monkeypatch, requirements=None)

    result = examples_cli.get_example(VERSION, CATALOG, name="hello-pt", destination=tmp_path / "hello-pt")

    assert not (tmp_path / "hello-pt/requirements.txt").exists()
    assert result["warnings"][0]["code"] == "EXAMPLE_DEPENDENCY_GUIDANCE"


def test_nested_requirement_is_not_modified(monkeypatch, tmp_path):
    requirement = "nvflare_nightly[HE] \\\n    >=2.10.0rc\nnvflare-helper==1.0\n"
    _mock_download(monkeypatch, requirements="nvflare[PT] (>=2.10)\ntorch\n", nested_requirements=requirement)

    result = examples_cli.get_example(VERSION, CATALOG, name="hello-pt", destination=tmp_path / "hello-pt")

    assert (tmp_path / "hello-pt/nested/requirements.txt").read_text() == requirement
    assert "paths" not in result["warnings"][0]


def test_rst_readme_is_reported(monkeypatch, tmp_path):
    _mock_download(monkeypatch, readme="README.rst")

    result = examples_cli.get_example(VERSION, CATALOG, name="hello-pt", destination=tmp_path / "hello-pt")

    assert result["readme"].endswith("README.rst")


def test_missing_root_readme_warns_without_failing(monkeypatch, tmp_path):
    _mock_download(monkeypatch, requirements=None, readme=None)

    result = examples_cli.get_example(VERSION, CATALOG, name="hello-pt", destination=tmp_path / "hello-pt")

    assert result["readme"] is None
    assert result["warnings"] == [
        {
            "code": "EXAMPLE_DEPENDENCY_GUIDANCE",
            "message": "Preserve the installed NVFlare distribution when setting up this example.",
            "hint": (
                "Skip any README or dependency-file instruction that installs nvflare or nvflare-nightly. "
                "Add required extras to the same stable, nightly, or editable distribution, then install only "
                "the remaining dependencies."
            ),
        },
        {
            "code": "EXAMPLE_README_MISSING",
            "message": "The downloaded example does not contain a root README.",
            "hint": "Inspect the downloaded files for dependency, preparation, and run instructions.",
        },
    ]
    assert (tmp_path / "hello-pt/.nvflare-example.json").is_file()


def test_download_fetches_path_scoped_tree(monkeypatch, tmp_path):
    tree = {
        "truncated": False,
        "tree": [
            {"path": "README.md", "type": "blob", "size": 10, "mode": "100644"},
            {"path": "nested/run.sh", "type": "blob", "size": 4, "mode": "100755"},
        ],
    }
    payloads = {"README.md": b"# Example\n", "run.sh": b"run\n"}
    session = _mock_session(
        monkeypatch,
        _Response(metadata=tree),
        _Response(data=payloads["README.md"]),
        _Response(data=payloads["run.sh"]),
    )
    destination = tmp_path / "example"

    examples_cli._download_example(REVISION, SOURCE_PATH, destination)

    assert session.requested[0].endswith(f"/{REVISION}:{SOURCE_PATH}?recursive=1")
    assert (destination / "README.md").read_bytes() == payloads["README.md"]
    assert (destination / "nested/run.sh").read_bytes() == payloads["run.sh"]
    assert (destination / "nested/run.sh").stat().st_mode & 0o111
    assert len(session.requested) == 3
    assert session.requested[1].endswith(f"/{REVISION}/{SOURCE_PATH}/README.md")
    assert session.requested[2].endswith(f"/{REVISION}/{SOURCE_PATH}/nested/run.sh")


def test_download_preserves_catalog_destination_path(monkeypatch, tmp_path):
    tree = {
        "truncated": False,
        "tree": [
            {"path": "README.md", "type": "blob", "mode": "100644"},
            {"path": "fedavg/job.py", "type": "blob", "mode": "100644"},
        ],
    }
    _mock_session(monkeypatch, _Response(metadata=tree), _Response(data=b"# Collab\n"), _Response(data=b"# job\n"))
    destination = tmp_path / "collab-pt"

    examples_cli._download_example(REVISION, "examples/advanced/collab/pt_cifar10", destination, "collab/pt_cifar10")

    assert (destination / "collab/pt_cifar10/README.md").read_text() == "# Collab\n"
    assert (destination / "collab/pt_cifar10/fedavg/job.py").read_text() == "# job\n"
    assert not (destination / "README.md").exists()


def test_get_reports_nested_readme_for_package_layout(monkeypatch, tmp_path):
    source_path = "examples/advanced/collab/pt_cifar10"
    _mock_download(monkeypatch, source_path=source_path)
    destination = tmp_path / "collab-pt"

    result = examples_cli.get_example(VERSION, CATALOG, name="collab-pt", destination=destination)

    assert result["destination_path"] == "collab/pt_cifar10"
    assert result["readme"] == str(destination / "collab/pt_cifar10/README.md")
    provenance = json.loads((destination / examples_cli.PROVENANCE_FILE).read_text())
    assert provenance["destination_path"] == "collab/pt_cifar10"


@pytest.mark.parametrize("fail_after_tree", [False, True])
def test_network_failure_is_structured(monkeypatch, tmp_path, fail_after_tree):
    tree_response = _Response(
        metadata={"truncated": False, "tree": [{"path": "README.md", "type": "blob", "mode": "100644"}]}
    )
    responses = (
        (tree_response, requests.ConnectionError("offline"))
        if fail_after_tree
        else (requests.ConnectionError("offline"),)
    )
    _mock_session(monkeypatch, *responses)

    destination = tmp_path / "example"
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._download_example(REVISION, SOURCE_PATH, destination)

    assert error.value.code == "EXAMPLE_NETWORK_ERROR"
    assert ("Remove the incomplete destination" in error.value.hint) is fail_after_tree
    assert (str(destination) in error.value.hint) is fail_after_tree
    assert destination.exists() is fail_after_tree


def test_missing_path_is_not_reported_as_network_failure(monkeypatch, tmp_path):
    _mock_session(monkeypatch, _Response(status_code=404))

    destination = tmp_path / "example"
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._download_example(REVISION, SOURCE_PATH, destination)

    assert error.value.code == "EXAMPLE_SOURCE_NOT_FOUND"
    assert "source revision or catalog path" in str(error.value)
    assert "editable install" in error.value.hint
    assert not destination.exists()


def test_missing_tree_key_is_structured_without_creating_destination(monkeypatch, tmp_path):
    _mock_session(monkeypatch, _Response(metadata={}))
    destination = tmp_path / "example"

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._download_example(REVISION, SOURCE_PATH, destination)

    assert error.value.code == "EXAMPLE_CONTENT_INVALID"
    assert not destination.exists()


@pytest.mark.parametrize(
    "metadata",
    [
        {"truncated": True, "tree": [{"path": "README.md", "type": "blob"}]},
        {"truncated": False, "tree": []},
        {"truncated": False, "tree": [None]},
        {"truncated": False, "tree": [{}]},
        {"truncated": False, "tree": [{"path": "README.md"}]},
        {"truncated": False, "tree": [{"path": 1, "type": "blob"}]},
        {
            "truncated": False,
            "tree": [
                {"path": "README.md", "type": "blob", "mode": "100644"},
                {"path": "linked-config.yaml", "type": "blob", "mode": "120000"},
            ],
        },
        {
            "truncated": False,
            "tree": [
                {"path": "README.md", "type": "blob", "mode": "100644"},
                {"path": "vendor/project", "type": "commit", "mode": "160000"},
            ],
        },
        {
            "truncated": False,
            "tree": [
                {"path": "README.md", "type": "blob", "mode": "100644"},
                {"path": "readme.md", "type": "blob", "mode": "100644"},
            ],
        },
        {
            "truncated": False,
            "tree": [
                {"path": "Foo", "type": "blob", "mode": "100644"},
                {"path": "foo/bar.txt", "type": "blob", "mode": "100644"},
            ],
        },
        {
            "truncated": False,
            "tree": [
                {"path": "Dir/a.txt", "type": "blob", "mode": "100644"},
                {"path": "dir/b.txt", "type": "blob", "mode": "100644"},
            ],
        },
        {
            "truncated": False,
            "tree": [
                {"path": "\u0390.txt", "type": "blob", "mode": "100644"},
                {"path": "\u03aa\u0301.txt", "type": "blob", "mode": "100644"},
            ],
        },
        {"truncated": False, "tree": [{"path": "folder\\file.txt", "type": "blob", "mode": "100644"}]},
        {"truncated": False, "tree": [{"path": "C:/file.txt", "type": "blob", "mode": "100644"}]},
    ],
)
def test_unusable_tree_metadata_is_rejected_before_creating_destination(monkeypatch, tmp_path, metadata):
    _mock_session(monkeypatch, _Response(metadata=metadata))
    destination = tmp_path / "example"

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._download_example(REVISION, SOURCE_PATH, destination)

    assert error.value.code == "EXAMPLE_CONTENT_INVALID"
    assert not destination.exists()


@pytest.mark.parametrize(
    "entry",
    [
        {"path": "linked-config.yaml", "type": "blob", "mode": 120000},
        {"path": "vendor/project", "type": "commit", "mode": "160000"},
    ],
)
def test_symlinks_and_submodules_report_the_specific_content_error(monkeypatch, tmp_path, entry):
    _mock_session(monkeypatch, _Response(metadata={"truncated": False, "tree": [entry]}))
    destination = tmp_path / "example"

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._download_example(REVISION, SOURCE_PATH, destination)

    assert error.value.code == "EXAMPLE_CONTENT_INVALID"
    assert "unsupported symlink or submodule" in str(error.value)
    assert not destination.exists()


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
        examples_cli.get_example(VERSION, CATALOG, name="hello-pt", destination=destination)

    assert error.value.code == "EXAMPLE_DESTINATION_EXISTS"


def test_unknown_example_and_unknown_revision_are_structured(tmp_path):
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(VERSION, CATALOG, name="unknown", destination=tmp_path / "unknown")
    assert error.value.code == "EXAMPLE_UNKNOWN"

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(
            {**VERSION, "full-revisionid": None}, CATALOG, name="hello-pt", destination=tmp_path / "hello-pt"
        )
    assert error.value.code == "EXAMPLE_VERSION_UNKNOWN"


def test_dirty_editable_revision_is_rejected(tmp_path):
    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli.get_example(
            {**VERSION, "dirty": True}, CATALOG, name="hello-pt", destination=tmp_path / "hello-pt"
        )

    assert error.value.code == "EXAMPLE_VERSION_DIRTY"
    assert "uncommitted changes" in str(error.value)
    assert not (tmp_path / "hello-pt").exists()


def test_example_revision_reads_download_provenance(tmp_path):
    provenance = tmp_path / examples_cli.PROVENANCE_FILE
    provenance.write_text(json.dumps({"revision": REVISION}), encoding="utf-8")

    assert examples_cli._example_revision(tmp_path) == {
        "revision": REVISION,
        "provenance_file": str(provenance),
    }


def test_revision_command_prints_download_provenance(monkeypatch, tmp_path, capsys):
    from nvflare import cli

    (tmp_path / examples_cli.PROVENANCE_FILE).write_text(json.dumps({"revision": REVISION}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "revision"])

    cli.run("nvflare")

    assert capsys.readouterr().out == f"{REVISION}\n"


@pytest.mark.parametrize("contents", ["not json", "{}", '{"revision": "main"}'])
def test_example_revision_rejects_invalid_provenance(tmp_path, contents):
    (tmp_path / examples_cli.PROVENANCE_FILE).write_text(contents, encoding="utf-8")

    with pytest.raises(examples_cli.ExampleError) as error:
        examples_cli._example_revision(tmp_path)

    assert error.value.code == "EXAMPLE_PROVENANCE_INVALID"


@pytest.mark.parametrize("command", [[], ["list"], ["get"], ["revision"]])
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
        assert "choices" not in name_arg
    if command == ["revision"]:
        assert schema["command"] == "nvflare examples revision"
        assert schema["mutating"] is False
        assert schema["idempotent"] is True


def test_list_prints_short_names_and_source_paths(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "list"])
    cli.run("nvflare")

    output = capsys.readouterr().out
    assert "HELLO WORLD" in output
    assert "ADVANCED" in output
    assert (
        output.index("ADVANCED")
        < output.index("AGENT SKILLS")
        < output.index("DEPLOYMENT")
        < output.index("HELLO WORLD")
    )
    assert "SHORT NAME" in output
    assert "hello-pt" in output
    assert "examples/hello-world/hello-pt" in output
    assert "cifar10-pt" in output
    assert "examples/advanced/cifar10/pt" in output
    assert "experiment-tracking" in output
    assert "examples/advanced/experiment-tracking" in output
    assert "tracking-tensorboard" not in output
    assert "skill-pytorch-conversion" in output
    assert "examples/hello-world/agent-skills/pytorch-conversion" in output


def test_list_json_is_machine_readable(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "list", "--format", "json"])
    cli.run("nvflare")

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    listed = result["data"]["examples"]
    assert len(listed) == len(CATALOG)
    assert [(entry["category"], entry["name"]) for entry in listed] == sorted(
        (entry["category"], entry["name"]) for entry in listed
    )
    assert {entry["name"]: (entry["category"], entry["source_path"]) for entry in listed} == {
        name: (entry["category"], entry["source_path"]) for name, entry in CATALOG.items()
    }
    assert result["data"]["catalog_errors"] == []


def test_list_keeps_valid_catalog_entries(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr(
        examples_cli,
        "load_catalog",
        lambda: (
            {"hello-pt": {"category": "hello-world", "source_path": SOURCE_PATH}},
            [{"name": "broken", "error": "bad path"}],
        ),
    )
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "list", "--format", "json"])
    cli.run("nvflare")

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["examples"] == [{"name": "hello-pt", "category": "hello-world", "source_path": SOURCE_PATH}]
    assert result["data"]["catalog_errors"] == [{"name": "broken", "error": "bad path"}]


def test_get_rejected_catalog_entry_reports_catalog_error(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr(
        examples_cli,
        "load_catalog",
        lambda: (
            {"hello-pt": {"category": "hello-world", "source_path": SOURCE_PATH}},
            [{"name": "broken", "error": "bad path"}],
        ),
    )
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "broken", "--format", "json"])

    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")

    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "EXAMPLE_CATALOG_INVALID"
    assert "bad path" in result["message"]


def test_catalog_failure_is_scoped_to_examples_command(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr(examples_cli, "load_catalog", lambda: (_ for _ in ()).throw(FileNotFoundError("missing")))
    monkeypatch.setattr("sys.argv", ["nvflare", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    assert "examples" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "list", "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "EXAMPLE_CATALOG_INVALID"


def test_unknown_subcommand_with_schema_is_rejected(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "--format", "json", "examples", "bogus", "--schema"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")

    assert error.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"


def test_human_output_points_to_readme(monkeypatch, capsys, tmp_path):
    from nvflare import cli

    _mock_download(monkeypatch)
    destination = tmp_path / "copy with ' quote"
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(destination)])

    cli.run("nvflare")

    output = capsys.readouterr().out
    assert f"Downloaded example: {destination}" in output
    assert f"  cd {shlex.quote(str(destination))}" in output
    assert "pip install -r requirements.txt" not in output
    assert "Warning: Preserve the installed NVFlare distribution when setting up this example." in output
    assert "same stable, nightly, or editable distribution" in output
    assert "Follow README.md for dependency, preparation, and run instructions." in output
    assert "python job.py" not in output


def test_human_output_points_to_nested_readme(monkeypatch, capsys, tmp_path):
    from nvflare import cli

    _mock_download(monkeypatch, source_path="examples/advanced/collab/pt_cifar10")
    destination = tmp_path / "collab-pt"
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "collab-pt", "--dest", str(destination)])

    cli.run("nvflare")

    assert (
        "Follow collab/pt_cifar10/README.md for dependency, preparation, and run instructions."
        in capsys.readouterr().out
    )


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
    assert "setup_commands" not in result["data"]
    assert result["data"]["readme"] == str(destination / "README.md")
    assert result["data"]["warnings"][0]["code"] == "EXAMPLE_DEPENDENCY_GUIDANCE"
    assert "paths" not in result["data"]["warnings"][0]
    assert "tree_url" not in result["data"]


def test_unknown_example_uses_structured_cli_error(monkeypatch, capsys):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "unknown", "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")

    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "EXAMPLE_UNKNOWN"
    assert "examples list" in result["hint"]


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
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == code
    if code in {"EXAMPLE_INTERRUPTED", "EXAMPLE_IO_ERROR"}:
        assert str(Path.cwd() / "hello-pt") in result["hint"]
