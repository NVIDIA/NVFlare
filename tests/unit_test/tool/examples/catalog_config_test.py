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

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from nvflare.app_common.dataset_config.api import parse_study_config
from nvflare.app_common.fdn.catalog.site.rules import parse_rule, parse_settings

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/advanced"


def _site_config(example, site, site_data="/prepared/site/data"):
    """The site's checked-in datasets.yaml, loaded through the dataset configuration API."""
    text = (EXAMPLES / example / "config" / site / "datasets.yaml").read_text().replace("${SITE_DATA}", site_data)
    return parse_study_config(yaml.safe_load(text), "default", f"{example}/{site}")


def _exclusions(example, site):
    rule = parse_rule(_site_config(example, site).get_dataset("omop").usage_rules("catalog"))
    assert rule.share
    return rule.exclude_columns


@pytest.mark.parametrize(
    "example,site,datasets",
    [
        ("federated-catalog", "site-1", {"omop": "parquet", "lab_results": "csv", "visits": "csv"}),
        ("federated-catalog", "site-2", {"omop": "sql"}),
        ("federated-catalog", "site-3", {"omop": "duckdb"}),
        ("federated-catalog-services", "site-1", {"omop": "unity-catalog"}),
        ("federated-catalog-services", "site-2", {"omop": "sql"}),
    ],
)
def test_example_site_configurations_are_valid_and_share_every_dataset(example, site, datasets):
    config = _site_config(example, site)
    assert {name: config.get_dataset(name).type for name in config.list_datasets()} == datasets
    for name in datasets:
        assert parse_rule(config.get_dataset(name).usage_rules("catalog")).share
    settings = parse_settings(config.site_usage("catalog"))
    if datasets.get("omop") == "duckdb":
        # DuckDB is not built in, so the site declares the connector class.
        assert settings.connectors["duckdb"].path == "nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector"
    else:
        assert settings.connectors == {}


def test_the_services_database_credentials_are_secrets_not_part_of_the_uri():
    omop = _site_config("federated-catalog-services", "site-2").get_dataset("omop")
    assert "@" not in omop.uri and omop.secret_keys == ("username", "password")
    compose = yaml.safe_load((EXAMPLES / "federated-catalog-services/compose.yaml").read_text())
    environment = compose["services"]["nvflare-site-2"]["environment"]
    assert {"CATALOG_DB_USER", "CATALOG_DB_PASSWORD"} <= set(environment)
    for site in ("site-1", "site-2"):
        service = compose["services"][f"nvflare-{site}"]
        assert service["environment"]["NVFLARE_DATASET_CONFIG"] == "/site-config/datasets.yaml"
        assert f"./config/{site}/datasets.yaml:/site-config/datasets.yaml:ro" in service["volumes"]


def _load(example, module):
    """Load one example without retaining or replacing sibling modules in the test interpreter."""
    folder = str(EXAMPLES / example)
    absent = object()
    siblings = {name: sys.modules.pop(name, absent) for name in ("constants", "omop_data")}
    original_path = sys.path[:]
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(f"{example}_{module}", EXAMPLES / example / f"{module}.py")
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path[:] = original_path
        for name, previous in siblings.items():
            if previous is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_the_simulation_example_generates_linked_tables_that_match_their_keys():
    prepare = _load("federated-catalog", "prepare_data")
    declared = prepare.table_keys()
    condition_columns = []
    for index, site in enumerate(prepare.SITES, start=1):
        tables = prepare.make_tables(index, prepare.PEOPLE[site], site)
        assert set(tables) == set(declared)
        for table, columns in tables.items():
            assert len({len(values) for values in columns.values()}) == 1
            assert set(declared[table]["primary_key"]) <= set(columns)
            for relation in declared[table]["relationships"]:
                assert set(relation["columns"]) <= set(columns)
                parents = set(tables[relation["target_table"]][relation["target_columns"][0]])
                assert set(columns[relation["columns"][0]]) <= parents
        condition_columns.append(set(tables["condition_occurrence"]))
    # every site has exactly one condition_occurrence column that no other site has
    common = set.intersection(*condition_columns)
    assert [len(columns - common) for columns in condition_columns] == [1, 1, 1]


def test_the_simulation_example_rules_exclude_columns_its_sites_publish():
    # an exclusion that matches nothing only logs a warning, so a typo would publish the column unnoticed
    prepare = _load("federated-catalog", "prepare_data")
    for index, site in enumerate(prepare.SITES, start=1):
        columns = {t: set(c) for t, c in prepare.make_tables(index, prepare.PEOPLE[site], site).items()}
        for table, excluded in _exclusions("federated-catalog", site).items():
            assert table in columns and set(excluded) <= columns[table], (site, table)


def test_preparation_installs_the_configuration_with_absolute_file_uris(tmp_path, monkeypatch):
    prepare = _load("federated-catalog", "prepare_data")
    monkeypatch.setattr(prepare, "SITES", ["site-1", "site-2"])  # site-3 needs the optional duckdb package
    prepare.prepare_data(tmp_path)
    for site in ("site-1", "site-2"):
        installed = tmp_path / site / "local" / "datasets.yaml"
        assert "${SITE_DATA}" not in installed.read_text()
        config = parse_study_config(yaml.safe_load(installed.read_text()), "default", str(installed))
        for name in config.list_datasets():
            path = Path(config.get_dataset(name).path)
            assert path.is_absolute() and path.exists() and path.is_relative_to(tmp_path / site / "data")
    assert not list(tmp_path.rglob("catalog.json")) and not list(tmp_path.rglob("*.catalog.json"))


def test_generated_omop_tables_match_their_declared_schema_and_keys():
    omop = _load("federated-catalog-services", "omop_data")
    for site in omop.SITE_OMOP:
        data, schema, declared = omop.generate(site), omop.schema(site), omop.table_keys(site)
        assert set(data) == set(schema) == set(declared)
        for table, columns in schema.items():
            names = [name for name, _kind, _required in columns]
            assert list(data[table]) == names
            assert len({len(values) for values in data[table].values()}) == 1 and data[table][names[0]]
            assert set(declared[table]["primary_key"]) <= set(names)
            for name, _kind, required in columns:
                assert not required or None not in data[table][name]
            for relation in declared[table]["relationships"]:
                assert set(relation["columns"]) <= set(names)
                assert set(relation["target_columns"]) <= {c for c, _k, _r in schema[relation["target_table"]]}
                parents = set(data[relation["target_table"]][relation["target_columns"][0]])
                assert {v for v in data[table][relation["columns"][0]] if v is not None} <= parents


@pytest.mark.parametrize("folder", ["plain", "hash #x", "percent %41"])
def test_preparation_handles_data_roots_with_uri_special_characters(tmp_path, monkeypatch, folder):
    prepare = _load("federated-catalog", "prepare_data")
    monkeypatch.setattr(prepare, "SITES", ["site-1", "site-2"])
    root = tmp_path / folder
    prepare.prepare_data(root)
    for site in ("site-1", "site-2"):
        installed = root / site / "local" / "datasets.yaml"
        config = parse_study_config(yaml.safe_load(installed.read_text()), "default", str(installed))
        for name in config.list_datasets():
            assert Path(config.get_dataset(name).path).parent == root / site / "data"
            assert Path(config.get_dataset(name).path).exists()


def test_preparation_refuses_a_site_folder_it_did_not_create(tmp_path, monkeypatch):
    prepare = _load("federated-catalog", "prepare_data")
    monkeypatch.setattr(prepare, "SITES", ["site-1"])
    workspace_local = tmp_path / "site-1" / "local"
    workspace_local.mkdir(parents=True)
    (workspace_local / "resources.json").write_text("{}")
    with pytest.raises(SystemExit, match="did not create"):
        prepare.prepare_data(tmp_path)
    assert (workspace_local / "resources.json").read_text() == "{}"
    # --force replaces it anyway.
    prepare.prepare_data(tmp_path, force=True)
    assert (workspace_local / "datasets.yaml").is_file() and not (workspace_local / "resources.json").exists()
    # A folder this example created is replaced on the next run.
    other = tmp_path / "fresh"
    prepare.prepare_data(other)
    prepare.prepare_data(other)
    assert (other / "site-1" / "local" / "datasets.yaml").is_file()


def test_services_rule_exclusions_name_columns_the_site_publishes():
    omop = _load("federated-catalog-services", "omop_data")
    for site in ("site-1", "site-2"):
        columns = {t: {c for c, _k, _r in cols} for t, cols in omop.schema(site).items()}
        for table, excluded in _exclusions("federated-catalog-services", site).items():
            assert table in columns and set(excluded) <= columns[table], (site, table)


def test_example_loader_preserves_existing_sibling_modules(monkeypatch):
    from types import ModuleType

    constants, omop = ModuleType("constants"), ModuleType("omop_data")
    constants.SITES = ["unrelated-site"]
    monkeypatch.setitem(sys.modules, "constants", constants)
    monkeypatch.setitem(sys.modules, "omop_data", omop)
    job = _load("federated-catalog", "job")
    assert job.SITES == ["site-1", "site-2", "site-3"]
    assert sys.modules["constants"] is constants and sys.modules["omop_data"] is omop


def test_service_container_sdk_matches_the_installed_source_revision(monkeypatch, tmp_path):
    from nvflare import _version

    setup = _load("federated-catalog-services", "example_setup")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.delenv("NVFLARE_PACKAGE", raising=False)
    revision = "a" * 40
    monkeypatch.setattr(
        _version, "get_versions", lambda: {"full-revisionid": revision, "dirty": False, "version": "3.1.0.dev4+local"}
    )
    assert setup.installed_sdk_package() == (
        f"nvflare[catalog] @ https://github.com/NVIDIA/NVFlare/archive/{revision}.tar.gz"
    )
    assert setup.installed_sdk_base_version() == "3.1.0"
    monkeypatch.setattr(_version, "get_versions", lambda: {"version": "archive/local-tag"})
    monkeypatch.setattr(setup, "distribution_version", lambda name: "3.2.0.dev1")
    assert setup.installed_sdk_base_version() == "3.2.0"
    monkeypatch.setattr(_version, "get_versions", lambda: {"full-revisionid": None})
    with pytest.raises(ValueError, match="Cannot identify a clean installed NVFlare revision"):
        setup.installed_sdk_package()
    wheel = tmp_path / "custom-sdk.whl"
    wheel.write_bytes(b"controlled wheel bytes")
    monkeypatch.setenv("NVFLARE_PACKAGE", f"{wheel}[catalog]")
    assert setup.installed_sdk_package() == "/opt/sdk/custom-sdk.whl[catalog]"
    assert (tmp_path / ".sdk/custom-sdk.whl").read_bytes() == wheel.read_bytes()


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n"])
def test_service_setup_rejects_multiline_wheel_paths(monkeypatch, tmp_path, newline):
    setup = _load("federated-catalog-services", "example_setup")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setenv("NVFLARE_PACKAGE", f"/host/nvflare-{newline}py3-none-any.whl[catalog]")
    with pytest.raises(ValueError, match="contains a line break"):
        setup.installed_sdk_package()


def test_service_setup_separates_exported_host_input_from_container_settings(monkeypatch, tmp_path):
    import os
    import shutil

    import yaml

    setup = _load("federated-catalog-services", "example_setup")
    example = tmp_path / "example"
    source = EXAMPLES / "federated-catalog-services"
    example.mkdir()
    shutil.copyfile(source / "project.yml", example / "project.yml")
    shutil.copytree(source / "config", example / "config")
    project = yaml.safe_load((example / "project.yml").read_text())
    workspace = tmp_path / "kits"
    kits = workspace / project["name"] / "prod_00"
    for participant in project["participants"]:
        (kits / participant["name"] / "startup").mkdir(parents=True)
    (kits / "nvflare-server/local").mkdir()
    (kits / setup.KIT_SDK_FILE).write_text("/opt/sdk/sdk.whl[catalog]")
    wheel = tmp_path / "sdk.whl"
    wheel.write_bytes(b"controlled SDK wheel")
    host_input = f"{wheel}[catalog]"
    monkeypatch.setenv("NVFLARE_PACKAGE", host_input)
    monkeypatch.setattr(setup, "ROOT", example)
    monkeypatch.setattr(setup, "installed_sdk_base_version", lambda: "3.1.0")
    monkeypatch.setattr(setup, "run_command", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["example_setup.py", "--workspace", str(workspace)])
    setup.main()
    values = dict(line.split("=", 1) for line in (example / ".env").read_text().splitlines())
    assert "NVFLARE_PACKAGE" not in values
    assert json.loads(values["CATALOG_SDK_PACKAGE"]) == "/opt/sdk/sdk.whl[catalog]"
    assert json.loads(values["CATALOG_SDK_BASE_VERSION"]) == "3.1.0"
    assert os.environ["NVFLARE_PACKAGE"] == host_input
    compose = yaml.safe_load((source / "compose.yaml").read_text())
    assert compose["x-flare"]["build"]["args"]["NVFLARE_PACKAGE"].startswith("${CATALOG_SDK_PACKAGE:")


@pytest.mark.parametrize("recorded", [None, "/opt/sdk/older-sdk.whl[catalog]"])
def test_service_setup_refuses_kits_created_for_another_sdk(monkeypatch, tmp_path, recorded):
    import yaml

    setup = _load("federated-catalog-services", "example_setup")
    example = tmp_path / "example"
    example.mkdir()
    (example / "project.yml").write_text((EXAMPLES / "federated-catalog-services/project.yml").read_text())
    project = yaml.safe_load((example / "project.yml").read_text())
    kits = tmp_path / "kits" / project["name"] / "prod_00"
    for participant in project["participants"]:
        (kits / participant["name"] / "startup").mkdir(parents=True)
    if recorded:
        (kits / setup.KIT_SDK_FILE).write_text(recorded)
    wheel = tmp_path / "sdk.whl"
    wheel.write_bytes(b"controlled SDK wheel")
    monkeypatch.setenv("NVFLARE_PACKAGE", f"{wheel}[catalog]")
    monkeypatch.setattr(setup, "ROOT", example)
    commands = []
    monkeypatch.setattr(setup, "run_command", lambda *args: commands.append(args))
    monkeypatch.setattr(sys, "argv", ["example_setup.py", "--workspace", str(tmp_path / "kits")])
    with pytest.raises(ValueError, match="different NVFlare SDK.*--reprovision"):
        setup.main()
    assert commands == [] and not (example / ".env").exists()


def test_service_setup_records_the_sdk_of_new_kits(monkeypatch, tmp_path):
    import yaml

    setup = _load("federated-catalog-services", "example_setup")
    example = tmp_path / "example"
    example.mkdir()
    (example / "project.yml").write_text((EXAMPLES / "federated-catalog-services/project.yml").read_text())
    project = yaml.safe_load((example / "project.yml").read_text())
    workspace = tmp_path / "kits"

    def provision(*args):
        if args[:2] == ("nvflare", "provision"):
            for participant in project["participants"]:
                (workspace / project["name"] / "prod_00" / participant["name"] / "startup").mkdir(parents=True)

    wheel = tmp_path / "sdk.whl"
    wheel.write_bytes(b"controlled SDK wheel")
    monkeypatch.setenv("NVFLARE_PACKAGE", f"{wheel}[catalog]")
    monkeypatch.setattr(setup, "ROOT", example)
    monkeypatch.setattr(setup, "installed_sdk_base_version", lambda: "3.1.0")
    monkeypatch.setattr(setup, "run_command", provision)
    monkeypatch.setattr(sys, "argv", ["example_setup.py", "--workspace", str(workspace)])
    setup.main()
    kits = workspace / project["name"] / "prod_00"
    assert (kits / setup.KIT_SDK_FILE).read_text() == "/opt/sdk/sdk.whl[catalog]"
    setup.main()  # the same SDK reuses them


def test_service_database_readiness_checks_database_and_credentials():
    import shlex

    import yaml

    compose = yaml.safe_load((EXAMPLES / "federated-catalog-services/compose.yaml").read_text())
    database = compose["services"]["postgres"]
    # Keep this demo isolated from older postgres-data volumes with other databases/credentials.
    assert database["volumes"] == ["omop-data:/var/lib/postgresql/data"]
    assert "omop-data" in compose["volumes"]
    kind, command = database["healthcheck"]["test"]
    assert kind == "CMD-SHELL"
    args = shlex.split(command)
    assert args[:2] == ["PGPASSWORD=$${POSTGRES_PASSWORD}", "psql"]
    assert args[args.index("-h") + 1] == "127.0.0.1"
    assert args[args.index("-U") + 1] == "postgres"
    assert args[args.index("-d") + 1] == database["environment"]["POSTGRES_DB"]
    assert args[args.index("-c") + 1] == "SELECT 1"


@pytest.mark.parametrize("fail_insert", [False, True])
def test_postgres_preparation_uses_pg8000_transaction_and_releases_engine(monkeypatch, fail_insert):
    from unittest.mock import MagicMock, Mock

    import sqlalchemy

    prepare = _load("federated-catalog-services", "prepare_data")
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    insertions = []

    def execute(statement, parameters=None):
        sql = str(statement)
        result = Mock()
        result.fetchall.return_value = []  # no existing schema: exercise creation and inserts
        result.fetchone.return_value = None
        if sql.startswith("INSERT"):
            insertions.append(parameters)
            if fail_insert:
                raise RuntimeError("controlled insert failure")
        return result

    connection.execute.side_effect = execute
    create = Mock(return_value=engine)
    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-bootstrap-password")
    monkeypatch.setattr(prepare.omop_data, "generate", lambda site: {"person": {"id": [1, 2]}})
    monkeypatch.setattr(prepare.omop_data, "schema", lambda site: {"person": [("id", "int", True)]})
    monkeypatch.setattr(
        prepare.omop_data,
        "create_statements",
        lambda site, prefix: [("person", "CREATE TABLE omop.person (id INTEGER PRIMARY KEY)")],
    )
    if fail_insert:
        with pytest.raises(RuntimeError, match="controlled insert failure"):
            prepare.prepare_postgres("site-2")
        assert engine.begin.return_value.__exit__.call_args.args[0] is RuntimeError
    else:
        prepare.prepare_postgres("site-2")
        engine.begin.return_value.__exit__.assert_called_once_with(None, None, None)
    assert create.call_args.args[0].drivername == "postgresql+pg8000"
    assert create.call_args.kwargs["connect_args"] == {"timeout": 5}
    assert insertions == [[{"id": 1}, {"id": 2}]]
    engine.dispose.assert_called_once_with()


RESULTS = {
    "study": "default",
    "job_id": "job-1",
    "sites": {
        "site-1": {"status": "OK", "reason_code": None, "unavailable": ["registry"], "skipped": ["big"]},
        "site-2": {"status": "NO_RESULT", "reason_code": None, "unavailable": [], "skipped": []},
    },
    "datasets": [
        {
            "site": "site-1",
            "dataset": "omop",
            "metadata": {"tables": [{"name": "person", "columns": []}], "provenance": {"source": "parquet"}},
        }
    ],
}


@pytest.mark.parametrize("example", ["federated-catalog", "federated-catalog-services"])
def test_result_display_shows_sites_and_datasets(capsys, example):
    _load(example, "job").print_catalog(RESULTS)
    output = capsys.readouterr().out
    assert "site-1: OK, unavailable: registry, skipped: big" in output
    assert "site-2: NO_RESULT" in output
    assert "site-1 / omop [parquet]: person" in output


def test_unity_catalog_seed_repairs_corrupt_parquet_and_publishes_atomically(tmp_path, monkeypatch):
    prepare = _load("federated-catalog-services", "prepare_data")
    monkeypatch.setattr(prepare.omop_data, "generate", lambda site: {"person": {"id": [1, 2]}})
    monkeypatch.setattr(prepare.omop_data, "schema", lambda site: {"person": [("id", "int", True)]})
    monkeypatch.setattr(prepare, "wait_for_unity_catalog", lambda api: None)
    monkeypatch.setattr(prepare, "create_in_unity_catalog", lambda *args: None)
    monkeypatch.setenv("UNITY_CATALOG_URL", "http://controlled-boundary")
    target = tmp_path / "person.parquet"
    target.write_bytes(b"interrupted old file")
    write = prepare.pq.write_table

    def interrupted(table, path):
        path.write_bytes(b"interrupted new file")
        raise OSError("simulated write failure")

    monkeypatch.setattr(prepare.pq, "write_table", interrupted)
    with pytest.raises(OSError, match="simulated"):
        prepare.prepare_unity_catalog("site-1", tmp_path)
    assert target.read_bytes() == b"interrupted old file"
    assert not (tmp_path / "person.parquet.tmp").exists()
    monkeypatch.setattr(prepare.pq, "write_table", write)
    prepare.prepare_unity_catalog("site-1", tmp_path)
    assert prepare.pq.read_metadata(target).num_rows == 2

    def should_not_rewrite(*args):
        raise AssertionError("complete seed was rewritten")

    monkeypatch.setattr(prepare.pq, "write_table", should_not_rewrite)
    prepare.prepare_unity_catalog("site-1", tmp_path)


def test_service_job_reads_the_catalog_and_saves_the_result(monkeypatch, tmp_path, capsys):
    job = _load("federated-catalog-services", "job")
    calls = {}
    monkeypatch.setattr(job, "resolve_startup_kit_dir", lambda: "/kits/admin")
    monkeypatch.setattr(job, "resolve_admin_user_and_dir_from_startup_kit", lambda d: ("admin@nvidia.com", d))
    monkeypatch.setattr(job, "ProdEnv", lambda **kwargs: calls.setdefault("env", kwargs) or kwargs)

    def get_catalog(env, **kwargs):
        calls["read"] = kwargs
        return RESULTS

    monkeypatch.setattr(job, "get_catalog", get_catalog)
    monkeypatch.setattr(sys, "argv", ["job.py", "--output", str(tmp_path / "out.json")])
    job.main()
    assert calls["env"]["startup_kit_location"] == "/kits/admin" and calls["env"]["username"] == "admin@nvidia.com"
    assert calls["read"] == {"sites": ["site-1", "site-2"], "output_path": str(tmp_path / "out.json")}
    assert "site-1 / omop [parquet]: person" in capsys.readouterr().out


def test_service_job_without_an_admin_kit_explains_the_setup_step(monkeypatch, capsys):
    job = _load("federated-catalog-services", "job")

    def missing():
        raise job.StartupKitConfigError("startup kit directory is not specified")

    monkeypatch.setattr(job, "resolve_startup_kit_dir", missing)
    monkeypatch.setattr(sys, "argv", ["job.py"])
    with pytest.raises(SystemExit) as result:
        job.main()
    assert result.value.code == 1
    assert "python example_setup.py" in capsys.readouterr().out


def test_service_job_failure_prints_the_logs_command(monkeypatch, capsys):
    job = _load("federated-catalog-services", "job")
    monkeypatch.setattr(job, "resolve_startup_kit_dir", lambda: "/kits/admin")
    monkeypatch.setattr(job, "resolve_admin_user_and_dir_from_startup_kit", lambda d: ("admin@nvidia.com", d))
    monkeypatch.setattr(job, "ProdEnv", lambda **kwargs: kwargs)

    def get_catalog(env, **kwargs):
        raise job.CatalogReadError("job ended with status FINISHED:ABORTED", "job-123", "FINISHED:ABORTED")

    monkeypatch.setattr(job, "get_catalog", get_catalog)
    monkeypatch.setattr(sys, "argv", ["job.py"])
    with pytest.raises(SystemExit) as result:
        job.main()
    assert result.value.code == 1
    assert "nvflare job logs job-123 --site all --tail 100" in capsys.readouterr().out


def test_unity_catalog_registration_repairs_changed_columns_only(monkeypatch):
    from copy import deepcopy
    from unittest.mock import Mock

    prepare = _load("federated-catalog-services", "prepare_data")
    body = {
        "name": "person",
        "catalog_name": "flare_demo",
        "schema_name": "omop",
        "table_type": "EXTERNAL",
        "data_source_format": "PARQUET",
        "storage_location": "file:///home/unitycatalog/demo-data/person.parquet",
        "columns": [{"name": "id", "type_text": "integer", "type_name": "INT", "position": 0, "nullable": False}],
    }
    existing = deepcopy(body)
    existing["columns"][0]["name"] = "old_id"
    requests = Mock()
    requests.get.return_value.status_code = 200
    requests.get.return_value.json.return_value = existing
    monkeypatch.setattr(prepare, "requests", requests)
    prepare.create_in_unity_catalog("http://controlled-boundary", "tables", body)
    requests.delete.assert_called_once_with("http://controlled-boundary/tables/flare_demo.omop.person", timeout=10)
    requests.post.assert_called_once_with("http://controlled-boundary/tables", json=body, timeout=10)

    requests.reset_mock()
    requests.get.return_value.json.return_value = deepcopy(body)
    prepare.create_in_unity_catalog("http://controlled-boundary", "tables", body)
    requests.delete.assert_not_called()
    requests.post.assert_not_called()

    for key, value in (
        ("table_type", "MANAGED"),
        ("data_source_format", "DELTA"),
        ("storage_location", "file:///another-source/person.parquet"),
    ):
        requests.reset_mock()
        requests.get.return_value.json.return_value = {**existing, key: value}
        with pytest.raises(ValueError, match=f"existing {key} differs"):
            prepare.create_in_unity_catalog("http://controlled-boundary", "tables", body)
        requests.delete.assert_not_called()
        requests.post.assert_not_called()
