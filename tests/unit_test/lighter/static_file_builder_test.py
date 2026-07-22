# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

import pytest

from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl.static_file import StaticFileBuilder


class _FakeCtx:
    def __init__(self, project, root_dir):
        self._project = project
        self._root_dir = root_dir
        self.calls = []

    def get_project(self):
        return self._project

    def get(self, key):
        return {
            CtxKey.ADMIN_PORT: 8002,
            CtxKey.FED_LEARN_PORT: 8003,
        }.get(key)

    def get_kit_dir(self, entity):
        return self._root_dir / entity.name / "startup"

    def get_local_dir(self, entity):
        return self._root_dir / entity.name / "local"

    def get_ws_dir(self, entity):
        return self._root_dir / entity.name

    def build_from_template(
        self, dest_dir, temp_section, file_name, replacement=None, mode="t", exe=False, content_modify_cb=None, **kwargs
    ):
        self.calls.append((str(dest_dir), file_name))


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _load_master_template():
    import yaml

    template_path = _repo_root() / "nvflare" / "lighter" / "templates" / "master_template.yml"
    assert template_path.exists()
    with open(template_path, "r") as f:
        return yaml.safe_load(f)


def _extract_class_allow_list(resource_template):
    """Extract the class_allow_list JSON array verbatim.

    Parses the embedded JSON array so trailing-dot package prefixes (which
    the previous regex-based extractor silently dropped) are included.
    """
    resource_template = resource_template.replace(
        "{~~class_allow_list~~}", json.dumps(DEFAULT_CLASS_ALLOW_LIST, indent=2)
    )
    list_pos = resource_template.index('"class_allow_list"')
    start = resource_template.index("[", list_pos)
    depth = 0
    end = None
    for i in range(start, len(resource_template)):
        if resource_template[i] == "[":
            depth += 1
        elif resource_template[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    return json.loads(resource_template[start : end + 1])


class TestStaticFileBuilder:
    @pytest.mark.parametrize(
        "scheme",
        [("grpc"), ("http"), ("tcp")],
    )
    def test_scheme(self, scheme):
        builder = StaticFileBuilder(scheme=scheme)
        assert builder.scheme == scheme

    def test_build_server_emits_study_registry_when_studies_exist(self, tmp_path):
        server = Participant(type="server", name="server1", org="org")
        project = Project(
            name="proj",
            description="desc",
            participants=[server],
            props={
                "api_version": 4,
                "studies": {
                    "study_a": {
                        "site_orgs": {"org": ["client1"]},
                        "admins": ["admin1@org.com"],
                    }
                },
            },
        )
        ctx = _FakeCtx(project=project, root_dir=tmp_path)

        builder = StaticFileBuilder()
        builder._build_server(project.get_server(), ctx)

        registry_path = tmp_path / server.name / "local" / "study_registry.json"
        assert registry_path.exists()
        assert json.loads(registry_path.read_text()) == {
            "format_version": "1.0",
            "studies": project.get_prop("studies"),
        }

    def test_build_server_does_not_emit_study_registry_without_studies(self, tmp_path):
        server = Participant(type="server", name="server1", org="org")
        project = Project(
            name="proj",
            description="desc",
            participants=[server],
            props={
                "api_version": 4,
            },
        )
        ctx = _FakeCtx(project=project, root_dir=tmp_path)

        builder = StaticFileBuilder()
        builder._build_server(project.get_server(), ctx)

        registry_path = tmp_path / server.name / "local" / "study_registry.json"
        assert not registry_path.exists()

    def test_build_renders_shared_default_component_policy(self, tmp_path):
        server = Participant(type="server", name="server", org="org")
        client = Participant(type="client", name="site-1", org="org")
        project = Project(
            name="proj",
            description="desc",
            participants=[server, client],
            props={"api_version": 4},
        )
        ctx = ProvisionContext(str(tmp_path), project)
        ctx.load_templates("master_template.yml")
        for participant in (server, client):
            Path(ctx.get_kit_dir(participant)).mkdir(parents=True)
            Path(ctx.get_local_dir(participant)).mkdir(parents=True)

        StaticFileBuilder().build(project, ctx)

        for participant in (server, client):
            resources_file = Path(ctx.get_local_dir(participant)) / "resources.json.default"
            resources = json.loads(resources_file.read_text())
            assert resources["class_allow_list"] == list(DEFAULT_CLASS_ALLOW_LIST)

    def test_auth_identity_config_omits_default_identity_fields(self):
        builder = StaticFileBuilder()

        assert builder._build_auth_identity_config(auth_identity="site-1", default_identity="site-1") == ""

    def test_auth_identity_config_emits_custom_identity_fields_as_valid_json(self):
        builder = StaticFileBuilder()

        fragment = builder._build_auth_identity_config(
            auth_identity="custom-site-cn",
            default_identity="site-1",
            auth_identity_map={"site-2": "custom-site-2-cn"},
            indent=6,
        )
        config_text = "\n".join(
            [
                "{",
                '  "client": {',
                '      "connection_security": "mtls"' + fragment,
                "  }",
                "}",
            ]
        )
        config = json.loads(config_text)

        assert config["client"]["auth_identity"] == "custom-site-cn"
        assert config["client"]["auth_identity_map"] == {"site-2": "custom-site-2-cn"}

    def test_master_template_moves_user_config_runtime_workspace(self):
        """CC startup kits live in plaintext /user_config, so runtime artifacts must not default there."""
        template = _load_master_template()

        sub_start = template["sub_start_sh"]
        stop_fl = template["stop_fl_sh"]
        copy_cmd = 'cp -a "$SOURCE_WORKSPACE" "$KIT_WORKSPACE"'
        verify_cmd = 'verify_startup_kits -f "$KIT_WORKSPACE" -c "$KIT_WORKSPACE/startup/rootCA.pem"'

        assert 'SOURCE_WORKSPACE" == "/user_config"' in sub_start
        assert 'SOURCE_WORKSPACE" == /user_config/*' in sub_start
        assert 'SOURCE_WORKSPACE" == "/user_config"' in stop_fl
        assert 'SOURCE_WORKSPACE" == /user_config/*' in stop_fl
        assert 'WORKSPACE="/vault/workspace"' in sub_start
        assert 'WORKSPACE="/vault/workspace/$(basename "$SOURCE_WORKSPACE")"' in sub_start
        assert 'WORKSPACE="/vault/workspace"' in stop_fl
        assert 'WORKSPACE="/vault/workspace/$(basename "$SOURCE_WORKSPACE")"' in stop_fl
        assert 'KIT_WORKSPACE="$WORKSPACE/kit"' in sub_start
        assert copy_cmd in sub_start
        assert 'rm -rf "$KIT_WORKSPACE"' in sub_start
        assert 'rm -rf "$WORKSPACE"' not in sub_start
        assert 'ln -s "$KIT_WORKSPACE/startup" "$WORKSPACE/startup"' in sub_start
        assert 'ln -s "$KIT_WORKSPACE/local" "$WORKSPACE/local"' in sub_start
        assert '-m "$WORKSPACE"' in sub_start
        assert verify_cmd in sub_start
        assert sub_start.index(copy_cmd) < sub_start.index(verify_cmd)
        assert sub_start.index(verify_cmd) < sub_start.index('mkdir -p "$WORKSPACE/transfer"')
        assert 'touch "$WORKSPACE/shutdown.fl"' in stop_fl

    def test_master_template_class_allow_list_is_exact(self):
        """The provisioned allow list must match the curated component list exactly."""
        template = _load_master_template()

        expected_paths = list(DEFAULT_CLASS_ALLOW_LIST)
        for resource_key in ("local_client_resources", "local_server_resources"):
            resource_template = template[resource_key]
            assert "{~~class_allow_list~~}" in resource_template
            assert _extract_class_allow_list(resource_template) == expected_paths
            assert '"class_list_enforcement_mode": "enforce"' in resource_template

    def test_master_template_class_allow_list_has_no_package_prefixes(self):
        """Package prefixes (entries ending in '.') broaden authorization to every class under that package.

        Future maintainers who add a broad prefix must enumerate the specific classes instead, or
        explicitly review-and-approve the prefix here. This guard prevents the previously-removed
        ``nvflare.edge.`` style entry from silently coming back via an expected_paths update.
        """
        template = _load_master_template()

        # If a future PR genuinely needs a broad package prefix, add it here with an explanation.
        explicitly_reviewed_package_prefixes: set = set()

        for resource_key in ("local_client_resources", "local_server_resources"):
            extracted = _extract_class_allow_list(template[resource_key])
            package_prefixes = {p for p in extracted if p.endswith(".")}
            unreviewed = package_prefixes - explicitly_reviewed_package_prefixes
            assert not unreviewed, (
                f"unreviewed package prefixes in {resource_key}: {sorted(unreviewed)}. "
                "Package prefixes authorize every class (current and future) under the package, "
                "which broadens the security posture of the allow_list. Either enumerate the "
                "specific classes needed, or add the prefix to "
                "explicitly_reviewed_package_prefixes with an in-test explanation."
            )

    def test_master_template_class_allow_list_excludes_edge_components(self):
        """Provisioned non-BYOC resources must not authorize edge components."""
        template = _load_master_template()

        for resource_key in ("local_client_resources", "local_server_resources"):
            extracted = _extract_class_allow_list(template[resource_key])
            edge_paths = [p for p in extracted if p.startswith("nvflare.edge.")]
            assert not edge_paths, f"edge classes should not be in {resource_key}: {edge_paths}"

    def test_master_template_default_authz_grants_submission_byoc_to_lead_and_project_admin(self):
        """Default authorization grants broad project_admin permissions and lead BYOC submission permission."""
        import yaml

        template = _load_master_template()

        default_authz = yaml.safe_load(template["default_authz"])
        permissions = default_authz["permissions"]

        assert permissions["project_admin"] == "any"
        assert permissions["lead"]["byoc"] == "any"

    def test_master_template_allows_non_edge_regression_components(self):
        """Pin the specific paths that regressed in 2.8.0rc4 (PR #4701 fallout).

        Non-BYOC jobs that loaded these built-in classes worked on 2.8.0rc3
        and broke on rc4 because they were missing from the provisioned
        class_allow_list. Edge components are intentionally outside this
        non-BYOC regression set.
        """
        from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer

        template = _load_master_template()

        regression_paths = [
            # PR #4701 regression: NPTrainer was omitted from the curated list.
            "nvflare.app_common.np.np_trainer.NPTrainer",
            # PR #4701 regression: ccwf re-export short paths were not matched
            # because the list only stored the full module path. Configs in
            # the wild reference the package-level alias.
            "nvflare.app_common.ccwf.CyclicServerController",
            "nvflare.app_common.ccwf.SwarmClientController",
            "nvflare.app_common.ccwf.CrossSiteEvalServerController",
        ]
        for resource_key in ("local_client_resources", "local_server_resources"):
            allow_list = _extract_class_allow_list(template[resource_key])
            for path in regression_paths:
                assert any(
                    ComponentPathAuthorizer._path_matches_prefix(path, prefix) for prefix in allow_list
                ), f"{path!r} would be rejected by ComponentPathAuthorizer against {resource_key}"

    def test_master_template_class_allow_list_excludes_code_exec_sinks(self):
        """Classes that deserialize or execute file content must never be allow-listed.

        The non-BYOC allow-list only restricts which component classes can be instantiated; it
        does not restrict the files a job ships to a site (a job's ``config/`` folder is deployed
        to the server and every client without BYOC). So any allow-listed class that loads a
        config-controlled file through an unsafe loader (pickle / ``torch.load`` / keras
        ``load_model`` / ``importlib``) is a remote-code-execution sink. The sinks below are
        intentionally OMITTED -- the corresponding ``*_locator`` components are listed instead.
        Do not add these classes (or a package prefix covering them) to ``class_allow_list``.
        """
        from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer

        template = _load_master_template()

        forbidden_paths = [
            # tf.keras.models.load_model executes code embedded in .keras/.h5/SavedModel files
            # (Lambda layers / custom objects); there is no safe-load flag.
            "nvflare.app_opt.tf.model_persistor.TFModelPersistor",
            # torch.load is pickle-based and runs arbitrary code when load_weights_only=False.
            "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor",
            # ConfigParser importlib-imports a class path read from a config file (plus
            # sys.path.append) -> arbitrary code import.
            "nvflare.edge.simulation.config.ConfigParser",
        ]

        for resource_key in ("local_client_resources", "local_server_resources"):
            allow_list = _extract_class_allow_list(template[resource_key])
            for forbidden in forbidden_paths:
                # Use the authorizer's own prefix matcher so the guard mirrors production
                # authorization semantics exactly (the private method is used intentionally).
                authorized_by = [
                    entry for entry in allow_list if ComponentPathAuthorizer._path_matches_prefix(forbidden, entry)
                ]
                assert not authorized_by, (
                    f"{forbidden!r} is a file-deserialization / code-execution sink and must not be "
                    f"authorized by {resource_key} (matched by {authorized_by}). Do not add this class, or a "
                    "package prefix covering it, to class_allow_list; list the corresponding *_locator "
                    "component instead."
                )
