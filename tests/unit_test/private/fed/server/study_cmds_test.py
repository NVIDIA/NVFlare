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

from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import MagicMock, patch

from nvflare.apis.client import Client, ClientPropKey
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server.study_cmds import StudyCommandModule
from nvflare.security.study_registry import StudyRegistry

_EMPTY_REGISTRY = {"format_version": "1.0", "studies": {}}

_REGISTRY_WITH_STUDY = {
    "format_version": "1.0",
    "studies": {
        "study1": {
            "site_orgs": {"org_a": ["site-existing"]},
            "admins": ["admin@example.com"],
        }
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConnection:
    def __init__(self, role, org, engine=None, user="admin@example.com"):
        self._props = {
            ConnProps.USER_NAME: user,
            ConnProps.USER_ROLE: role,
            ConnProps.USER_ORG: org,
        }
        self.app_ctx = engine
        self.replies = []

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def append_dict(self, data, meta=None):
        self.replies.append(data)

    @property
    def last_reply(self):
        return self.replies[-1] if self.replies else None


def _make_client(org: str) -> Client:
    client = MagicMock(spec=Client)
    client.get_prop.side_effect = lambda key, default="": org if key == ClientPropKey.ORG else default
    return client


def _make_engine(site_map: dict):
    """
    site_map: {site_name: org_str}
    Omitting a site name → disconnected (get_client_from_name returns None).
    """
    engine = MagicMock()

    def _get_client(name):
        if name not in site_map:
            return None
        return _make_client(site_map[name])

    engine.client_manager.get_client_from_name.side_effect = _get_client
    return engine


@contextmanager
def _mutation_ctx(initial_config=None):
    """Patches all I/O and locking so _with_mutation runs without disk access."""
    if initial_config is None:
        initial_config = _EMPTY_REGISTRY
    with (
        patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.acquire_lock", return_value=True),
        patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.release_lock"),
        patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.initialize"),
        # Make isinstance(engine, ServerEngine) pass for MagicMock engines
        patch("nvflare.private.fed.server.study_cmds.ServerEngine", MagicMock),
        patch.object(StudyCommandModule, "_registry_path", return_value="/fake/path"),
        patch.object(StudyCommandModule, "_load_registry_config", side_effect=lambda _: deepcopy(initial_config)),
        patch.object(StudyCommandModule, "_write_registry_config"),
    ):
        yield


# ---------------------------------------------------------------------------
# Section 1: _validate_sites_for_org (direct unit tests)
# ---------------------------------------------------------------------------


class TestValidateSitesForOrg:
    def test_all_valid_returns_empty(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_a"})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a", "site-b"], "org_a")
        assert result == []

    def test_wrong_org_is_rejected(self):
        engine = _make_engine({"site-a": "org_b"})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_a")
        assert result == ["site-a"]

    def test_disconnected_site_is_rejected(self):
        engine = _make_engine({})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-unknown"], "org_a")
        assert result == ["site-unknown"]

    def test_empty_org_on_client_is_rejected(self):
        engine = _make_engine({"site-a": ""})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_a")
        assert result == ["site-a"]

    def test_empty_sites_list_returns_empty(self):
        engine = _make_engine({})
        result = StudyCommandModule._validate_sites_for_org(engine, [], "org_a")
        assert result == []

    def test_mixed_returns_only_bad_sites(self):
        engine = _make_engine(
            {
                "site-ok": "org_a",  # valid
                "site-wrong": "org_b",  # wrong org
                # site-offline → disconnected
            }
        )
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-ok", "site-wrong", "site-offline"], "org_a")
        assert set(result) == {"site-wrong", "site-offline"}

    def test_multiple_orgs_validated_independently(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_b"})
        assert StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_a") == []
        assert StudyCommandModule._validate_sites_for_org(engine, ["site-b"], "org_b") == []
        assert StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_b") == ["site-a"]

    def test_empty_expected_org_rejects_all_sites(self):
        # Empty caller-cert org must never pass — reject every site regardless of what
        # the site cert carries, including a site that also has an empty org.
        engine = _make_engine({"site-a": ""})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "")
        assert result == ["site-a"]

    def test_synthetic_admin_client_name_is_rejected(self):
        # get_client_from_name can return a synthetic Client for admin-style names
        # (e.g. "admin@example.com"). That client has no ORG prop, so it must be
        # rejected when a real org is expected.
        engine = _make_engine({"admin@example.com": ""})
        result = StudyCommandModule._validate_sites_for_org(engine, ["admin@example.com"], "org_a")
        assert result == ["admin@example.com"]


# ---------------------------------------------------------------------------
# Section 2: cmd_register_study — site->org validation
# ---------------------------------------------------------------------------


class TestRegisterStudySiteOrgValidation:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_valid_connected_site_succeeds(self):
        engine = _make_engine({"site-a": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply is not None
        assert "error_code" not in conn.last_reply
        assert conn.last_reply.get("name") == "study1"

    def test_org_admin_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_org_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-offline"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_valid_site_org_succeeds(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_a"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_a:site-a,site-b"])
        assert "error_code" not in conn.last_reply

    def test_project_admin_site_with_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_a:site-a,site-b"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_a"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(
                conn, ["register_study", "study1", "--site-org", "org_a:site-a,site-offline"]
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_multiple_site_org_groups_all_valid_succeeds(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(
                conn,
                ["register_study", "study1", "--site-org", "org_a:site-a", "--site-org", "org_b:site-b"],
            )
        assert "error_code" not in conn.last_reply

    def test_project_admin_multiple_groups_one_bad_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(
                conn,
                ["register_study", "study1", "--site-org", "org_a:site-a", "--site-org", "org_b:site-b"],
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"


# ---------------------------------------------------------------------------
# Section 3: cmd_add_study_site — site->org validation
# ---------------------------------------------------------------------------


class TestAddStudySiteOrgValidation:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_valid_site_succeeds(self):
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-new"])
        assert "error_code" not in conn.last_reply
        assert "site-new" in conn.last_reply.get("added", [])

    def test_org_admin_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-new"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_org_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-offline"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_valid_site_org_succeeds(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_b:site-new"])
        assert "error_code" not in conn.last_reply

    def test_project_admin_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-new": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_b:site-new"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_b:site-offline"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_mixed_valid_and_invalid_returns_invalid_site(self):
        engine = _make_engine({"site-ok": "org_b", "site-bad": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(
                conn, ["add_study_site", "study1", "--site-org", "org_b:site-ok,site-bad"]
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"


# ---------------------------------------------------------------------------
# Section 4: cmd_remove_study_site — site->org validation
# ---------------------------------------------------------------------------


class TestRemoveStudySiteOrgValidation:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_valid_site_succeeds(self):
        engine = _make_engine({"site-existing": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_org_admin_site_in_different_engine_org_still_succeeds(self):
        # Engine reports site-existing under org_b, but the study registry has it under org_a.
        # For removal the registry is the source of truth; engine org is irrelevant.
        engine = _make_engine({"site-existing": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_org_admin_disconnected_site_succeeds(self):
        # Site is offline (not in engine) but is enrolled in the study — removal must still work.
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_project_admin_valid_site_org_succeeds(self):
        engine = _make_engine({"site-existing": "org_a"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn, ["remove_study_site", "study1", "--site-org", "org_a:site-existing"]
            )
        assert "error_code" not in conn.last_reply

    def test_project_admin_site_in_different_engine_org_still_succeeds(self):
        # Engine reports site-existing under org_b, but admin requests removal from org_a.
        # Study registry has it under org_a; engine org is irrelevant for removal.
        engine = _make_engine({"site-existing": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn, ["remove_study_site", "study1", "--site-org", "org_a:site-existing"]
            )
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_project_admin_disconnected_site_succeeds(self):
        # Site is offline but enrolled — removal must succeed.
        engine = _make_engine({})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn, ["remove_study_site", "study1", "--site-org", "org_a:site-existing"]
            )
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_project_admin_mixed_sites_enrolled_and_not_enrolled(self):
        # site-existing is enrolled under org_a; site-b is specified under org_b but not enrolled.
        # Enrolled site is removed; unenrolled site lands in not_enrolled (no error).
        engine = _make_engine({"site-existing": "org_a", "site-b": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn,
                [
                    "remove_study_site",
                    "study1",
                    "--site-org",
                    "org_a:site-existing",
                    "--site-org",
                    "org_b:site-b",
                ],
            )
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])
        assert "site-b" in conn.last_reply.get("not_enrolled", [])

    def test_unenrolled_org_does_not_get_phantom_registry_entry(self):
        # org_b has no sites in study1. Requesting removal of org_b:site-b must not
        # create a phantom {"org_b": []} entry that would grant org_b visibility.
        engine = _make_engine({})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        written = {}

        def capture_write(_path, config):
            written.update(config)

        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            import nvflare.private.fed.server.study_cmds as sc_mod

            with __import__("unittest.mock", fromlist=["patch"]).patch.object(
                sc_mod.StudyCommandModule, "_write_registry_config", side_effect=capture_write
            ):
                self._module().cmd_remove_study_site(
                    conn, ["remove_study_site", "study1", "--site-org", "org_b:site-b"]
                )
        assert "org_b" not in written.get("studies", {}).get("study1", {}).get("site_orgs", {})


# ---------------------------------------------------------------------------
# Section 5: INVALID_ARGS — input-shape enforcement (server-side authoritative)
# ---------------------------------------------------------------------------


class TestInvalidArgsInputShape:
    """
    Verifies that the server rejects the three forbidden input shapes with
    INVALID_ARGS regardless of which site-mutation command is used.
    Rules under test:
      1. mixed --sites + --site-org
      2. org_admin using --site-org
      3. project_admin using --sites
    """

    def _module(self):
        return StudyCommandModule()

    # --- register_study ---

    def test_register_mixed_sites_and_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_register_study(
            conn, ["register_study", "study1", "--sites", "site-a", "--site-org", "org_a:site-b"]
        )
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_register_org_admin_with_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_a:site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_register_project_admin_with_sites_returns_invalid_args(self):
        conn = _FakeConnection(role="project_admin", org="project")
        self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    # --- add_study_site ---

    def test_add_site_mixed_sites_and_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_add_study_site(
            conn, ["add_study_site", "study1", "--sites", "site-a", "--site-org", "org_a:site-b"]
        )
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_add_site_org_admin_with_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_a:site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_add_site_project_admin_with_sites_returns_invalid_args(self):
        conn = _FakeConnection(role="project_admin", org="project")
        self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    # --- remove_study_site ---

    def test_remove_site_mixed_sites_and_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_remove_study_site(
            conn, ["remove_study_site", "study1", "--sites", "site-a", "--site-org", "org_a:site-b"]
        )
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_remove_site_org_admin_with_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--site-org", "org_a:site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_remove_site_project_admin_with_sites_returns_invalid_args(self):
        conn = _FakeConnection(role="project_admin", org="project")
        self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"


# ---------------------------------------------------------------------------
# Section 6: STUDY_ALREADY_EXISTS — register when org not enrolled
# ---------------------------------------------------------------------------


class TestStudyAlreadyExists:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_register_existing_study_with_no_enrollment_returns_already_exists(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_b", engine=engine)
        # study1 exists but org_b is not in site_orgs
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-new"])
        assert conn.last_reply["error_code"] == "STUDY_ALREADY_EXISTS"

    def test_org_admin_register_existing_study_already_enrolled_merges(self):
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        # org_a is already in study1 site_orgs — register should merge not reject
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-new"])
        assert "error_code" not in conn.last_reply
        assert "site-new" in conn.last_reply.get("site_orgs", {}).get("org_a", [])

    def test_project_admin_register_existing_study_succeeds(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_b:site-new"])
        assert "error_code" not in conn.last_reply


# ---------------------------------------------------------------------------
# Section 7: cmd_remove_study
# ---------------------------------------------------------------------------


class TestRemoveStudy:
    def _module(self):
        return StudyCommandModule()

    def _engine_no_jobs(self):
        engine = MagicMock()
        engine.job_def_manager = MagicMock(spec=JobDefManagerSpec)
        engine.job_def_manager.get_all_jobs.return_value = []
        engine.new_context.return_value.__enter__ = MagicMock(return_value=MagicMock())
        engine.new_context.return_value.__exit__ = MagicMock(return_value=False)
        return engine

    def _engine_with_jobs(self, study_name):
        engine = self._engine_no_jobs()
        job = MagicMock()
        job.meta = {JobMetaKey.STUDY.value: study_name}
        engine.job_def_manager.get_all_jobs.return_value = [job]
        return engine

    def test_project_admin_removes_existing_study(self):
        engine = self._engine_no_jobs()
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study(conn, ["remove_study", "study1"])
        assert conn.last_reply.get("removed") is True

    def test_remove_nonexistent_study_returns_not_found(self):
        engine = self._engine_no_jobs()
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_EMPTY_REGISTRY):
            self._module().cmd_remove_study(conn, ["remove_study", "ghost"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"

    def test_remove_study_with_associated_jobs_returns_study_has_jobs(self):
        engine = self._engine_with_jobs("study1")
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study(conn, ["remove_study", "study1"])
        assert conn.last_reply["error_code"] == "STUDY_HAS_JOBS"


# ---------------------------------------------------------------------------
# Section 8: cmd_list_studies — visibility filtering
# ---------------------------------------------------------------------------


def _make_registry(studies_dict):
    config = {
        "format_version": "1.0",
        "studies": {
            name: {"site_orgs": def_["site_orgs"], "admins": def_.get("admins", [])}
            for name, def_ in studies_dict.items()
        },
    }
    return StudyRegistry(config)


class TestListStudiesVisibility:
    def _module(self):
        return StudyCommandModule()

    def test_project_admin_sees_all_studies(self):
        registry = _make_registry(
            {
                "study-alpha": {"site_orgs": {"org_a": ["site-a"]}},
                "study-beta": {"site_orgs": {"org_b": ["site-b"]}},
            }
        )
        conn = _FakeConnection(role="project_admin", org="project")
        with patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.get_registry", return_value=registry):
            self._module().cmd_list_studies(conn, ["list_studies"])
        assert set(conn.last_reply["studies"]) == {"study-alpha", "study-beta"}

    def test_org_admin_sees_only_enrolled_studies(self):
        registry = _make_registry(
            {
                "study-alpha": {"site_orgs": {"org_a": ["site-a"]}},
                "study-beta": {"site_orgs": {"org_b": ["site-b"]}},
            }
        )
        conn = _FakeConnection(role="org_admin", org="org_a")
        with patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.get_registry", return_value=registry):
            self._module().cmd_list_studies(conn, ["list_studies"])
        assert conn.last_reply["studies"] == ["study-alpha"]

    def test_org_admin_with_no_enrollment_sees_empty_list(self):
        registry = _make_registry({"study-alpha": {"site_orgs": {"org_b": ["site-b"]}}})
        conn = _FakeConnection(role="org_admin", org="org_a")
        with patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.get_registry", return_value=registry):
            self._module().cmd_list_studies(conn, ["list_studies"])
        assert conn.last_reply["studies"] == []


# ---------------------------------------------------------------------------
# Section 9: cmd_show_study
# ---------------------------------------------------------------------------


class TestShowStudy:
    def _module(self):
        return StudyCommandModule()

    def test_project_admin_can_show_any_study(self):
        registry = _make_registry({"study1": {"site_orgs": {"org_a": ["site-a"]}, "admins": ["u@x.com"]}})
        conn = _FakeConnection(role="project_admin", org="project")
        with patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.get_registry", return_value=registry):
            self._module().cmd_show_study(conn, ["show_study", "study1"])
        assert conn.last_reply.get("name") == "study1"
        assert "error_code" not in conn.last_reply

    def test_org_admin_can_show_enrolled_study(self):
        registry = _make_registry({"study1": {"site_orgs": {"org_a": ["site-a"]}}})
        conn = _FakeConnection(role="org_admin", org="org_a")
        with patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.get_registry", return_value=registry):
            self._module().cmd_show_study(conn, ["show_study", "study1"])
        assert conn.last_reply.get("name") == "study1"

    def test_org_admin_cannot_show_hidden_study(self):
        registry = _make_registry({"study1": {"site_orgs": {"org_b": ["site-b"]}}})
        conn = _FakeConnection(role="org_admin", org="org_a")
        with patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.get_registry", return_value=registry):
            self._module().cmd_show_study(conn, ["show_study", "study1"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"

    def test_show_nonexistent_study_returns_not_found(self):
        registry = _make_registry({})
        conn = _FakeConnection(role="project_admin", org="project")
        with patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.get_registry", return_value=registry):
            self._module().cmd_show_study(conn, ["show_study", "ghost"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"


# ---------------------------------------------------------------------------
# Section 10: user membership commands
# ---------------------------------------------------------------------------


class TestUserMembership:
    def _module(self):
        return StudyCommandModule()

    def test_add_user_succeeds(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_user(conn, ["add_study_user", "study1", "newuser@x.com"])
        assert conn.last_reply.get("user") == "newuser@x.com"
        assert "error_code" not in conn.last_reply

    def test_add_user_duplicate_returns_user_already_in_study(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_user(conn, ["add_study_user", "study1", "admin@example.com"])
        assert conn.last_reply["error_code"] == "USER_ALREADY_IN_STUDY"

    def test_add_user_to_hidden_study_returns_not_found(self):
        conn = _FakeConnection(role="org_admin", org="org_b", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_user(conn, ["add_study_user", "study1", "newuser@x.com"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"

    def test_remove_user_succeeds(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_user(conn, ["remove_study_user", "study1", "admin@example.com"])
        assert conn.last_reply.get("removed") is True
        assert "error_code" not in conn.last_reply

    def test_remove_user_not_in_study_returns_user_not_in_study(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_user(conn, ["remove_study_user", "study1", "ghost@x.com"])
        assert conn.last_reply["error_code"] == "USER_NOT_IN_STUDY"

    def test_remove_user_from_hidden_study_returns_not_found(self):
        conn = _FakeConnection(role="org_admin", org="org_b", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_user(conn, ["remove_study_user", "study1", "admin@example.com"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"


# ---------------------------------------------------------------------------
# Section 11: LOCK_TIMEOUT
# ---------------------------------------------------------------------------


class TestLockTimeout:
    def _module(self):
        return StudyCommandModule()

    def test_lock_timeout_returns_lock_timeout_error(self):
        engine = _make_engine({"site-a": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with (
            patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.acquire_lock", return_value=False),
            patch("nvflare.private.fed.server.study_cmds.ServerEngine", MagicMock),
        ):
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "LOCK_TIMEOUT"
        assert conn.last_reply["exit_code"] == 3


# ---------------------------------------------------------------------------
# Section 12: atomicity — no write on validation failure
# ---------------------------------------------------------------------------


@contextmanager
def _mutation_ctx_with_write_tracker(initial_config=None):
    if initial_config is None:
        initial_config = _EMPTY_REGISTRY
    with (
        patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.acquire_lock", return_value=True),
        patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.release_lock"),
        patch("nvflare.private.fed.server.study_cmds.StudyRegistryService.initialize"),
        patch("nvflare.private.fed.server.study_cmds.ServerEngine", MagicMock),
        patch.object(StudyCommandModule, "_registry_path", return_value="/fake/path"),
        patch.object(StudyCommandModule, "_load_registry_config", side_effect=lambda _: deepcopy(initial_config)),
        patch.object(StudyCommandModule, "_write_registry_config") as mock_write,
    ):
        yield mock_write


class TestAtomicityGuarantee:
    def _module(self):
        return StudyCommandModule()

    def test_invalid_site_prevents_registry_write(self):
        engine = _make_engine({"site-a": "org_b"})  # wrong org
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx_with_write_tracker() as mock_write:
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"
        mock_write.assert_not_called()

    def test_partial_invalid_site_org_group_prevents_registry_write(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_c"})  # org_b:site-b is wrong
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx_with_write_tracker() as mock_write:
            self._module().cmd_register_study(
                conn,
                ["register_study", "study1", "--site-org", "org_a:site-a", "--site-org", "org_b:site-b"],
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"
        mock_write.assert_not_called()

    def test_study_not_found_prevents_registry_write_on_add_site(self):
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx_with_write_tracker(_EMPTY_REGISTRY) as mock_write:
            self._module().cmd_add_study_site(conn, ["add_study_site", "ghost", "--sites", "site-new"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"
        mock_write.assert_not_called()
