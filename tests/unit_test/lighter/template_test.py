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

"""Tests for the NVFlare lighter templating system."""

import os
import tempfile

import pytest

import nvflare.lighter as lighter
from nvflare.lighter.constants import PropKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Project


def create_test_project(name="test_project"):
    """Helper to create a minimal project for testing."""
    project = Project(name=name, description="Test project")
    project.set_server(
        name="server1",
        org="nvidia",
        props={
            PropKey.ADMIN_PORT: 8003,
            PropKey.FED_LEARN_PORT: 8002,
        },
    )
    return project


def get_template_names():
    """Get all template names from the templates directory."""
    templates_dir = os.path.join(os.path.dirname(lighter.__file__), "templates")
    return [f[:-3] for f in os.listdir(templates_dir) if f.endswith(".j2")]


class TestAllTemplatesExist:
    """Verify all .j2 template files render correctly."""

    @pytest.fixture
    def ctx(self):
        """Create a context for testing."""
        project = create_test_project()
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ProvisionContext(tmpdir, project)

    @pytest.mark.parametrize("template_name", get_template_names())
    def test_template_renders(self, ctx, template_name):
        """Each .j2 template file renders without error."""
        section = ctx.build_section_from_template(template_name)
        assert section is not None and len(section) > 0, f"Template {template_name} is empty"


class TestBuildFromTemplate:
    """Tests for build_from_template and build_section_from_template."""

    @pytest.fixture
    def ctx_with_workspace(self):
        """Create a context with a workspace."""
        project = create_test_project()
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ProvisionContext(tmpdir, project), tmpdir

    def test_replacement(self, ctx_with_workspace):
        """Variable replacement works in templates."""
        ctx, _ = ctx_with_workspace
        section = ctx.build_section_from_template(
            "fed_server", replacement={"name": "myserver", "target": "localhost:8002"}
        )
        assert "myserver" in section
        assert "localhost:8002" in section

    def test_multiple_sections(self, ctx_with_workspace):
        """List of sections concatenates correctly."""
        ctx, _ = ctx_with_workspace
        section = ctx.build_section_from_template(
            [
                "readme_fs",
                "readme_fc",
            ]
        )
        assert "server" in section.lower()
        assert "client" in section.lower()

    def test_callback(self, ctx_with_workspace):
        """Content modification callback is applied."""
        ctx, _ = ctx_with_workspace
        section = ctx.build_section_from_template("readme_fs", content_modify_cb=lambda c: c.upper())
        assert section == section.upper()

    def test_executable_flag(self, ctx_with_workspace):
        """exe=True sets executable permission."""
        ctx, tmpdir = ctx_with_workspace
        dest_dir = os.path.join(tmpdir, "output")
        os.makedirs(dest_dir)
        ctx.build_from_template(dest_dir, "start_svr_sh", "start.sh", exe=True)
        mode = os.stat(os.path.join(dest_dir, "start.sh")).st_mode
        assert mode & 0o111  # At least one execute bit set
