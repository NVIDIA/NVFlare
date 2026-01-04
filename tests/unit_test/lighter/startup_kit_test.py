# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for nvflare.lighter.startup_kit module."""

import json
import os
import shutil
import tempfile

import pytest

from nvflare.lighter.startup_kit import (
    filter_builders,
    generate_package,
    load_default_project,
    package_from_project,
    parse_endpoint_uri,
)


class TestParseEndpointUri:
    """Tests for parse_endpoint_uri function."""

    def test_parse_single_port_grpc(self):
        """Test parsing single port gRPC URI."""
        result = parse_endpoint_uri("grpc://server.example.com:8002")
        assert result["scheme"] == "grpc"
        assert result["host"] == "server.example.com"
        assert result["fl_port"] == 8002
        assert result["admin_port"] == 8002  # Defaults to fl_port

    def test_parse_two_ports_grpc(self):
        """Test parsing two port gRPC URI."""
        result = parse_endpoint_uri("grpc://server:8002:8003")
        assert result["scheme"] == "grpc"
        assert result["host"] == "server"
        assert result["fl_port"] == 8002
        assert result["admin_port"] == 8003

    def test_parse_http_scheme(self):
        """Test parsing HTTP URI."""
        result = parse_endpoint_uri("http://server:8443")
        assert result["scheme"] == "http"
        assert result["host"] == "server"
        assert result["fl_port"] == 8443
        assert result["admin_port"] == 8443

    def test_parse_tcp_scheme(self):
        """Test parsing TCP URI."""
        result = parse_endpoint_uri("tcp://localhost:9002")
        assert result["scheme"] == "tcp"
        assert result["host"] == "localhost"
        assert result["fl_port"] == 9002

    def test_invalid_uri_no_port(self):
        """Test invalid URI without port."""
        with pytest.raises(ValueError, match="Invalid endpoint URI"):
            parse_endpoint_uri("grpc://server")

    def test_invalid_uri_no_scheme(self):
        """Test invalid URI without scheme."""
        with pytest.raises(ValueError, match="Invalid endpoint URI"):
            parse_endpoint_uri("server:8002")

    def test_invalid_uri_format(self):
        """Test completely invalid URI format."""
        with pytest.raises(ValueError, match="Invalid endpoint URI"):
            parse_endpoint_uri("not-a-valid-uri")


class TestFilterBuilders:
    """Tests for filter_builders function."""

    def test_filter_removes_cert_builder(self):
        """Test that CertBuilder is removed."""
        builders = [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder"},
        ]
        result = filter_builders(builders)
        assert len(result) == 2
        paths = [b["path"] for b in result]
        assert "nvflare.lighter.impl.cert.CertBuilder" not in paths

    def test_filter_removes_signature_builder(self):
        """Test that SignatureBuilder is removed."""
        builders = [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ]
        result = filter_builders(builders)
        assert len(result) == 1
        assert result[0]["path"] == "nvflare.lighter.impl.workspace.WorkspaceBuilder"

    def test_filter_removes_both_builders(self):
        """Test that both CertBuilder and SignatureBuilder are removed."""
        builders = [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder"},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ]
        result = filter_builders(builders)
        assert len(result) == 2
        paths = [b["path"] for b in result]
        assert "nvflare.lighter.impl.cert.CertBuilder" not in paths
        assert "nvflare.lighter.impl.signature.SignatureBuilder" not in paths

    def test_filter_preserves_custom_builders(self):
        """Test that custom builders are preserved."""
        builders = [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.docker.DockerBuilder", "args": {"image": "test"}},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "custom.my_builder.MyCustomBuilder"},
        ]
        result = filter_builders(builders)
        assert len(result) == 3
        paths = [b["path"] for b in result]
        assert "nvflare.lighter.impl.docker.DockerBuilder" in paths
        assert "custom.my_builder.MyCustomBuilder" in paths


class TestLoadDefaultProject:
    """Tests for load_default_project function."""

    def test_load_default_project(self):
        """Test loading the default project.yml."""
        project = load_default_project()
        assert project is not None
        assert "api_version" in project
        assert project["api_version"] == 3
        assert "builders" in project
        assert "participants" in project

    def test_default_project_has_cert_builder(self):
        """Test that default project includes CertBuilder (before filtering)."""
        project = load_default_project()
        builders = project.get("builders", [])
        paths = [b.get("path") for b in builders]
        assert "nvflare.lighter.impl.cert.CertBuilder" in paths


class TestGeneratePackage:
    """Tests for generate_package function."""

    @pytest.fixture
    def output_dir(self):
        """Create and cleanup a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        output = os.path.join(temp_dir, "test-package")
        yield output
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_generate_client_package(self, output_dir):
        """Test generating a client package."""
        endpoint_info = {
            "scheme": "grpc",
            "host": "server.example.com",
            "fl_port": 8002,
            "admin_port": 8002,
        }

        result = generate_package(
            name="test-site",
            participant_type="client",
            endpoint_info=endpoint_info,
            output_dir=output_dir,
        )

        assert result == output_dir
        assert os.path.exists(output_dir)

        # Check expected files exist
        assert os.path.exists(os.path.join(output_dir, "startup", "fed_client.json"))
        assert os.path.exists(os.path.join(output_dir, "startup", "start.sh"))
        assert os.path.exists(os.path.join(output_dir, "local", "resources.json.default"))

        # Verify fed_client.json content
        with open(os.path.join(output_dir, "startup", "fed_client.json")) as f:
            config = json.load(f)
        assert config["servers"][0]["identity"] == "server.example.com"
        assert config["servers"][0]["service"]["scheme"] == "grpc"
        assert "sp_end_point" in config["overseer_agent"]["args"]

    def test_generate_admin_package(self, output_dir):
        """Test generating an admin package."""
        endpoint_info = {
            "scheme": "grpc",
            "host": "server",
            "fl_port": 8002,
            "admin_port": 8003,
        }

        result = generate_package(
            name="admin@example.com",
            participant_type="admin",
            endpoint_info=endpoint_info,
            output_dir=output_dir,
            role="org_admin",
        )

        assert result == output_dir
        assert os.path.exists(os.path.join(output_dir, "startup", "fed_admin.json"))

        # Verify fed_admin.json content
        with open(os.path.join(output_dir, "startup", "fed_admin.json")) as f:
            config = json.load(f)
        assert config["admin"]["username"] == "admin@example.com"
        assert config["admin"]["port"] == 8003

    def test_generate_relay_package(self, output_dir):
        """Test generating a relay package."""
        endpoint_info = {
            "scheme": "http",
            "host": "server",
            "fl_port": 8443,
            "admin_port": 8443,
        }

        result = generate_package(
            name="relay-east",
            participant_type="relay",
            endpoint_info=endpoint_info,
            output_dir=output_dir,
            listening_host="0.0.0.0",
            listening_port=8010,
        )

        assert result == output_dir
        assert os.path.exists(os.path.join(output_dir, "startup", "fed_relay.json"))
        assert os.path.exists(os.path.join(output_dir, "local", "comm_config.json"))

        # Verify comm_config.json has listening configuration
        with open(os.path.join(output_dir, "local", "comm_config.json")) as f:
            config = json.load(f)
        assert config["internal"]["resources"]["port"] == 8010

    def test_no_certificates_generated(self, output_dir):
        """Test that no certificates are generated in the package."""
        endpoint_info = {
            "scheme": "grpc",
            "host": "server",
            "fl_port": 8002,
            "admin_port": 8002,
        }

        generate_package(
            name="test-site",
            participant_type="client",
            endpoint_info=endpoint_info,
            output_dir=output_dir,
        )

        # Ensure no certificate files exist
        startup_dir = os.path.join(output_dir, "startup")
        assert not os.path.exists(os.path.join(startup_dir, "client.crt"))
        assert not os.path.exists(os.path.join(startup_dir, "client.key"))
        assert not os.path.exists(os.path.join(startup_dir, "rootCA.pem"))
        assert not os.path.exists(os.path.join(startup_dir, "signature.json"))

    def test_custom_org(self, output_dir):
        """Test generating package with custom organization."""
        endpoint_info = {
            "scheme": "grpc",
            "host": "server",
            "fl_port": 8002,
            "admin_port": 8002,
        }

        generate_package(
            name="test-site",
            participant_type="client",
            endpoint_info=endpoint_info,
            output_dir=output_dir,
            org="my_hospital",  # Use underscore (hyphens not allowed in org names)
        )

        # Package should be generated successfully with custom org
        assert os.path.exists(output_dir)

    def test_generate_server_package(self, output_dir):
        """Test generating a server package."""
        endpoint_info = {
            "scheme": "grpc",
            "host": "server.example.com",
            "fl_port": 8002,
            "admin_port": 8003,
        }

        result = generate_package(
            name="server1",
            participant_type="server",
            endpoint_info=endpoint_info,
            output_dir=output_dir,
        )

        assert result == output_dir
        assert os.path.exists(output_dir)

        # Check expected server files exist
        assert os.path.exists(os.path.join(output_dir, "startup", "fed_server.json"))
        assert os.path.exists(os.path.join(output_dir, "startup", "start.sh"))

        # Verify fed_server.json content
        with open(os.path.join(output_dir, "startup", "fed_server.json")) as f:
            config = json.load(f)
        assert "servers" in config

    def test_server_package_no_certificates(self, output_dir):
        """Test that server package has no certificates."""
        endpoint_info = {
            "scheme": "grpc",
            "host": "server.example.com",
            "fl_port": 8002,
            "admin_port": 8002,
        }

        generate_package(
            name="server1",
            participant_type="server",
            endpoint_info=endpoint_info,
            output_dir=output_dir,
        )

        # Ensure no certificate files exist
        startup_dir = os.path.join(output_dir, "startup")
        assert not os.path.exists(os.path.join(startup_dir, "server.crt"))
        assert not os.path.exists(os.path.join(startup_dir, "server.key"))
        assert not os.path.exists(os.path.join(startup_dir, "rootCA.pem"))


class TestPackageFromProject:
    """Tests for package_from_project function."""

    @pytest.fixture
    def workspace_dir(self):
        """Create and cleanup a temporary workspace directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def project_file(self, workspace_dir):
        """Create a test project.yml file."""
        import yaml

        project_content = {
            "api_version": 3,
            "name": "test_project",
            "description": "Test project for package_from_project",
            "participants": [
                {"name": "server1", "type": "server", "org": "org1", "fed_learn_port": 8002, "admin_port": 8003},
                {"name": "site-1", "type": "client", "org": "org1"},
                {"name": "site-2", "type": "client", "org": "org1"},
                {"name": "admin@test.com", "type": "admin", "org": "org1", "role": "lead"},
            ],
            "builders": [
                {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
                {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"scheme": "grpc"}},
                {"path": "nvflare.lighter.impl.cert.CertBuilder"},
                {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
            ],
        }

        project_path = os.path.join(workspace_dir, "project.yml")
        with open(project_path, "w") as f:
            yaml.dump(project_content, f)

        return project_path

    def test_package_all_participants(self, project_file, workspace_dir):
        """Test packaging all participants from a project file."""
        output_workspace = os.path.join(workspace_dir, "output")

        result_dir = package_from_project(project_file, output_workspace)

        assert os.path.exists(result_dir)

        # All participants should be created
        assert os.path.exists(os.path.join(result_dir, "server1"))
        assert os.path.exists(os.path.join(result_dir, "site-1"))
        assert os.path.exists(os.path.join(result_dir, "site-2"))
        assert os.path.exists(os.path.join(result_dir, "admin@test.com"))

    def test_no_certificates_in_project_packages(self, project_file, workspace_dir):
        """Test that no certificates are generated when using project file."""
        output_workspace = os.path.join(workspace_dir, "output")

        result_dir = package_from_project(project_file, output_workspace)

        # Check client package has no certificates
        client_startup = os.path.join(result_dir, "site-1", "startup")
        assert not os.path.exists(os.path.join(client_startup, "client.crt"))
        assert not os.path.exists(os.path.join(client_startup, "client.key"))
        assert not os.path.exists(os.path.join(client_startup, "rootCA.pem"))
        assert not os.path.exists(os.path.join(client_startup, "signature.json"))

        # Check server package has no certificates
        server_startup = os.path.join(result_dir, "server1", "startup")
        assert not os.path.exists(os.path.join(server_startup, "server.crt"))
        assert not os.path.exists(os.path.join(server_startup, "server.key"))

    def test_config_files_exist(self, project_file, workspace_dir):
        """Test that configuration files are generated correctly."""
        output_workspace = os.path.join(workspace_dir, "output")

        result_dir = package_from_project(project_file, output_workspace)

        # Client config files
        assert os.path.exists(os.path.join(result_dir, "site-1", "startup", "fed_client.json"))
        assert os.path.exists(os.path.join(result_dir, "site-1", "startup", "start.sh"))

        # Server config files
        assert os.path.exists(os.path.join(result_dir, "server1", "startup", "fed_server.json"))

        # Admin config files
        assert os.path.exists(os.path.join(result_dir, "admin@test.com", "startup", "fed_admin.json"))


class TestEnrollmentFiles:
    """Tests for enrollment file generation (Auto-Scale workflow)."""

    @pytest.fixture
    def workspace_dir(self):
        """Create a temporary workspace directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_generate_package_with_cert_service_url(self, workspace_dir):
        """Test that enrollment.json is created when --cert-service is provided."""
        from nvflare.lighter.startup_kit import generate_single_package

        endpoint_info = {"scheme": "grpc", "host": "server", "fl_port": 8002, "admin_port": 8002}

        result_dir = generate_single_package(
            name="hospital-1",
            participant_type="client",
            endpoint_info=endpoint_info,
            workspace=workspace_dir,
            cert_service_url="https://cert-service.example.com:8443",
        )

        enrollment_json_path = os.path.join(result_dir, "startup", "enrollment.json")
        assert os.path.exists(enrollment_json_path)

        with open(enrollment_json_path) as f:
            enrollment_config = json.load(f)

        assert enrollment_config["cert_service_url"] == "https://cert-service.example.com:8443"

    def test_generate_package_with_token(self, workspace_dir):
        """Test that enrollment_token is created when --token is provided."""
        from nvflare.lighter.startup_kit import generate_single_package

        endpoint_info = {"scheme": "grpc", "host": "server", "fl_port": 8002, "admin_port": 8002}

        result_dir = generate_single_package(
            name="hospital-2",
            participant_type="client",
            endpoint_info=endpoint_info,
            workspace=workspace_dir,
            enrollment_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test-payload.signature",
        )

        token_path = os.path.join(result_dir, "startup", "enrollment_token")
        assert os.path.exists(token_path)

        with open(token_path) as f:
            token = f.read()

        assert token == "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test-payload.signature"

        # Check restrictive permissions
        mode = os.stat(token_path).st_mode & 0o777
        assert mode == 0o600

    def test_generate_package_with_both_options(self, workspace_dir):
        """Test that both enrollment files are created when both options provided."""
        from nvflare.lighter.startup_kit import generate_single_package

        endpoint_info = {"scheme": "grpc", "host": "server", "fl_port": 8002, "admin_port": 8002}

        result_dir = generate_single_package(
            name="hospital-3",
            participant_type="client",
            endpoint_info=endpoint_info,
            workspace=workspace_dir,
            cert_service_url="https://cert-service:8443",
            enrollment_token="test-token",
        )

        # Both files should exist
        assert os.path.exists(os.path.join(result_dir, "startup", "enrollment.json"))
        assert os.path.exists(os.path.join(result_dir, "startup", "enrollment_token"))

    def test_generate_package_without_enrollment_options(self, workspace_dir):
        """Test that no enrollment files are created when options not provided."""
        from nvflare.lighter.startup_kit import generate_single_package

        endpoint_info = {"scheme": "grpc", "host": "server", "fl_port": 8002, "admin_port": 8002}

        result_dir = generate_single_package(
            name="hospital-4",
            participant_type="client",
            endpoint_info=endpoint_info,
            workspace=workspace_dir,
            # No cert_service_url or enrollment_token
        )

        # Neither file should exist
        assert not os.path.exists(os.path.join(result_dir, "startup", "enrollment.json"))
        assert not os.path.exists(os.path.join(result_dir, "startup", "enrollment_token"))
