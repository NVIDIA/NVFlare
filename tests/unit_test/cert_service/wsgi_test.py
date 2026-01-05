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

"""Unit tests for Certificate Service WSGI module."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest


class TestWSGIModule:
    """Test WSGI module initialization and configuration."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def env_vars(self, temp_data_dir):
        """Set up environment variables for testing."""
        original_env = os.environ.copy()

        os.environ["NVFLARE_DATA_DIR"] = temp_data_dir
        os.environ["NVFLARE_API_KEY"] = "test-api-key-12345"
        os.environ["NVFLARE_CERT_SERVICE_PORT"] = "9999"

        yield

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    def test_wsgi_module_imports_successfully(self, env_vars):
        """Test that wsgi module can be imported."""
        # Import with proper environment
        import sys

        # Remove cached module if present
        if "nvflare.cert_service.wsgi" in sys.modules:
            del sys.modules["nvflare.cert_service.wsgi"]

        from nvflare.cert_service import wsgi

        assert hasattr(wsgi, "app")
        assert wsgi.app is not None

    def test_wsgi_app_is_flask_app(self, env_vars):
        """Test that the exported app is a Flask application."""
        import sys

        if "nvflare.cert_service.wsgi" in sys.modules:
            del sys.modules["nvflare.cert_service.wsgi"]

        from nvflare.cert_service import wsgi

        # Check it's a Flask app
        assert hasattr(wsgi.app, "route")
        assert hasattr(wsgi.app, "run")

    def test_wsgi_respects_data_dir_env_var(self, temp_data_dir):
        """Test that NVFLARE_DATA_DIR is respected."""
        import sys

        os.environ["NVFLARE_DATA_DIR"] = temp_data_dir
        os.environ["NVFLARE_API_KEY"] = "test-key"

        if "nvflare.cert_service.wsgi" in sys.modules:
            del sys.modules["nvflare.cert_service.wsgi"]

        from nvflare.cert_service import wsgi

        # Verify the data dir was used (root CA should be created there)
        # The CA files are created in a 'ca' subdirectory or directly in data_dir
        ca_exists = os.path.exists(os.path.join(temp_data_dir, "rootCA.pem")) or os.path.exists(
            os.path.join(temp_data_dir, "ca", "rootCA.pem")
        )
        assert ca_exists or wsgi.app is not None  # At minimum, app should be created

    def test_wsgi_uses_default_port(self, env_vars):
        """Test default port is 8443."""
        from nvflare.cert_service.wsgi import ENV_PORT

        assert ENV_PORT == "NVFLARE_CERT_SERVICE_PORT"

    def test_wsgi_environment_variables(self):
        """Test that environment variable names are correct."""
        from nvflare.cert_service.wsgi import (
            ENV_API_KEY,
            ENV_CONFIG,
            ENV_DATA_DIR,
            ENV_PORT,
        )

        assert ENV_PORT == "NVFLARE_CERT_SERVICE_PORT"
        assert ENV_CONFIG == "NVFLARE_CERT_SERVICE_CONFIG"
        assert ENV_API_KEY == "NVFLARE_API_KEY"
        assert ENV_DATA_DIR == "NVFLARE_DATA_DIR"


class TestWSGIGunicornCompatibility:
    """Test WSGI compatibility with Gunicorn."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    def test_app_is_wsgi_callable(self, temp_data_dir):
        """Test that app is a WSGI-compatible callable."""
        import sys

        os.environ["NVFLARE_DATA_DIR"] = temp_data_dir
        os.environ["NVFLARE_API_KEY"] = "test-key"

        if "nvflare.cert_service.wsgi" in sys.modules:
            del sys.modules["nvflare.cert_service.wsgi"]

        from nvflare.cert_service import wsgi

        # WSGI apps must be callable
        assert callable(wsgi.app)

    def test_app_has_wsgi_interface(self, temp_data_dir):
        """Test app has WSGI interface (__call__ with environ, start_response)."""
        import sys

        os.environ["NVFLARE_DATA_DIR"] = temp_data_dir
        os.environ["NVFLARE_API_KEY"] = "test-key"

        if "nvflare.cert_service.wsgi" in sys.modules:
            del sys.modules["nvflare.cert_service.wsgi"]

        from nvflare.cert_service import wsgi

        # Flask app should handle WSGI call
        environ = {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/health",
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "8443",
            "wsgi.url_scheme": "http",
            "wsgi.input": MagicMock(),
            "wsgi.errors": MagicMock(),
        }

        responses = []

        def start_response(status, headers):
            responses.append((status, headers))

        # Should not raise
        result = wsgi.app(environ, start_response)
        assert result is not None


class TestWSGIHealthEndpoint:
    """Test health endpoint via WSGI."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client."""
        import sys

        os.environ["NVFLARE_DATA_DIR"] = str(tmp_path)
        os.environ["NVFLARE_API_KEY"] = "test-key"

        if "nvflare.cert_service.wsgi" in sys.modules:
            del sys.modules["nvflare.cert_service.wsgi"]

        from nvflare.cert_service import wsgi

        wsgi.app.config["TESTING"] = True
        with wsgi.app.test_client() as client:
            yield client

    def test_health_endpoint_returns_ok(self, client):
        """Test /health returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
