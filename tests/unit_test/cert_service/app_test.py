# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for CertServiceApp (app.py)."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestCertServiceAppInit:
    """Tests for CertServiceApp initialization."""

    def test_init_with_new_ca_generation(self):
        """Test initialization generates new root CA on first start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(
                data_dir=tmpdir,
                project_name="TestProject",
                api_key="test-api-key",
            )

            # Check CA files were generated
            assert os.path.exists(os.path.join(tmpdir, "rootCA.pem"))
            assert os.path.exists(os.path.join(tmpdir, "rootCA.key"))

            # Check state/cert.json was created
            cert_json = os.path.join(tmpdir, "state", "cert.json")
            assert os.path.exists(cert_json)

            # Verify cert.json contents
            with open(cert_json) as f:
                cert_state = json.load(f)
            assert "root_cert" in cert_state
            assert "root_pri_key" in cert_state
            assert cert_state["issuer"] == "TestProject"

            assert app.cert_service is not None
            assert app.token_service is not None

    def test_init_with_existing_ca(self):
        """Test initialization uses existing root CA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            # First init creates CA
            CertServiceApp(
                data_dir=tmpdir,
                project_name="TestProject",
                api_key="test-api-key",
            )

            # Get cert modification time
            cert_path = os.path.join(tmpdir, "rootCA.pem")
            mtime1 = os.path.getmtime(cert_path)

            # Second init should reuse CA
            app2 = CertServiceApp(
                data_dir=tmpdir,
                project_name="TestProject",
                api_key="test-api-key",
            )

            mtime2 = os.path.getmtime(cert_path)
            assert mtime1 == mtime2  # File not modified

            assert app2.cert_service is not None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(
                data_dir=tmpdir,
                api_key="my-secure-key",
            )

            assert app.api_key == "my-secure-key"

    def test_init_without_api_key(self):
        """Test initialization without API key logs warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            # Clear env var if set
            with patch.dict(os.environ, {}, clear=True):
                app = CertServiceApp(data_dir=tmpdir)

            assert app.api_key is None

    def test_init_api_key_from_env(self):
        """Test initialization with API key from environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            with patch.dict(os.environ, {"NVFLARE_API_KEY": "env-key"}):
                app = CertServiceApp(data_dir=tmpdir)

            assert app.api_key == "env-key"


class TestLoadConfig:
    """Tests for _load_config method."""

    def test_load_config_defaults(self):
        """Test default configuration values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(data_dir=tmpdir, api_key="test")

            assert app.config["server"]["host"] == "0.0.0.0"
            assert app.config["server"]["port"] == 8443
            assert app.config["storage"]["type"] == "sqlite"
            assert app.config["pending"]["timeout"] == 604800

    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(
                    """
server:
  host: "127.0.0.1"
  port: 9000
project_name: "CustomProject"
api_key: "file-api-key"
pending:
  timeout: 86400
"""
                )
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(config_path, data_dir=tmpdir)

            assert app.config["server"]["host"] == "127.0.0.1"
            assert app.config["server"]["port"] == 9000
            assert app.config["project_name"] == "CustomProject"
            assert app.config["pending"]["timeout"] == 86400

    def test_load_config_overrides(self):
        """Test that kwargs override config file values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(
                    """
server:
  host: "127.0.0.1"
  port: 9000
"""
                )
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(
                config_path,
                data_dir=tmpdir,
                host="0.0.0.0",
                port=8443,
                api_key="test",
            )

            # Kwargs should override file
            assert app.config["server"]["host"] == "0.0.0.0"
            assert app.config["server"]["port"] == 8443

    def test_load_config_derived_paths(self):
        """Test that CA and DB paths are derived from data_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(data_dir=tmpdir, api_key="test")

            assert app.config["ca"]["cert"] == os.path.join(tmpdir, "rootCA.pem")
            assert app.config["ca"]["key"] == os.path.join(tmpdir, "rootCA.key")
            assert app.config["storage"]["path"] == os.path.join(tmpdir, "enrollment.db")


class TestValidateRootCA:
    """Tests for _validate_root_ca method."""

    def test_validate_valid_ca(self):
        """Test validation passes for valid CA files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            # First init creates valid CA
            CertServiceApp(data_dir=tmpdir, api_key="test")

            # Manually call validation
            app = CertServiceApp.__new__(CertServiceApp)
            app.logger = MagicMock()

            cert_path = os.path.join(tmpdir, "rootCA.pem")
            key_path = os.path.join(tmpdir, "rootCA.key")

            result = app._validate_root_ca(cert_path, key_path)
            assert result is not None
            cert, key = result
            assert cert is not None
            assert key is not None

    def test_validate_missing_cert(self):
        """Test validation fails for missing certificate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp.__new__(CertServiceApp)
            app.logger = MagicMock()

            result = app._validate_root_ca(
                os.path.join(tmpdir, "missing.pem"),
                os.path.join(tmpdir, "missing.key"),
            )
            assert result is None

    def test_validate_mismatched_key(self):
        """Test validation fails when cert and key don't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp
            from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key

            # Generate two different key pairs
            key1, pub1 = generate_keys()
            key2, pub2 = generate_keys()

            # Create cert with key1
            cert = generate_cert(
                subject=Identity("Test"),
                issuer=Identity("Test"),
                signing_pri_key=key1,
                subject_pub_key=pub1,
                valid_days=365,
                ca=True,
            )

            cert_path = os.path.join(tmpdir, "rootCA.pem")
            key_path = os.path.join(tmpdir, "rootCA.key")

            # Save cert but use different key
            with open(cert_path, "wb") as f:
                f.write(serialize_cert(cert))
            with open(key_path, "wb") as f:
                f.write(serialize_pri_key(key2))  # Mismatched key!

            app = CertServiceApp.__new__(CertServiceApp)
            app.logger = MagicMock()

            result = app._validate_root_ca(cert_path, key_path)
            assert result is None


class TestRequireApiKey:
    """Tests for _require_api_key decorator."""

    @pytest.fixture
    def app_with_key(self):
        """Create CertServiceApp with API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(data_dir=tmpdir, api_key="secret-key")
            yield app

    @pytest.fixture
    def flask_test_client(self, app_with_key):
        """Create Flask test client."""
        app_with_key.flask_app.testing = True
        return app_with_key.flask_app.test_client()

    def test_valid_api_key(self, flask_test_client):
        """Test request with valid API key succeeds."""
        response = flask_test_client.get(
            "/api/v1/enrolled",
            headers={"Authorization": "Bearer secret-key"},
        )
        # Should not get 401
        assert response.status_code != 401

    def test_invalid_api_key(self, flask_test_client):
        """Test request with invalid API key fails."""
        response = flask_test_client.get(
            "/api/v1/enrolled",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401
        assert b"Invalid API key" in response.data

    def test_missing_auth_header(self, flask_test_client):
        """Test request without Authorization header fails."""
        response = flask_test_client.get("/api/v1/enrolled")
        assert response.status_code == 401
        assert b"Missing or invalid Authorization header" in response.data

    def test_malformed_auth_header(self, flask_test_client):
        """Test request with malformed Authorization header fails."""
        response = flask_test_client.get(
            "/api/v1/enrolled",
            headers={"Authorization": "Basic secret-key"},  # Not Bearer
        )
        assert response.status_code == 401

    def test_no_api_key_configured(self):
        """Test admin endpoints fail when no API key configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            with patch.dict(os.environ, {}, clear=True):
                app = CertServiceApp(data_dir=tmpdir)

            app.flask_app.testing = True
            client = app.flask_app.test_client()

            response = client.get(
                "/api/v1/enrolled",
                headers={"Authorization": "Bearer any-key"},
            )
            assert response.status_code == 500
            assert b"API key not configured on server" in response.data


class TestGetWsgiApp:
    """Tests for get_wsgi_app method."""

    def test_get_wsgi_app(self):
        """Test getting WSGI application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(data_dir=tmpdir, api_key="test")
            wsgi_app = app.get_wsgi_app()

            assert wsgi_app is app.flask_app


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_with_path(self):
        """Test create_app with config path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(
                    f"""
data_dir: {tmpdir}
api_key: test-key
"""
                )
            from nvflare.cert_service.app import create_app

            app = create_app(config_path)
            assert app is not None
            assert app.api_key == "test-key"

    def test_create_app_from_env(self):
        """Test create_app loads config from environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(
                    f"""
data_dir: {tmpdir}
api_key: env-test-key
"""
                )
            with patch.dict(os.environ, {"CERT_SERVICE_CONFIG": config_path}):
                from nvflare.cert_service.app import create_app

                app = create_app()
                assert app.api_key == "env-test-key"


class TestEnsureRootCAExists:
    """Tests for _ensure_root_ca_exists method."""

    def test_ensure_creates_directory(self):
        """Test that data directory is created if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "path")

            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(data_dir=nested_dir, api_key="test")

            assert os.path.exists(nested_dir)
            assert os.path.exists(os.path.join(nested_dir, "rootCA.pem"))

    def test_ensure_sets_key_permissions(self):
        """Test that private key has restrictive permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            CertServiceApp(data_dir=tmpdir, api_key="test")

            key_path = os.path.join(tmpdir, "rootCA.key")
            mode = os.stat(key_path).st_mode & 0o777
            assert mode == 0o600

    def test_ensure_raises_on_invalid_existing_ca(self):
        """Test that invalid existing CA files raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid cert file
            cert_path = os.path.join(tmpdir, "rootCA.pem")
            key_path = os.path.join(tmpdir, "rootCA.key")

            with open(cert_path, "w") as f:
                f.write("invalid cert data")
            with open(key_path, "w") as f:
                f.write("invalid key data")

            from nvflare.cert_service.app import CertServiceApp

            with pytest.raises(ValueError, match="Root CA files exist but are invalid"):
                CertServiceApp(data_dir=tmpdir, api_key="test")
