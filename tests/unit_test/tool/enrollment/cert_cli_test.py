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

"""Unit tests for nvflare.tool.enrollment.cert_cli module."""

import json
import os
import shutil
import tempfile
from argparse import Namespace

import pytest

from nvflare.tool.enrollment.cert_cli import (
    _handle_init,
    _handle_server,
    _load_root_ca,
    handle_cert,
)


class TestHandleInit:
    """Tests for the init subcommand."""

    @pytest.fixture
    def output_dir(self):
        """Create and cleanup a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_init_creates_root_ca(self, output_dir):
        """Test that init creates root CA files."""
        args = Namespace(
            name="TestProject",
            output=output_dir,
            valid_days=365,
        )

        result = _handle_init(args)

        assert result == 0
        assert os.path.exists(os.path.join(output_dir, "rootCA.pem"))
        assert os.path.exists(os.path.join(output_dir, "rootCA.key"))
        assert os.path.exists(os.path.join(output_dir, "state", "cert.json"))

    def test_init_cert_json_has_required_fields(self, output_dir):
        """Test that cert.json has required fields."""
        args = Namespace(
            name="TestProject",
            output=output_dir,
            valid_days=365,
        )

        _handle_init(args)

        with open(os.path.join(output_dir, "state", "cert.json")) as f:
            cert_state = json.load(f)

        assert "root_cert" in cert_state
        assert "root_pri_key" in cert_state
        assert "issuer" in cert_state
        assert cert_state["issuer"] == "TestProject"

    def test_init_private_key_has_restrictive_permissions(self, output_dir):
        """Test that private key files have restrictive permissions."""
        args = Namespace(
            name="TestProject",
            output=output_dir,
            valid_days=365,
        )

        _handle_init(args)

        key_path = os.path.join(output_dir, "rootCA.key")
        mode = os.stat(key_path).st_mode & 0o777
        assert mode == 0o600


class TestHandleServer:
    """Tests for the server subcommand."""

    @pytest.fixture
    def ca_dir(self):
        """Create a temporary CA directory with root CA."""
        temp_dir = tempfile.mkdtemp()
        # Initialize a root CA
        args = Namespace(
            name="TestProject",
            output=temp_dir,
            valid_days=365,
        )
        _handle_init(args)
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def output_dir(self):
        """Create and cleanup a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_server_creates_certificates(self, ca_dir, output_dir):
        """Test that server creates server certificate files."""
        args = Namespace(
            name="server1",
            ca_path=ca_dir,
            output=output_dir,
            org="org",
            host=None,
            additional_hosts=None,
            valid_days=365,
        )

        result = _handle_server(args)

        assert result == 0
        assert os.path.exists(os.path.join(output_dir, "server.crt"))
        assert os.path.exists(os.path.join(output_dir, "server.key"))
        assert os.path.exists(os.path.join(output_dir, "rootCA.pem"))

    def test_server_with_custom_host(self, ca_dir, output_dir):
        """Test server certificate with custom host name."""
        args = Namespace(
            name="server1",
            ca_path=ca_dir,
            output=output_dir,
            org="org",
            host="server.example.com",
            additional_hosts=["server1.example.com", "localhost"],
            valid_days=365,
        )

        result = _handle_server(args)

        assert result == 0
        assert os.path.exists(os.path.join(output_dir, "server.crt"))

    def test_server_private_key_has_restrictive_permissions(self, ca_dir, output_dir):
        """Test that server private key has restrictive permissions."""
        args = Namespace(
            name="server1",
            ca_path=ca_dir,
            output=output_dir,
            org="org",
            host=None,
            additional_hosts=None,
            valid_days=365,
        )

        _handle_server(args)

        key_path = os.path.join(output_dir, "server.key")
        mode = os.stat(key_path).st_mode & 0o777
        assert mode == 0o600

    def test_server_fails_without_ca(self, output_dir):
        """Test that server fails when CA path doesn't exist."""
        args = Namespace(
            name="server1",
            ca_path="/nonexistent/path",
            output=output_dir,
            org="org",
            host=None,
            additional_hosts=None,
            valid_days=365,
        )

        result = _handle_server(args)

        assert result == 1


class TestLoadRootCA:
    """Tests for _load_root_ca function."""

    @pytest.fixture
    def ca_dir(self):
        """Create a temporary CA directory with root CA."""
        temp_dir = tempfile.mkdtemp()
        args = Namespace(
            name="TestProject",
            output=temp_dir,
            valid_days=365,
        )
        _handle_init(args)
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_from_cert_json(self, ca_dir):
        """Test loading root CA from state/cert.json."""
        root_cert, root_key, issuer = _load_root_ca(ca_dir)

        assert root_cert is not None
        assert root_key is not None
        assert issuer == "TestProject"
        assert b"BEGIN CERTIFICATE" in root_cert
        assert b"BEGIN" in root_key  # Could be RSA PRIVATE KEY or PRIVATE KEY

    def test_load_from_pem_files(self, ca_dir):
        """Test loading root CA from PEM files when cert.json is removed."""
        # Remove cert.json
        os.remove(os.path.join(ca_dir, "state", "cert.json"))
        os.rmdir(os.path.join(ca_dir, "state"))

        root_cert, root_key, issuer = _load_root_ca(ca_dir)

        assert root_cert is not None
        assert root_key is not None

    def test_load_fails_when_missing(self):
        """Test that loading fails when CA files are missing."""
        with pytest.raises(FileNotFoundError, match="Root CA not found"):
            _load_root_ca("/nonexistent/path")


class TestHandleCert:
    """Tests for the main handle_cert function."""

    def test_no_subcommand_shows_error(self):
        """Test that no subcommand shows an error."""
        args = Namespace(cert_cmd=None)
        result = handle_cert(args)
        assert result == 1

    def test_unknown_subcommand_shows_error(self):
        """Test that unknown subcommand shows an error."""
        args = Namespace(cert_cmd="unknown")
        result = handle_cert(args)
        assert result == 1

