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

import logging
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError, CCTokenVerifyError
from nvflare.app_opt.confidential_computing.cc_manager import (
    CC_INFO,
    CC_NAMESPACE,
    CC_TOKEN,
    CC_TOKEN_VALIDATED,
    CC_VERIFICATION_FAILED,
    CCManager,
)
from nvflare.app_opt.confidential_computing.tdx_authorizer import TDX_NAMESPACE, TDXAuthorizer

VALID_TOKEN = "valid_token"
INVALID_TOKEN = "invalid_token"


def _verify_token(token: str) -> bool:
    """Verify if the token is valid.

    Args:
        token: The token to verify

    Returns:
        bool: True if token is valid, False otherwise
    """
    if token == INVALID_TOKEN:
        raise CCTokenVerifyError("Invalid token")
    return token == VALID_TOKEN


def _create_peer_cc_context(site_name: str, token: str) -> tuple[list[dict[str, str]], FLContext]:
    """Create a peer context with CC token information for testing.

    Args:
        site_name: Name of the site
        token: Token to add

    Returns:
        Tuple[List[Dict[str, str]], FLContext]: A tuple containing token info and FL context
    """
    peer_ctx = FLContext()
    cc_info = [{CC_TOKEN: token, CC_NAMESPACE: TDX_NAMESPACE, CC_TOKEN_VALIDATED: False}]
    peer_ctx.set_prop(CC_INFO, {site_name: cc_info})
    peer_ctx.set_prop(ReservedKey.IDENTITY_NAME, site_name)
    fl_ctx = Mock(spec=FLContext)
    fl_ctx.get_peer_context.return_value = peer_ctx
    return cc_info, fl_ctx


@pytest.fixture(scope="module")
def logger():
    """Fixture for logger.

    Returns:
        logging.Logger: Configured logger for the test module
    """
    # Get logger for this module
    test_logger = logging.getLogger(__name__)
    test_logger.setLevel(logging.INFO)

    # Only add handler if it doesn't exist
    if not test_logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        # Add handler to logger
        test_logger.addHandler(console_handler)

    return test_logger


@pytest.fixture
def basic_config():
    """Fixture for basic CC manager configuration."""
    return {
        "issues_conf": [{"issuer_id": "tdx_authorizer", "token_expiration": 10}],
        "verify_ids": ["tdx_authorizer"],
        "enabled_sites": ["server", "client1", "client2"],
    }


@pytest.fixture
def cc_test_env(basic_config) -> Generator[tuple[CCManager, FLContext, Mock], None, None]:
    """Fixture for setting up the complete CC test environment.

    Args:
        basic_config: Basic configuration for CC manager

    Returns:
        Generator[Tuple[CCManager, FLContext, Mock], None, None]: A generator yielding
        the CC manager, FL context, and TDX authorizer mock
    """
    # Create CC manager
    cc_manager = CCManager(
        cc_issuers_conf=basic_config["issues_conf"],
        cc_verifier_ids=basic_config["verify_ids"],
        cc_enabled_sites=basic_config["enabled_sites"],
    )

    # Set up FL context
    fl_ctx = Mock(spec=FLContext)
    fl_ctx.get_identity_name.return_value = "server"
    engine = Mock(spec=ServerEngineSpec)
    fl_ctx.get_engine.return_value = engine

    # Set up TDX authorizer
    tdx_authorizer = Mock(spec=TDXAuthorizer)
    tdx_authorizer.get_namespace.return_value = TDX_NAMESPACE
    tdx_authorizer.verify = _verify_token
    tdx_authorizer.generate.return_value = VALID_TOKEN
    engine.get_component.return_value = tdx_authorizer

    cc_manager._setup_cc_authorizers(fl_ctx)

    yield cc_manager, fl_ctx, tdx_authorizer


class TestCCManager:
    """Test suite for CCManager class."""

    def test_setup_cc_authorizers(self, basic_config):
        """Test setting up CC authorizers."""
        cc_manager = CCManager(
            cc_issuers_conf=basic_config["issues_conf"],
            cc_verifier_ids=basic_config["verify_ids"],
            cc_enabled_sites=basic_config["enabled_sites"],
        )

        # Set up FL context
        fl_ctx = Mock(spec=FLContext)
        fl_ctx.get_identity_name.return_value = "server"
        engine = Mock(spec=ServerEngineSpec)
        fl_ctx.get_engine.return_value = engine

        # Set up TDX authorizer
        tdx_authorizer = Mock(spec=TDXAuthorizer)
        tdx_authorizer.get_namespace.return_value = TDX_NAMESPACE
        tdx_authorizer.verify = _verify_token
        engine.get_component.return_value = tdx_authorizer
        cc_manager._setup_cc_authorizers(fl_ctx)

        assert cc_manager.cc_issuers == {tdx_authorizer: 10}
        assert cc_manager.cc_verifiers == {TDX_NAMESPACE: tdx_authorizer}

    def test_generate_fresh_tokens(self, logger, cc_test_env):
        """Test generating fresh tokens for validation."""
        logger.info("Testing fresh token generation")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        # Generate fresh tokens
        fresh_tokens = cc_manager._generate_fresh_tokens_for_validation()

        # Verify tokens were generated
        assert len(fresh_tokens) == 1
        assert fresh_tokens[0][CC_TOKEN] == VALID_TOKEN
        assert fresh_tokens[0][CC_NAMESPACE] == TDX_NAMESPACE
        assert fresh_tokens[0][CC_TOKEN_VALIDATED] is False

    def test_token_generation_error(self, logger, cc_test_env):
        """Test handling of token generation errors."""
        logger.info("Testing token generation error handling")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        # Make the authorizer throw an error
        tdx_authorizer.generate.side_effect = CCTokenGenerateError("Failed to generate token")

        # Generate tokens should handle the error gracefully and return empty list
        fresh_tokens = cc_manager._generate_fresh_tokens_for_validation()

        # Should return empty list when token generation fails
        assert len(fresh_tokens) == 0

    @pytest.mark.parametrize(
        "participants_tokens,expected_error",
        [
            # Single participant with valid token
            ({"client1": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}]}, None),
            # Multiple participants with valid tokens
            (
                {
                    "client1": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
                    "client2": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
                },
                None,
            ),
            # One participant with invalid token
            (
                {
                    "client1": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
                    "client2": [{CC_TOKEN: INVALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
                },
                CC_VERIFICATION_FAILED,
            ),
            # Empty participants dict
            ({}, None),
        ],
    )
    def test_validate_participants_tokens(self, logger, cc_test_env, participants_tokens, expected_error):
        """Test token validation with various scenarios.

        Args:
            logger: Logger fixture
            cc_test_env: CC test environment fixture
            participants_tokens: Dict of participant tokens to validate
            expected_error: Expected error message or None for success
        """
        logger.info(f"Testing validation with participants {list(participants_tokens.keys())}")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        # Validate tokens
        err = cc_manager._validate_participants_tokens(participants_tokens)

        if expected_error:
            assert expected_error in err
            # Check that the error mentions the participant with invalid token
            for participant, tokens in participants_tokens.items():
                if tokens and tokens[0][CC_TOKEN] == INVALID_TOKEN:
                    assert participant in err
        else:
            assert not err

    def test_verify_participants_tokens(self, logger, cc_test_env):
        """Test the _verify_participants_tokens method directly."""
        logger.info("Testing _verify_participants_tokens")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        participants_tokens = {
            "client1": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
            "client2": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
        }

        result, invalid_list = cc_manager._verify_participants_tokens(participants_tokens)

        # Both clients should be valid
        assert result["client1." + TDX_NAMESPACE] is True
        assert result["client2." + TDX_NAMESPACE] is True
        assert len(invalid_list) == 0

    def test_verify_participants_tokens_with_invalid(self, logger, cc_test_env):
        """Test verification with invalid tokens."""
        logger.info("Testing _verify_participants_tokens with invalid token")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        participants_tokens = {
            "client1": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
            "client2": [{CC_TOKEN: INVALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
        }

        result, invalid_list = cc_manager._verify_participants_tokens(participants_tokens)

        # Client1 should be valid, client2 should be invalid
        assert result["client1." + TDX_NAMESPACE] is True
        assert len(invalid_list) == 1
        assert "client2" in invalid_list[0]

    def test_verify_participants_tokens_not_in_enabled_sites(self, logger, cc_test_env):
        """Test that sites not in enabled_sites are automatically validated."""
        logger.info("Testing sites not in enabled_sites list")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        participants_tokens = {
            "client3": [{CC_TOKEN: INVALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],  # Not in enabled_sites
        }

        result, invalid_list = cc_manager._verify_participants_tokens(participants_tokens)

        # client3 is not in enabled_sites, so it should be marked as valid without verification
        assert result["client3"] is True
        assert len(invalid_list) == 0

    def test_validate_client_tokens(self, logger, cc_test_env):
        """Test validating client tokens from peer context."""
        logger.info("Testing client token validation")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        # Create peer context with valid token
        cc_info, mock_fl_ctx = _create_peer_cc_context("client1", VALID_TOKEN)

        # Mock _shutdown_system
        with patch.object(cc_manager, "_shutdown_system") as mock_shutdown:
            cc_manager._validate_client_tokens(mock_fl_ctx)
            # Should not call shutdown for valid token
            mock_shutdown.assert_not_called()

    def test_validate_client_tokens_invalid(self, logger, cc_test_env):
        """Test validating invalid client tokens."""
        logger.info("Testing invalid client token validation")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        # Create peer context with invalid token
        cc_info, mock_fl_ctx = _create_peer_cc_context("client1", INVALID_TOKEN)

        # Mock _shutdown_system
        with patch.object(cc_manager, "_shutdown_system") as mock_shutdown:
            cc_manager._validate_client_tokens(mock_fl_ctx)
            # Should call shutdown for invalid token
            mock_shutdown.assert_called_once()
            args = mock_shutdown.call_args[0]
            assert "CC info validation failed" in args[0]

    def test_generate_and_attach_tokens(self, logger, cc_test_env):
        """Test generating and attaching tokens to FL context."""
        logger.info("Testing token generation and attachment")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        cc_manager._generate_and_attach_tokens(fl_ctx)

        # Verify set_prop was called with CC_INFO
        fl_ctx.set_prop.assert_called_once()
        call_args = fl_ctx.set_prop.call_args
        assert call_args[1]["key"] == CC_INFO
        assert call_args[1]["sticky"] is False
        assert call_args[1]["private"] is False

        # Verify the value contains token info
        cc_info = call_args[1]["value"]
        print(f"CFUCK {cc_info=}")
        assert len(cc_info) == 1
        assert cc_info["server"][0][CC_TOKEN] == VALID_TOKEN
        assert cc_info["server"][0][CC_NAMESPACE] == TDX_NAMESPACE
