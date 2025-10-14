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
from typing import Dict, Generator, List, Tuple
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_opt.confidential_computing.cc_manager import (
    CC_INFO,
    CC_NAMESPACE,
    CC_TOKEN,
    CC_TOKEN_VALIDATED,
    CC_VERIFICATION_FAILED,
    TOKEN_GENERATION_TIME,
    CCManager,
    CCTokenGenerateError,
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
    return token == VALID_TOKEN


def _create_peer_cc_context(site_name: str, token: str) -> Tuple[List[Dict[str, str]], FLContext]:
    """Create a peer context with CC token information for testing.

    Args:
        site_name: Name of the site
        token: Token to add

    Returns:
        Tuple[List[Dict[str, str]], FLContext]: A tuple containing token info and FL context
    """
    peer_ctx = FLContext()
    cc_info = [{CC_TOKEN: token, CC_NAMESPACE: TDX_NAMESPACE, CC_TOKEN_VALIDATED: False}]
    peer_ctx.set_prop(CC_INFO, cc_info)
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
        "enabled_sites": ["client1", "client2"],
    }


@pytest.fixture
def cc_test_env(basic_config) -> Generator[Tuple[CCManager, FLContext, TDXAuthorizer], None, None]:
    """Fixture for setting up the complete CC test environment.

    Args:
        basic_config: Basic configuration for CC manager

    Returns:
        Generator[Tuple[CCManager, FLContext, TDXAuthorizer], None, None]: A generator yielding
        the CC manager, FL context, and TDX authorizer
    """
    # Create CC manager with mocked methods
    with patch(
        "nvflare.app_opt.confidential_computing.cc_manager.CCManager._verify_running_jobs"
    ) as mock_verify_running_jobs:
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
        engine.get_component.return_value = tdx_authorizer
        cc_manager._setup_cc_authorizers(fl_ctx)

        tdx_authorizer.generate.return_value = VALID_TOKEN
        cc_manager._generate_tokens(fl_ctx)

        yield cc_manager, fl_ctx, tdx_authorizer


@pytest.fixture
def cc_test_env_with_mock_refresh(cc_test_env) -> Generator[Tuple[CCManager, FLContext, TDXAuthorizer], None, None]:
    """Fixture that provides CC test environment with mocked refresh method.

    Args:
        cc_test_env: Base CC test environment

    Returns:
        Generator[Tuple[CCManager, FLContext, TDXAuthorizer], None, None]: A generator yielding
        the CC manager, FL context, and TDX authorizer with mocked refresh
    """
    cc_manager, fl_ctx, tdx_authorizer = cc_test_env
    with patch(
        "nvflare.app_opt.confidential_computing.cc_manager.CCManager._ensure_fresh_tokens"
    ) as mock_ensure_fresh_tokens:
        yield cc_manager, fl_ctx, tdx_authorizer


class TestCCManager:
    """Test suite for CCManager class."""

    def test_setup_cc_authorizers(self, basic_config):
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

    @pytest.mark.parametrize(
        "client_name,token",
        [
            ("client1", VALID_TOKEN),
            ("client1", INVALID_TOKEN),
        ],
    )
    def test_add_client_token(self, logger, cc_test_env_with_mock_refresh, client_name, token):
        """Test adding client token to FL context."""
        logger.info("Testing add client token")
        cc_manager, _, _ = cc_test_env_with_mock_refresh
        cc_info, fl_ctx = _create_peer_cc_context(client_name, token)
        cc_manager._add_client_token(fl_ctx)

        assert cc_manager.participant_cc_info[client_name] == cc_info

    def test_collect_participants_tokens(self, logger, cc_test_env_with_mock_refresh):
        """Test collecting tokens for participants."""
        logger.info("Testing collect participants tokens")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env_with_mock_refresh
        cc_info1, fl_ctx = _create_peer_cc_context("client1", VALID_TOKEN)
        cc_manager._add_client_token(fl_ctx)
        cc_info2, fl_ctx = _create_peer_cc_context("client2", VALID_TOKEN)
        cc_manager._add_client_token(fl_ctx)

        server_cc_info, _ = _create_peer_cc_context("server", VALID_TOKEN)

        participants_tokens = cc_manager._collect_participants_tokens(["client1", "client2"])

        assert participants_tokens == {
            "server": server_cc_info,
            "client1": cc_info1,
            "client2": cc_info2,
        }

    def test_token_generation_error(self, logger, cc_test_env_with_mock_refresh):
        """Test handling of token generation errors."""
        logger.info("Testing token generation error handling")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env_with_mock_refresh
        tdx_authorizer.generate.side_effect = CCTokenGenerateError("Failed to generate token")

        with pytest.raises(RuntimeError) as exc_info:
            cc_manager._generate_tokens(fl_ctx)

        assert "failed to generate CC token" in str(exc_info.value)

    @pytest.mark.parametrize(
        "participants,tokens,expected_error",
        [
            (["client1"], [VALID_TOKEN], None),  # Single client, valid token
            (["client1", "client2"], [VALID_TOKEN, VALID_TOKEN], None),  # Multiple clients, all valid
            (
                ["client1", "client2"],
                [VALID_TOKEN, INVALID_TOKEN],
                CC_VERIFICATION_FAILED,
            ),  # Multiple clients, one invalid
            ([], [], None),  # Empty participants list - verify system handles no participants gracefully
        ],
    )
    def test_token_verification(self, logger, cc_test_env_with_mock_refresh, participants, tokens, expected_error):
        """Test token verification with various scenarios.

        Args:
            logger: Logger fixture
            cc_test_env_with_mock_refresh: CC test environment with mocked refresh
            participants: List of participant names
            tokens: List of tokens corresponding to participants
            expected_error: Expected error message or None for success
        """
        logger.info(f"Testing verification with participants {participants}")
        cc_manager, fl_ctx, _ = cc_test_env_with_mock_refresh

        # Add tokens for all participants
        for client_name, token in zip(participants, tokens):
            _, fl_ctx = _create_peer_cc_context(client_name, token)
            cc_manager._add_client_token(fl_ctx)

        # Verify tokens
        participants_tokens = cc_manager._collect_participants_tokens(participants)
        err = cc_manager._validate_participants_tokens(participants_tokens)

        if expected_error:
            assert expected_error in err
            # Check that the error mentions the client with invalid token
            invalid_client = participants[tokens.index(next(t for t in tokens if t != VALID_TOKEN))]
            assert invalid_client in err
        else:
            assert not err

    def test_token_expiration(self, logger, cc_test_env):
        """Test token expiration handling.

        Args:
            logger: Logger fixture
            cc_test_env: CC test environment fixture
        """
        logger.info("Testing token expiration")
        cc_manager, fl_ctx, tdx_authorizer = cc_test_env

        cc_info = cc_manager.participant_cc_info["server"]
        logger.info(f"cc_info: {cc_info}")

        # Get initial token and generation time
        initial_token = cc_info[0][CC_TOKEN]
        initial_time = cc_info[0][TOKEN_GENERATION_TIME]

        # Simulate token expiration by setting generation time to 0
        cc_info[0][TOKEN_GENERATION_TIME] = 0

        # Refresh expired tokens
        cc_manager._ensure_fresh_tokens()

        # Verify token was refreshed
        assert cc_info[0][TOKEN_GENERATION_TIME] > initial_time
