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

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthenticated
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer, CCTokenGenerateError, CCTokenVerifyError
from nvflare.app_opt.confidential_computing.cc_manager import (
    CC_INFO,
    CC_NAMESPACE,
    CC_TOKEN,
    CC_TOKEN_VALIDATED,
    CC_VERIFICATION_FAILED,
    CCManager,
)
from nvflare.app_opt.confidential_computing.tdx_authorizer import TDX_NAMESPACE, TDXAuthorizer
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as F3ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message

VALID_TOKEN = "valid_token"
INVALID_TOKEN = "invalid_token"


@pytest.mark.parametrize("name", ["registration_token_timeout", "refresh_token_timeout", "get_token_request_timeout"])
@pytest.mark.parametrize("value", [0, -1, True, "30", float("nan"), float("inf")])
def test_invalid_generation_budget(name, value):
    with pytest.raises(ValueError, match=name):
        CCManager([], [], **{name: value})


def test_peer_timeout_must_exceed_refresh_budget():
    with pytest.raises(ValueError, match="must exceed"):
        CCManager([], [], get_token_request_timeout=30, refresh_token_timeout=30)


def test_manager_selects_registration_and_refresh_budgets(cc_test_env):
    manager, context, issuer = cc_test_env
    manager._generate_and_attach_tokens(context)
    timeout, stop = issuer.generate_with_retry.call_args.args
    assert 299 < timeout <= 300
    assert stop is manager.cross_validation_stop_event
    manager._generate_fresh_tokens_for_validation()
    timeout, stop = issuer.generate_with_retry.call_args.args
    assert 29 < timeout <= 30
    assert manager.get_token_request_timeout == 45


def test_stop_cancels_generation_before_validation_thread_starts(cc_test_env):
    manager, _, issuer = cc_test_env
    manager._stop_cross_site_validation()
    assert manager.cross_validation_stop_event.is_set()
    assert manager._generate_fresh_tokens_for_validation() == []
    issuer.generate.assert_not_called()


def test_generation_budget_is_shared_across_issuers(cc_test_env):
    manager, _, first = cc_test_env
    second = Mock(spec=CCAuthorizer)
    second.generate_with_retry.return_value = VALID_TOKEN
    manager.cc_issuers[second] = 300
    with patch("nvflare.app_opt.confidential_computing.cc_manager.time.monotonic", side_effect=[0, 1, 9]):
        assert len(manager._generate_fresh_tokens_for_validation(timeout=10)) == 2
    assert first.generate_with_retry.call_args.args[0] == 9
    assert second.generate_with_retry.call_args.args[0] == 1


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
    fl_ctx.get_prop.side_effect = lambda key, default=None: site_name if key == FLContextKey.CLIENT_NAME else default
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
    tdx_authorizer.verify_for_site.side_effect = lambda token, site_name: _verify_token(token)
    tdx_authorizer.generate.return_value = VALID_TOKEN
    tdx_authorizer.generate_with_retry.side_effect = lambda timeout, cancel_event: CCAuthorizer.generate_with_retry(
        tdx_authorizer, timeout, cancel_event
    )
    engine.get_component.return_value = tdx_authorizer

    cc_manager._setup_cc_authorizers(fl_ctx)

    yield cc_manager, fl_ctx, tdx_authorizer


class TestCCManager:
    """Test suite for CCManager class."""

    @pytest.mark.parametrize("failure", ["empty", "issuer_error", "unexpected_error"])
    def test_token_refresh_failure_returns_error_reply(self, cc_test_env, failure):
        manager, _, issuer = cc_test_env
        if failure == "empty":
            issuer.generate.return_value = ""
        elif failure == "issuer_error":
            issuer.generate.side_effect = CCTokenGenerateError("issuer unavailable")
        else:
            issuer.generate.side_effect = RuntimeError("unexpected issuer failure")

        reply = manager._handle_token_refresh_request(new_cell_message({}, {"requester": "client1"}))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.PROCESS_EXCEPTION
        expected_error = (
            "Failed to generate token: unexpected issuer failure"
            if failure == "unexpected_error"
            else "Failed to generate tokens"
        )
        assert reply.get_header(MessageHeaderKey.ERROR) == expected_error
        assert reply.payload is None

    def test_token_refresh_success_reply(self, cc_test_env):
        manager, _, _ = cc_test_env
        manager.site_name = "client1"

        reply = manager._handle_token_refresh_request(new_cell_message({}, {"requester": "server"}))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.OK
        assert reply.get_header(MessageHeaderKey.ERROR) is None
        assert reply.payload == {
            "site_name": "client1",
            "cc_info": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE, CC_TOKEN_VALIDATED: False}],
        }

    def test_get_sites_failure_returns_error_reply(self, cc_test_env):
        manager, _, _ = cc_test_env
        with patch.object(manager, "_get_all_sites", side_effect=RuntimeError("site lookup failed")):
            reply = manager._handle_get_sites_request(new_cell_message({}, {"requester": "client1"}))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.PROCESS_EXCEPTION
        assert reply.get_header(MessageHeaderKey.ERROR) == "Failed to get sites: site lookup failed"
        assert reply.payload is None

    def test_get_sites_success_reply(self, cc_test_env):
        manager, _, _ = cc_test_env
        sites = [("client1-fqcn", "client1")]
        with patch.object(manager, "_get_all_sites", return_value=sites):
            reply = manager._handle_get_sites_request(new_cell_message({}, {"requester": "client1"}))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.OK
        assert reply.get_header(MessageHeaderKey.ERROR) is None
        assert reply.payload == {"sites": sites}

    def test_periodic_validation_error_reply_remains_fail_closed(self, cc_test_env):
        manager, context, issuer = cc_test_env
        issuer.generate.return_value = ""
        reply = manager._handle_token_refresh_request(new_cell_message({}, {"requester": "server"}))
        issuer.generate.return_value = VALID_TOKEN
        manager.site_name = "server"
        manager.cc_enabled_sites = ["server", "client1"]
        cell = Mock()
        cell.send_request.return_value = reply
        context.get_engine().get_cell = Mock(return_value=cell)

        with (
            patch.object(manager, "_get_all_cc_enabled_sites", return_value=[("client1-fqcn", "client1")]),
            patch.object(manager, "_shutdown_system") as shutdown,
            patch.object(manager, "_validate_participants_tokens") as validate,
        ):
            assert manager._perform_cross_site_validation(context) is False

        validate.assert_not_called()
        shutdown.assert_called_once_with(
            "Exception in cross-site validation: Failed to collect tokens from sites: ['client1-fqcn']", context
        )

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
            with pytest.raises(NotAuthenticated, match="CC info validation failed"):
                cc_manager._validate_client_tokens(mock_fl_ctx)
            mock_shutdown.assert_not_called()

    @pytest.mark.parametrize("payload", [None, {}, {"server": []}, {"client1": []}, {"client1": [], "server": []}])
    def test_registration_requires_exact_protected_client(self, cc_test_env, payload):
        manager, _, _ = cc_test_env
        _, context = _create_peer_cc_context("client1", VALID_TOKEN)
        context.get_peer_context().set_prop(CC_INFO, payload)
        with pytest.raises(NotAuthenticated):
            manager._validate_client_tokens(context)

    def test_ordinary_client_needs_no_attestation(self, cc_test_env):
        manager, _, verifier = cc_test_env
        _, context = _create_peer_cc_context("plain-client", VALID_TOKEN)
        context.get_peer_context.return_value = None
        with patch.object(manager, "_shutdown_system") as shutdown:
            manager._validate_client_tokens(context)
        shutdown.assert_not_called()
        verifier.verify_for_site.assert_not_called()

    def test_verifier_gets_expected_participant(self, cc_test_env):
        manager, _, verifier = cc_test_env
        _, context = _create_peer_cc_context("client1", VALID_TOKEN)
        manager._validate_client_tokens(context)
        verifier.verify_for_site.assert_called_once_with(VALID_TOKEN, "client1")

    @pytest.mark.parametrize("tokens", [[None], [{}], [{CC_NAMESPACE: "unknown"}], [{CC_NAMESPACE: []}]])
    def test_invalid_token_envelopes_fail_closed(self, cc_test_env, tokens):
        manager, _, _ = cc_test_env
        assert manager._validate_participants_tokens({"client1": tokens})

    @pytest.mark.parametrize("returned_name", ["client1", "server", "client2"])
    def test_periodic_response_cannot_rename_requested_site(self, cc_test_env, returned_name):
        manager, context, _ = cc_test_env
        manager.site_name = "server"
        manager.cc_enabled_sites = ["server", "client1"]
        response = Mock()
        response.get_header.return_value = "ok"
        response.payload = {
            "site_name": returned_name,
            "cc_info": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}],
        }
        context.get_engine().get_cell = Mock()
        context.get_engine().get_cell.return_value.send_request.return_value = response
        with patch.object(manager, "_get_all_cc_enabled_sites", return_value=[("client1-fqcn", "client1")]):
            if returned_name == "client1":
                assert "client1" in manager._collect_all_site_tokens(context)
            else:
                with pytest.raises(RuntimeError, match="Failed to collect tokens"):
                    manager._collect_all_site_tokens(context)

    @pytest.mark.parametrize(
        "payload", [None, {}, {"x": []}, {"client1": []}, {"server": []}, {"server": [], "x": []}, []]
    )
    def test_protected_server_requires_exact_identity(self, cc_test_env, payload):
        manager, _, verifier = cc_test_env
        context = FLContext()
        context.set_prop(CC_INFO, payload)
        with patch.object(manager, "_shutdown_system") as shutdown:
            manager._validate_server_tokens(context)
        shutdown.assert_called_once()
        verifier.verify_for_site.assert_not_called()

    @pytest.mark.parametrize("token,valid", [(VALID_TOKEN, True), (INVALID_TOKEN, False)])
    def test_protected_server_token_is_bound_to_root_identity(self, cc_test_env, token, valid):
        manager, _, verifier = cc_test_env
        context = FLContext()
        context.set_prop(CC_INFO, {"server": [{CC_TOKEN: token, CC_NAMESPACE: TDX_NAMESPACE}]})
        with patch.object(manager, "_shutdown_system") as shutdown:
            manager._validate_server_tokens(context)
        assert shutdown.called is not valid
        verifier.verify_for_site.assert_called_once_with(token, "server")

    @pytest.mark.parametrize("payload", [None, {}, {"server": []}])
    def test_explicitly_ordinary_server_needs_no_token(self, cc_test_env, payload):
        manager, _, verifier = cc_test_env
        manager.cc_enabled_sites = ["client1", "client2"]
        context = FLContext()
        context.set_prop(CC_INFO, payload)
        with patch.object(manager, "_shutdown_system") as shutdown:
            manager._validate_server_tokens(context)
        shutdown.assert_not_called()
        verifier.verify_for_site.assert_not_called()

    @pytest.mark.parametrize("sites", [[("client1", "client1")], [("server", "server"), ("client1", "client1")]])
    def test_server_cannot_omit_locally_required_participants(self, cc_test_env, sites):
        manager, _, verifier = cc_test_env
        manager.site_name = "client1"
        context = Mock(spec=FLContext)
        cell = context.get_engine().get_cell()
        cell.send_request.return_value = new_cell_message(
            {MessageHeaderKey.RETURN_CODE: F3ReturnCode.OK}, {"sites": sites}
        )
        with patch.object(manager, "_shutdown_system") as shutdown:
            assert manager._perform_cross_site_validation(context) is False
        shutdown.assert_called_once()
        assert "Missing required CC participants" in shutdown.call_args.args[0]
        # Only the discovery request was sent; an incomplete set never passes.
        assert cell.send_request.call_count == 1
        verifier.verify_for_site.assert_not_called()

    @pytest.mark.parametrize("server_protected", [False, True])
    def test_complete_periodic_coverage_passes(self, cc_test_env, server_protected):
        manager, _, verifier = cc_test_env
        manager.site_name = "client1"
        manager.cc_enabled_sites = ["client1", "client2"] + (["server"] if server_protected else [])
        context = Mock(spec=FLContext)
        cell = context.get_engine().get_cell()
        tokens = [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}]
        sites = [(name, name) for name in manager.cc_enabled_sites]
        cell.send_request.side_effect = [
            new_cell_message({MessageHeaderKey.RETURN_CODE: F3ReturnCode.OK}, {"sites": sites}),
            *[
                new_cell_message(
                    {MessageHeaderKey.RETURN_CODE: F3ReturnCode.OK}, {"site_name": name, "cc_info": tokens}
                )
                for name in manager.cc_enabled_sites
                if name != manager.site_name
            ],
        ]
        with patch.object(manager, "_shutdown_system") as shutdown:
            assert manager._perform_cross_site_validation(context) is True
        shutdown.assert_not_called()
        assert {call.args[1] for call in verifier.verify_for_site.call_args_list} == set(manager.cc_enabled_sites)

    def test_collected_tokens_cannot_omit_required_site(self, cc_test_env):
        manager, context, _ = cc_test_env
        own = {"client1": [{CC_TOKEN: VALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}]}
        with (
            patch.object(manager, "_collect_all_site_tokens", return_value=own),
            patch.object(manager, "_shutdown_system") as shutdown,
        ):
            assert manager._perform_cross_site_validation(context) is False
        assert "Missing required CC participants" in shutdown.call_args.args[0]

    def test_first_periodic_round_has_bounded_startup_window(self, cc_test_env):
        manager, context, _ = cc_test_env
        stop = Mock()
        stop.wait.return_value = True
        manager.cross_validation_stop_event = stop
        with patch("nvflare.app_opt.confidential_computing.cc_manager.random.uniform", return_value=0):
            manager._cross_site_validation_loop(context)
        stop.wait.assert_called_once_with(timeout=manager.cross_validation_interval)

    @pytest.mark.parametrize(
        "sites",
        [
            [None],
            [["client2"]],
            [["", "client2"]],
            [("client2", "client2"), ("client2", "client2")],
            [("server", "client2")],
            [("client2", "server")],
        ],
    )
    def test_invalid_discovery_routes_are_rejected(self, cc_test_env, sites):
        manager, context, _ = cc_test_env
        manager.site_name = "client1"
        context.get_engine().get_cell = Mock()
        with patch.object(manager, "_get_all_cc_enabled_sites", return_value=sites):
            with pytest.raises(RuntimeError):
                manager._collect_all_site_tokens(context)

    def test_complete_coverage_does_not_accept_invalid_tokens(self, cc_test_env):
        manager, context, _ = cc_test_env
        tokens = {name: [{CC_TOKEN: INVALID_TOKEN, CC_NAMESPACE: TDX_NAMESPACE}] for name in manager.cc_enabled_sites}
        with (
            patch.object(manager, "_collect_all_site_tokens", return_value=tokens),
            patch.object(manager, "_shutdown_system") as shutdown,
        ):
            assert manager._perform_cross_site_validation(context) is False
        assert "Cross-site validation failed" in shutdown.call_args.args[0]

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
