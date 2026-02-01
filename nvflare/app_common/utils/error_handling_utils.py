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

from typing import Optional, Set


def should_ignore_result_error(
    ignore_result_error: Optional[bool],
    client_name: str,
    failed_clients: Set[str],
    num_targets: int,
    min_responses: int,
) -> bool:
    """Determine whether a client result error should be ignored or cause a panic.

    This function implements the three-mode error handling policy:
    - None (Dynamic): Ignore errors if min_responses can still be reached, panic otherwise.
    - False (Strict): Never ignore errors, always panic.
    - True (Resilient): Always ignore errors, never panic.

    Note: This function can be safely called multiple times for the same client error.
    The failed_clients set uses idempotent add() operations, so duplicate calls for
    the same client will not affect the remaining count calculation.

    Args:
        ignore_result_error: The error handling mode.
            - None: Dynamic mode - ignore if min_responses still reachable.
            - False: Strict mode - always panic on error.
            - True: Resilient mode - always ignore errors.
        client_name: Name of the client with the error.
        failed_clients: Set of client names that have already failed (will be updated
            in dynamic mode only).
        num_targets: Total number of target clients for the current task.
        min_responses: Minimum number of responses required.

    Returns:
        True if the error should be ignored (no panic needed).
        False if a panic should be triggered.
    """
    if ignore_result_error is True:
        # Resilient mode - always ignore errors
        return True
    elif ignore_result_error is False:
        # Strict mode - always panic on error
        return False
    else:
        # Dynamic mode (None) - check if min_responses still reachable
        failed_clients.add(client_name)
        remaining_good_clients = num_targets - len(failed_clients)
        return remaining_good_clients >= min_responses


def get_error_handling_message(
    ignore_result_error: Optional[bool],
    client_name: str,
    error_code: str,
    current_round: int,
    controller_name: str,
    failed_clients: Set[str],
    num_targets: int,
    min_responses: int,
) -> str:
    """Generate appropriate log message based on error handling mode.

    Args:
        ignore_result_error: The error handling mode (None, False, or True).
        client_name: Name of the client with the error.
        error_code: The return code from the client result.
        current_round: Current training round.
        controller_name: Name of the controller class.
        failed_clients: Set of client names that have failed.
        num_targets: Total number of target clients.
        min_responses: Minimum number of responses required.

    Returns:
        Appropriate message string for logging.
    """
    if ignore_result_error is True:
        return f"Ignore the result from {client_name} at round {current_round}. " f"Result error code: {error_code}"
    elif ignore_result_error is False:
        return (
            f"Result from {client_name} is bad, error code: {error_code}. "
            f"{controller_name} exiting at round {current_round}."
        )
    else:
        remaining_good_clients = num_targets - len(failed_clients)
        if remaining_good_clients >= min_responses:
            return (
                f"Ignore the result from {client_name} at round {current_round}. "
                f"Result error code: {error_code}. "
                f"Remaining good clients ({remaining_good_clients}) >= min_responses ({min_responses})."
            )
        else:
            return (
                f"Result from {client_name} is bad, error code: {error_code}. "
                f"Cannot reach min_responses: remaining good clients ({remaining_good_clients}) < min_responses ({min_responses}). "
                f"{controller_name} exiting at round {current_round}."
            )
