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

import os
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

# _ERROR_REGISTRY is a plain mutable dict so entries can be added simply at module load time.
# It is never exported; callers must use ERROR_REGISTRY (the frozen public view) or the
# helper functions below.  The two-name pattern avoids accidental mutation from call sites
# while keeping the definition readable.
_ERROR_REGISTRY: Dict[str, Dict[str, str]] = {
    # --- General ---
    "CONNECTION_FAILED": {
        "message": "Could not connect to the FLARE server.",
        "hint": "Check server status with 'nvflare system status'.",
    },
    "AUTH_FAILED": {
        "message": "Authentication failed.",
        "hint": "Check startup kit credentials.",
    },
    "TIMEOUT": {
        "message": "Operation timed out.",
        "hint": "Increase --timeout or check server load.",
    },
    "INVALID_ARGS": {
        "message": "Invalid arguments.",
        "hint": "Run the command with -h or --help for usage.",
    },
    "STARTUP_KIT_MISSING": {
        "message": "Startup kit not found.",
        "hint": "Run 'nvflare config list' and 'nvflare config use <id>', pass --kit-id <id> or --startup-kit <path>, or set NVFLARE_STARTUP_KIT_DIR for automation.",
    },
    "SITE_NOT_FOUND": {
        "message": "Site '{site}' is not connected.",
        "hint": "Use 'nvflare system status' to list connected sites.",
    },
    "LOG_CONFIG_INVALID": {
        "message": "Log config is not valid JSON or a recognised log mode.",
        "hint": "Supply a valid dictConfig JSON file or one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, msg_only, full, verbose, reload.",
    },
    "SERVER_UNREACHABLE": {
        "message": "Server stopped or job ended before command was delivered.",
        "hint": "Check server status with 'nvflare system status'.",
    },
    "SYSTEM_NOT_READY": {
        "message": "FLARE system is not ready yet.",
        "hint": "Wait for clients to connect, then retry 'nvflare system status'. If this persists, check POC service logs or client logs.",
    },
    "INTERNAL_ERROR": {
        "message": "An unexpected error occurred.",
        "hint": "This is likely a bug. Re-run in a development environment for a traceback, or report the issue.",
    },
    "CLI_ERROR": {
        "message": "Command failed.",
        "hint": "",
    },
    "STUDY_NOT_FOUND": {
        "message": "Study '{study}' not found.",
        "hint": "Verify the study name. If the study exists and you expect access, contact a project_admin.",
    },
    "STUDY_ALREADY_EXISTS": {
        "message": "Study '{study}' already exists.",
        "hint": "Use 'nvflare study show {study}' or contact a project_admin to update access.",
    },
    "STUDY_HAS_JOBS": {
        "message": "Study '{study}' has associated jobs and cannot be removed.",
        "hint": "Archive or delete the associated jobs before retrying.",
    },
    "INVALID_STUDY_NAME": {
        "message": "Invalid study name '{study}'.",
        "hint": "Use only lowercase letters, numbers, underscores, and hyphens.",
    },
    "INVALID_SITE": {
        "message": "Invalid site value.",
        "hint": "Use a comma-separated list of valid site names.",
    },
    "USER_ALREADY_IN_STUDY": {
        "message": "User '{user}' is already in study '{study}'.",
        "hint": "Use a different user or remove the existing entry first.",
    },
    "USER_NOT_IN_STUDY": {
        "message": "User '{user}' is not in study '{study}'.",
        "hint": "Use 'nvflare study add-user' to add the user first.",
    },
    "STARTUP_KIT_NOT_CONFIGURED": {
        "message": "No active startup kit is configured.",
        "hint": "Run 'nvflare config list' and 'nvflare config use <id>', pass --kit-id <id> or --startup-kit <path>, or set NVFLARE_STARTUP_KIT_DIR for automation.",
    },
    "LOCK_TIMEOUT": {
        "message": "Study registry is busy.",
        "hint": "Another study mutation is in progress. Retry shortly.",
    },
    "NOT_AUTHORIZED": {
        "message": "Not authorized for this operation.",
        "hint": "Use a startup kit with the required admin role.",
    },
    # --- Job commands ---
    "JOB_NOT_FOUND": {
        "message": "Job '{job_id}' does not exist.",
        "hint": "Use 'nvflare job list' to see available job IDs.",
    },
    "JOB_NOT_RUNNING": {
        "message": "Job '{job_id}' is not currently running.",
        "hint": "Use 'nvflare job list' to check job status.",
    },
    "JOB_INVALID": {
        "message": "Job folder is not a valid NVFlare job.",
        "hint": "Check meta.json and config_fed_server.json.",
    },
    "SUBMIT_TOKEN_CONFLICT": {
        "message": "A job with this submit token already exists with different content.",
        "hint": (
            "Use a new submit token when submitting different job content, "
            "or resubmit identical job content to reuse the existing job."
        ),
    },
    "SUBMIT_TOKEN_JOB_DELETED": {
        "message": "This submit token refers to a deleted job.",
        "hint": "Use a new submit token to submit the job again.",
    },
    "LOG_NOT_FOUND": {
        "message": "Job logs are not available for site '{site}'.",
        "hint": "Verify that client log streaming is enabled and that the site has run this job.",
    },
    # --- Cert commands ---
    "OUTPUT_DIR_NOT_WRITABLE": {
        "message": "Cannot write to output directory {path}.",
        "hint": "Check directory permissions or choose a different output directory.",
    },
    "CERT_GENERATION_FAILED": {
        "message": "Failed to generate certificate.",
        "hint": "Check that the cryptography package is installed and up-to-date.",
    },
    "CA_ALREADY_EXISTS": {
        "message": "Root CA already exists at {path}.",
        "hint": "Use --force to overwrite, or choose a different output directory.",
    },
    "CA_NOT_FOUND": {
        "message": "No root CA found at {ca_dir}.",
        "hint": "Run 'nvflare cert init' first, or specify the correct --ca-dir.",
    },
    "CA_LOAD_FAILED": {
        "message": "Failed to load root CA material from {ca_dir}.",
        "hint": "Check that rootCA.pem and rootCA.key are readable, valid, and unencrypted.",
    },
    "CSR_NOT_FOUND": {
        "message": "CSR file not found: {path}.",
        "hint": "Check the path to the .csr file.",
    },
    "REQUEST_ZIP_NOT_FOUND": {
        "message": "Request zip not found: {path}.",
        "hint": "Provide the .request.zip file created by 'nvflare cert request'.",
    },
    "INVALID_CSR": {
        "message": "Invalid or corrupt CSR file: {path}.",
        "hint": "Create a new request with 'nvflare cert request'.",
    },
    "CERT_ALREADY_EXISTS": {
        "message": "Signed certificate already exists at {path}.",
        "hint": "Use --force to overwrite.",
    },
    "ROOTCA_ALREADY_EXISTS": {
        "message": "Root CA certificate already exists at {path}.",
        "hint": "Use --force to overwrite.",
    },
    "INVALID_CERT_TYPE": {
        "message": "Invalid certificate type '{cert_type}'.",
        "hint": "Use one of: client, server, org_admin, lead, member.",
    },
    "KEY_ALREADY_EXISTS": {
        "message": "Private key already exists at {path}.",
        "hint": "Use --force to overwrite, or choose a different output directory.",
    },
    "INVALID_NAME": {
        "message": "Invalid name '{name}': {reason}",
        "hint": "The name must be 64 characters or fewer and must not contain leading/trailing whitespace.",
    },
    "CSR_GENERATION_FAILED": {
        "message": "CSR generation failed.",
        "hint": "Check that the cryptography package is installed and up-to-date.",
    },
    "CERT_TYPE_UNKNOWN": {
        "message": "Unknown certificate type in '{cert}': the certificate type is missing or unrecognized.",
        "hint": "Use a signed zip produced by 'nvflare cert approve'.",
    },
    "CERT_SIGNING_FAILED": {
        "message": "Certificate signing failed: {reason}",
        "hint": "Check that the CA key and certificate are valid and not corrupted.",
    },
    "CERT_OUTPUT_WRITE_FAILED": {
        "message": "Failed to write signed certificate output to {path}.",
        "hint": "Check output directory permissions and available disk space, then retry.",
    },
    # --- Package commands ---
    "CERT_NOT_FOUND": {
        "message": "Certificate file not found: {path}.",
        "hint": "Provide the signed certificate received from the Project Admin.",
    },
    "KEY_NOT_FOUND": {
        "message": "Private key file not found: {path}.",
        "hint": "Provide the private key generated by 'nvflare cert request'.",
    },
    "ROOTCA_NOT_FOUND": {
        "message": "Root CA file not found: {path}.",
        "hint": "Provide the rootCA.pem received from the Project Admin.",
    },
    "INVALID_ENDPOINT": {
        "message": "Invalid endpoint URI: {endpoint}.",
        "hint": "Use format: grpc://host:port, tcp://host:port, or http://host:port.",
    },
    "OUTPUT_DIR_EXISTS": {
        "message": "Output directory already exists: {path}.",
        "hint": "Use --force to package into a new prod_NN stage directory.",
    },
    "SIGNED_ZIP_NOT_FOUND": {
        "message": "Signed zip not found: {path}.",
        "hint": "Provide the .signed.zip returned by 'nvflare cert approve'.",
    },
    "AMBIGUOUS_KEY": {
        "message": "Multiple *.key files found in {path}: {files}",
        "hint": "Select one participant key for this internal packaging operation.",
    },
    # --- Distributed provisioning: signed zip validation ---
    "INVALID_SIGNED_ZIP": {
        "message": "Invalid signed zip.",
        "hint": "Use the .signed.zip returned by 'nvflare cert approve'.",
    },
    "INVALID_PROJECT_NAME": {
        "message": "Invalid project name.",
        "hint": "Project name must start with a letter or digit and contain only letters, digits, hyphens, underscores, or dots.",
    },
    "INVALID_ROOTCA_FINGERPRINT": {
        "message": "Invalid root CA SHA256 fingerprint.",
        "hint": "Use SHA256:AA:BB:... or OpenSSL output such as 'sha256 Fingerprint=AA:BB:...'.",
    },
    "ROOTCA_FINGERPRINT_MISMATCH": {
        "message": "Root CA SHA256 fingerprint does not match the expected out-of-band value.",
        "hint": "Verify that the signed zip came from the intended Project Admin.",
    },
    "SIGNED_ZIP_IDENTITY_CONFLICT": {
        "message": "Signed zip identity conflict.",
        "hint": "The signed zip project/org/name does not match the local request material.",
    },
    # --- Distributed provisioning: local site yaml ---
    "LOCAL_SITE_MISMATCH": {
        "message": "Local site.yaml does not match the signed zip identity.",
        "hint": "Use the site.yaml created by 'nvflare cert request' for this participant.",
    },
    "LOCAL_SITE_INVALID": {
        "message": "Local site.yaml is invalid or missing required fields.",
        "hint": "Use the site.yaml created by 'nvflare cert request', or re-run 'nvflare cert request'.",
    },
    "LOCAL_SITE_UNSUPPORTED_FEATURE": {
        "message": "Local site.yaml contains an unsupported feature.",
        "hint": "Remove or update the unsupported configuration in your participant definition file.",
    },
    # --- Distributed provisioning: key/cert ---
    "KEY_INVALID": {
        "message": "Private key is invalid or corrupt.",
        "hint": "Re-run 'nvflare cert request' to generate a new key pair.",
    },
    "KEY_CERT_MISMATCH": {
        "message": "Private key does not match the signed certificate.",
        "hint": "Ensure the private key from 'nvflare cert request' matches the signed zip from 'nvflare cert approve'.",
    },
    # --- Distributed provisioning: request directory ---
    "REQUEST_DIR_NOT_FOUND": {
        "message": "Request directory not found: {path}.",
        "hint": "Provide the directory created by 'nvflare cert request', or omit --request-dir to auto-discover.",
    },
    "REQUEST_DIR_INCOMPLETE": {
        "message": "Request directory is missing required local material.",
        "hint": "Re-run 'nvflare cert request' to regenerate the request directory.",
    },
    "REQUEST_DIR_MISMATCH": {
        "message": "Request directory does not match the signed zip request_id.",
        "hint": "Use the directory created by 'nvflare cert request' for this signed zip.",
    },
    # --- Distributed provisioning: request metadata ---
    "REQUEST_METADATA_NOT_FOUND": {
        "message": "Request metadata (request.json) not found in the request directory.",
        "hint": "Re-run 'nvflare cert request' to regenerate the request directory.",
    },
    "REQUEST_METADATA_INVALID": {
        "message": "Request metadata (request.json) is invalid or corrupted.",
        "hint": "Re-run 'nvflare cert request' to regenerate the request directory.",
    },
    "REQUEST_METADATA_MISMATCH": {
        "message": "Request metadata does not match the signed zip.",
        "hint": "Ensure the request directory matches the signed zip from 'nvflare cert approve'.",
    },
    # --- Distributed provisioning: project/CA binding ---
    "PROJECT_CA_MISMATCH": {
        "message": "Request project does not match the CA project.",
        "hint": "Use a CA directory initialized for the same project as this request.",
    },
    "PROJECT_PROFILE_MISMATCH": {
        "message": "Request project does not match the project profile.",
        "hint": "Use the project_profile.yaml for the same project as this request.",
    },
    # --- Package build ---
    "BUILD_FAILED": {
        "message": "Package build failed.",
        "hint": "Check builder configuration and logs for details.",
    },
    "UNSIGNED_JOB_REJECTED": {
        "message": "Unsigned job rejected — require_signed_jobs is enabled.",
        "hint": "Sign the job with an admin cert, or disable require_signed_jobs in fed_server.json.",
    },
    "CERT_CHAIN_INVALID": {
        "message": "Certificate {cert} is not signed by root CA {rootca}.",
        "hint": "Ensure the cert was signed by the Project Admin using the same root CA.",
    },
    "CERT_EXPIRED": {
        "message": "Certificate {cert} expired at {expiry}.",
        "hint": "Request a new certificate from the Project Admin.",
    },
    "PROJECT_FILE_NOT_FOUND": {
        "message": "Project file not found: {path}.",
        "hint": "Provide the path to a site-scoped project yaml file.",
    },
    "INVALID_PROJECT_FILE": {
        "message": "Invalid project file.",
        "hint": "Ensure the file is schema-compatible with 'nvflare provision' project.yaml (api_version: 3 or 4).",
    },
    "UNSUPPORTED_TOPOLOGY": {
        "message": "Relay participants found in project file — hierarchical FL is not supported by 'nvflare package'.",
        "hint": "Use 'nvflare provision' for relay topologies.",
    },
    "NO_PARTICIPANTS": {
        "message": "No participants to build after applying type filter.",
        "hint": "Check the project file and participant type filter.",
    },
    # --- Version ---
    "VERSION_MISMATCH": {
        "message": "Remote server and client sites are running different NVFlare versions.",
        "hint": "Run 'nvflare system version' to see per-site versions. Run 'nvflare preflight-check' to verify compatibility.",
    },
    # --- Job lifecycle ---
    "JOB_FAILED": {
        "message": "Job '{job_id}' reached terminal state FAILED.",
        "hint": "Use 'nvflare job logs <job_id>' and 'nvflare job meta <job_id>' to inspect the failure.",
    },
    "JOB_ABORTED": {
        "message": "Job '{job_id}' was aborted.",
        "hint": "Use 'nvflare job meta <job_id>' to see abort details.",
    },
    "JOB_FINISHED_EXCEPTION": {
        "message": "Job '{job_id}' reached terminal state FINISHED_EXCEPTION.",
        "hint": "Use 'nvflare job logs <job_id>' and 'nvflare job meta <job_id>' to inspect the failure.",
    },
    "JOB_ABANDONED": {
        "message": "Job '{job_id}' was abandoned.",
        "hint": "Use 'nvflare job meta <job_id>' to inspect the abandonment details.",
    },
    # --- Recipe / run ---
    "RECIPE_ENTRY_NOT_FOUND": {
        "message": "Recipe entry not found.",
        "hint": "Check --entry module:symbol matches a file in --recipe-folder. Run with --entry to specify explicitly.",
    },
    "RECIPE_ENTRY_AMBIGUOUS": {
        "message": "Multiple Recipe subclasses found; use --entry to select one.",
        "hint": "Use --entry module:ClassName to select one explicitly.",
    },
    "RECIPE_EXPORT_FAILED": {
        "message": "Recipe export failed.",
        "hint": "Check the recipe.export() implementation for errors.",
    },
    "RECIPE_RUNNER_FAILED": {
        "message": "Recipe runner failed.",
        "hint": "Check the recipe class run() method for errors.",
    },
}
# Freeze the registry into an immutable MappingProxyType so callers cannot accidentally
# add, remove, or overwrite entries at runtime.  Use get_error_entry() / get_error() to
# look up codes; use _ERROR_REGISTRY directly only when adding new entries in this file.
ERROR_REGISTRY: Mapping[str, Mapping[str, str]] = MappingProxyType(_ERROR_REGISTRY)


def get_error_entry(code: str) -> Optional[Mapping[str, str]]:
    entry = ERROR_REGISTRY.get(code)
    if entry is None and os.getenv("NVFLARE_DEV") == "1":
        raise KeyError(f"Unknown CLI error code: {code}")
    return entry


def get_error(code: str, **kwargs) -> Tuple[str, str]:
    """Return (message, hint) for the given error code with placeholders filled.

    Transitional helper for legacy cert/package call sites. New CLI code should prefer
    output_error()/output_error_message(). Falls back to a generic tuple for unknown codes.
    """
    entry = get_error_entry(code)
    if entry is None:
        return "Unknown error.", "Check logs for details."
    template = entry["message"]
    hint = entry["hint"]
    if code == "CONNECTION_FAILED":
        host = kwargs.get("host")
        port = kwargs.get("port")
        if host is not None and port is not None:
            return f"Cannot connect to the FLARE server at {host}:{port}.", hint
        if host is not None:
            return f"Cannot connect to the FLARE server at {host}.", hint
        return "Cannot connect to the FLARE server.", hint
    if code == "AUTH_FAILED" and "username" in kwargs:
        return f"Authentication failed for user {kwargs['username']}.", hint
    if code == "TIMEOUT" and "timeout" in kwargs:
        return f"Operation timed out after {kwargs['timeout']} seconds.", hint
    if code == "INVALID_ARGS" and "detail" in kwargs:
        return f"Invalid arguments: {kwargs['detail']}", hint
    try:
        message = template.format_map(kwargs)
    except KeyError:
        message = template
    return message, hint
