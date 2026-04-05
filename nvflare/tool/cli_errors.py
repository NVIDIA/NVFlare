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

from typing import Dict, Tuple

# Primary error registry — Phase 0+1 CLI commands use this via output_error().
ERROR_REGISTRY: Dict[str, Dict[str, str]] = {
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
        "hint": "Run the command with --schema for usage.",
    },
    "STARTUP_KIT_MISSING": {
        "message": "Startup kit not found.",
        "hint": "Set --startup or configure via 'nvflare config'.",
    },
    "SITE_NOT_FOUND": {
        "message": "Site '{site}' is not connected.",
        "hint": "Use 'nvflare system status' to list connected sites.",
    },
    "LOG_CONFIG_INVALID": {
        "message": "Log config is not valid JSON or a recognised log mode.",
        "hint": "Supply a valid dictConfig JSON file or one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, full, verbose, reload.",
    },
    "SERVER_UNREACHABLE": {
        "message": "Server stopped or job ended before command was delivered.",
        "hint": "Check server status with 'nvflare system status'.",
    },
    "INTERNAL_ERROR": {
        "message": "An unexpected error occurred.",
        "hint": "This is likely a bug. Run with --verbose for a traceback.",
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
    # --- Cert commands ---
    "OUTPUT_DIR_NOT_WRITABLE": {
        "message": "Cannot write to output directory {path}: {detail}",
        "hint": "Check directory permissions or choose a different output directory.",
    },
    "CERT_GENERATION_FAILED": {
        "message": "Failed to generate certificate: {detail}",
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
    "CSR_NOT_FOUND": {
        "message": "CSR file not found: {path}.",
        "hint": "Check the path to the .csr file.",
    },
    "INVALID_CSR": {
        "message": "Invalid or corrupt CSR file: {path}.",
        "hint": "Regenerate the CSR with 'nvflare cert csr'.",
    },
    "CERT_ALREADY_EXISTS": {
        "message": "Signed certificate already exists at {path}.",
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
        "message": "CSR generation failed: {detail}",
        "hint": "Check that the cryptography package is installed and up-to-date.",
    },
    "CERT_TYPE_UNKNOWN": {
        "message": "Unknown certificate type in '{cert}': the type embedded by 'nvflare cert sign' is missing or unrecognized.",
        "hint": "Re-sign the CSR with 'nvflare cert sign -t <type>' to embed the correct type.",
    },
    "CERT_SIGNING_FAILED": {
        "message": "Certificate signing failed: {reason}",
        "hint": "Check that the CA key and certificate are valid and not corrupted.",
    },
    # --- Package commands ---
    "CERT_NOT_FOUND": {
        "message": "Certificate file not found: {path}.",
        "hint": "Provide the signed certificate received from the Project Admin.",
    },
    "KEY_NOT_FOUND": {
        "message": "Private key file not found: {path}.",
        "hint": "Provide the private key generated by 'nvflare cert csr'.",
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
    "AMBIGUOUS_KEY": {
        "message": "Multiple *.key files found in {path}: {files}",
        "hint": "--dir mode packages one participant at a time. Use -n to select one or --key/--cert explicitly.",
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
        "message": "Invalid project file: {detail}",
        "hint": "Ensure the file is schema-compatible with 'nvflare provision' project.yaml (api_version: 3).",
    },
    "UNSUPPORTED_TOPOLOGY": {
        "message": "Relay participants found in project file — hierarchical FL is not supported by 'nvflare package'.",
        "hint": "Use 'nvflare provision' for relay topologies.",
    },
    "NO_PARTICIPANTS": {
        "message": "No participants to build after applying type filter.",
        "hint": "Check the project file and -t filter.",
    },
    # --- Version ---
    "VERSION_MISMATCH": {
        "message": "Remote server and client sites are running different NVFlare versions.",
        "hint": "Run 'nvflare system version' to see per-site versions. Run 'nvflare preflight' to verify compatibility.",
    },
    # --- Recipe / run ---
    "RECIPE_ENTRY_NOT_FOUND": {
        "message": "No Recipe class found.",
        "hint": "Specify --entry module:ClassName or add a Recipe subclass to the recipe folder.",
    },
    "RECIPE_ENTRY_AMBIGUOUS": {
        "message": "Multiple Recipe subclasses found; cannot auto-select.",
        "hint": "Use --entry module:ClassName to select one explicitly.",
    },
    "RECIPE_EXPORT_FAILED": {
        "message": "Recipe export failed.",
        "hint": "Check the recipe class export() method for errors.",
    },
    "RECIPE_RUNNER_FAILED": {
        "message": "Recipe runner failed.",
        "hint": "Check the recipe class run() method for errors.",
    },
}

# Backward-compatible tuple dict — cert/package commands call get_error() which returns (message, hint).
CLI_ERRORS: Dict[str, Tuple[str, str]] = {
    code: (entry["message"], entry["hint"]) for code, entry in ERROR_REGISTRY.items()
}


def get_error(code: str, **kwargs) -> Tuple[str, str]:
    """Return (message, hint) for the given error code with placeholders filled.

    Used by cert/package commands: message, hint = get_error("CODE", key=value)
    Falls back to a generic tuple for unknown codes.
    """
    entry = ERROR_REGISTRY.get(code)
    if entry is None:
        return "Unknown error.", "Check logs for details."
    template = entry["message"]
    hint = entry["hint"]
    try:
        message = template.format(**kwargs)
    except KeyError:
        message = template
    return message, hint
