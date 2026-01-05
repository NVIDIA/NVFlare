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

"""WSGI entry point for Certificate Service.

Usage:
    # Development (Flask built-in server)
    python -m nvflare.cert_service.wsgi

    # Production (Gunicorn)
    gunicorn -w 4 -b 0.0.0.0:8443 nvflare.cert_service.wsgi:app

Environment Variables:
    NVFLARE_CERT_SERVICE_PORT: Port to listen on (default: 8443)
    NVFLARE_CERT_SERVICE_CONFIG: Path to config file (optional)
    NVFLARE_API_KEY: API key for admin operations
    NVFLARE_DATA_DIR: Data directory for CA and database (default: /var/lib/cert_service)
"""

import os

from nvflare.cert_service.app import CertServiceApp

# Environment variable names
ENV_PORT = "NVFLARE_CERT_SERVICE_PORT"
ENV_CONFIG = "NVFLARE_CERT_SERVICE_CONFIG"
ENV_API_KEY = "NVFLARE_API_KEY"
ENV_DATA_DIR = "NVFLARE_DATA_DIR"

# Load configuration
config_path = os.environ.get(ENV_CONFIG)
api_key = os.environ.get(ENV_API_KEY)
data_dir = os.environ.get(ENV_DATA_DIR, "/var/lib/cert_service")

# Initialize the application
cert_service_app = CertServiceApp(
    config_path=config_path,
    api_key=api_key,
    data_dir=data_dir,
)

# Export Flask app for WSGI servers (gunicorn, uwsgi, etc.)
app = cert_service_app.flask_app


if __name__ == "__main__":
    # Development server
    port = int(os.environ.get(ENV_PORT, "8443"))
    print(f"Starting Certificate Service on port {port}...")
    print(f"Data directory: {data_dir}")
    print(f"Config file: {config_path or '(using defaults)'}")
    print()
    print("API Endpoints:")
    print(f"  POST /api/v1/enroll     - Enrollment (token required)")
    print(f"  POST /api/v1/token      - Token generation (API key required)")
    print(f"  GET  /api/v1/ca-cert    - Download root CA certificate")
    print(f"  GET  /api/v1/health     - Health check")
    print()

    # Run Flask development server (HTTP only - use reverse proxy for HTTPS in production)
    app.run(host="0.0.0.0", port=port, debug=False)
