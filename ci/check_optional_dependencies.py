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

"""Smoke-test a wheel installed in a clean environment, without test/dev extras.

Run outside the checkout so source imports cannot conceal wheel packaging errors:
    cd /tmp && /path/to/venv/bin/python /path/to/checkout/ci/check_optional_dependencies.py base
"""

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

WEB_MODULES = (
    "flask",
    "flask_jwt_extended",
    "flask_sqlalchemy",
    "werkzeug",
    "jinja2",
    "itsdangerous",
    "blinker",
    "sqlalchemy",
    "gunicorn",
)
CORE_MODULES = (
    "nvflare.cli",
    "nvflare.private.fed.app.server.server_train",
    "nvflare.private.fed.app.client.client_train",
    "nvflare.fuel.flare_api.flare_api",
    "nvflare.client",
    "nvflare.edge.widgets.api_service",
    "jwt",  # Confidential-computing authorizers require PyJWT independently of Dashboard.
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("base", "dashboard", "edge-http"))
    args = parser.parse_args()

    import nvflare

    source_package = Path(__file__).resolve().parents[1] / "nvflare"
    assert Path(nvflare.__file__).resolve().parent != source_package, "Run against the installed wheel, not source"
    for name in CORE_MODULES:
        importlib.import_module(name)
    subprocess.run([sys.executable, "-m", "nvflare.cli", "--help"], check=True)

    if args.profile == "base":
        for name in WEB_MODULES:
            assert importlib.util.find_spec(name) is None, f"Base install unexpectedly includes {name}"
    else:
        importlib.import_module("nvflare.edge.web.routing_proxy")
        importlib.import_module("nvflare.edge.web.web_server")
        if args.profile == "edge-http":
            for name in ("flask_jwt_extended", "flask_sqlalchemy", "sqlalchemy", "gunicorn"):
                assert importlib.util.find_spec(name) is None, f"Edge HTTP install unexpectedly includes {name}"
        else:
            importlib.import_module("gunicorn")
            with tempfile.TemporaryDirectory() as web_root:
                os.environ["NVFL_WEB_ROOT"] = web_root
                os.environ["NVFL_CREDENTIAL"] = "admin@example.org:test-password:demo"
                os.environ["DATABASE_URL"] = "sqlite:///:memory:"
                application = importlib.import_module("nvflare.dashboard.application")
                app = application.init_app()
                assert app.test_client().get("/nvflare-dashboard/api/v1/clients").status_code == 401
    print(f"Optional dependency smoke check passed: {args.profile}")


if __name__ == "__main__":
    main()
