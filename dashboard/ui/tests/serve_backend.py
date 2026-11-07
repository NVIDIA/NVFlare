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

"""Serve the exported UI against an isolated, real NVFlare Dashboard database."""

import os
import shutil
import signal
from pathlib import Path
from tempfile import TemporaryDirectory

from werkzeug.serving import make_server

from nvflare.dashboard.application import db, init_app
from nvflare.dashboard.application.store import Store
from nvflare.dashboard.config import Config

ROOT = Path(__file__).resolve().parents[1]


def stop(_signum, _frame):
    raise SystemExit(0)


if __name__ == "__main__":
    if not (ROOT / "out" / "index.html").is_file():
        raise SystemExit("Build the Dashboard before running backend browser tests.")
    signal.signal(signal.SIGTERM, stop)
    with TemporaryDirectory(prefix="nvflare-mui-live-") as folder:
        temporary = Path(folder)
        static = temporary / "static"
        static.mkdir()
        shutil.copytree(ROOT / "out", static / "nvflare-dashboard")
        os.environ["NVFL_WEB_ROOT"] = folder
        os.environ["NVFL_CREDENTIAL"] = "admin@example.test:Testing1234!:Project_org"
        os.environ["NVFL_DASHBOARD_STATIC_FOLDER"] = str(static)
        Config.SQLALCHEMY_DATABASE_URI = f"sqlite:///{temporary / 'db.sqlite'}"
        Config.SECRET_KEY = "isolated-dashboard-browser-test-key-2026"
        app = init_app()
        with app.app_context():
            Store.set_project({"short_name": "mui-test", "server1": "server.test", "public": True})
        server = make_server("127.0.0.1", 3110, app, threaded=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
            with app.app_context():
                db.session.remove()
                db.engine.dispose()
