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

"""Verify the source archive and serve the Dashboard from an installed wheel."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

PREFIX = "nvflare/dashboard/application/static/nvflare-dashboard/"
SMOKE = """
from pathlib import Path
import hashlib
import json
import os
import nvflare
from nvflare.dashboard.application import init_app
assert Path(nvflare.__file__).resolve().is_relative_to(Path(os.environ['PYTHONPATH']).resolve())
app = init_app()
client = app.test_client()
base = '/nvflare-dashboard/'
response = client.get(base, follow_redirects=True)
assert response.status_code == 200 and b'/nvflare-dashboard/_next/' in response.data
manifest = client.get(base + 'dashboard-build.json').get_json()
assert manifest['files']
for name, expected in manifest['files'].items():
    asset = client.get(base + name)
    assert asset.status_code == 200, name
    assert hashlib.sha256(asset.data).hexdigest() == expected, name
assert client.get(base + 'api/v1/project').get_json()['status'] == 'ok'
print('Installed wheel serves Dashboard pages, notices, assets, and API without Node.')
"""


def check_wheel(wheel):
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(name.startswith("dashboard/") or "node_modules/" in name for name in names)
        manifest = json.loads(archive.read(PREFIX + "dashboard-build.json"))
        assert PREFIX + "THIRD_PARTY_NOTICES.txt" in names
        assert manifest["files"]
        for name, expected in manifest["files"].items():
            assert not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
            assert hashlib.sha256(archive.read(PREFIX + name)).hexdigest() == expected, name
        assert not any(
            name.startswith(PREFIX) and ("node_modules/" in name or name.endswith((".node", ".dylib", ".so")))
            for name in names
        )


def check_sdist(sdist):
    with tarfile.open(sdist) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    for required in (
        "dashboard/build_ui.py",
        "dashboard/smoke_wheel.py",
        "dashboard/ui/package.json",
        "dashboard/ui/package-lock.json",
        "dashboard/ui/.npmrc",
        "dashboard/ui/.nvmrc",
        "dashboard/ui/scripts/third-party-notices.mjs",
        "dashboard/ui/src/pages/index.tsx",
        PREFIX + "dashboard-build.json",
        PREFIX + "THIRD_PARTY_NOTICES.txt",
    ):
        assert required in names, required
    forbidden = {"node_modules", ".next", "out", "coverage", "test-results", "playwright-report"}
    assert not any(
        name.startswith("dashboard/ui/") and forbidden.intersection(PurePosixPath(name).parts) for name in names
    )
    assert not any(name.endswith(".tsbuildinfo") or Path(name).match(".env*.local") for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    check_wheel(wheel)
    if args.sdist:
        check_sdist(args.sdist)
    with tempfile.TemporaryDirectory(prefix="nvflare-dashboard-wheel-") as folder:
        temporary = Path(folder)
        target = temporary / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
        )
        environment = dict(
            os.environ,
            PYTHONPATH=str(target),
            PYTHONDONTWRITEBYTECODE="1",
            NVFL_WEB_ROOT=str(temporary / "database"),
            DATABASE_URL=f"sqlite:///{temporary / 'database' / 'db.sqlite'}",
            NVFL_CREDENTIAL="wheel-admin@example.test:Wheel-test-password!:Wheel_org",
            SECRET_KEY="isolated-dashboard-wheel-smoke-key-2026",
        )
        environment.pop("NVFL_DASHBOARD_STATIC_FOLDER", None)
        subprocess.run([sys.executable, "-c", SMOKE], cwd=temporary, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
