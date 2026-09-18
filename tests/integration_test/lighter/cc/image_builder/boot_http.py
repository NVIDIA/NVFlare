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

"""Boot a disposable test delivery and verify its generic HTTP application."""

import argparse
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from builder.common import digest_file, require, run, write_json
from builder.policy import verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Optional override; default uses the delivery's embedded bundle")
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected", required=True, help="Expected response line, without its newline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle or args.vault / "cvm_bundle"
    manifest = verify_bundle(bundle)
    require(manifest["profile_version"].startswith(("test-", "dev-")), "Use disposable test artifacts only")
    args.output.mkdir(parents=True)
    copy = args.output / "delivery"
    run(["cp", "-a", "--reflink=auto", args.vault, copy])
    log = args.output / "boot.log"
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    process = None
    started = time.monotonic()
    try:
        command = [str((copy / "launch_cvm.sh").resolve())]
        if args.bundle:
            command.extend(["--cvm-bundle", str(args.bundle.resolve())])
        with log.open("wb") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            require(process.poll() is None, "Guest exited before HTTP readiness; inspect boot.log")
            try:
                with http.open(f"http://127.0.0.1:{args.port}/", timeout=3) as response:
                    require(response.read() == (args.expected + "\n").encode(), "Unexpected application response")
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        else:
            raise RuntimeError("Generic application did not become ready")
        process.terminate()
        process.wait(timeout=45)
        write_json(
            args.output / "result.json",
            {
                "generic_application": True,
                "dev_mode": manifest.get("dev_mode", False),
                "platform": manifest["platform"],
                "manifest_sha256": digest_file(bundle / "cvm_manifest.json"),
                "zero_argument_launcher": args.bundle is None,
                "boot_log_sha256": digest_file(log),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )
        print("Generic HTTP application passed.")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=45)
        shutil.rmtree(copy)


if __name__ == "__main__":
    main()
