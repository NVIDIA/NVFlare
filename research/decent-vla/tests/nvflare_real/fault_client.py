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

"""Test client for the real-FLARE tests: the real client loop with faults
injected into some sites. A test copies this file over the exported job's
custom/client_script.py and picks the faults with DVLA_TEST_FAULT, comma
separated:

    c1:raise                c1 raises in every round
    c1:sleep_once:<s>       c1 sleeps <s> seconds before its first round
    c1:fail_after_round:<r> c1 trains round <r>, then raises (once)
    c1:die                  c1's script fails before the client loop starts
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from decent_vla.runtime import client_session
from decent_vla.runtime.nvflare.client import client_main

_FAULTS = [f.split(":") for f in os.environ.get("DVLA_TEST_FAULT", "").split(",") if f]
_real_handle_round = client_session.handle_round
_real_run_client_round = client_session.run_client_round
_slept: set[str] = set()
_failed: set[str] = set()


def _handle_round(cid, *args, **kwargs):
    for who, kind, *rest in _FAULTS:
        if cid == who and kind == "raise":
            raise RuntimeError(f"injected failure on {cid}")
        if cid == who and kind == "sleep_once" and cid not in _slept:
            _slept.add(cid)
            time.sleep(float(rest[0]))
    return _real_handle_round(cid, *args, **kwargs)


def _run_client_round(**kwargs):
    update = _real_run_client_round(**kwargs)
    cid, round_num = kwargs["client_id"], kwargs["round_num"]
    for who, kind, *rest in _FAULTS:
        if cid == who and kind == "fail_after_round" and round_num == int(rest[0]) and cid not in _failed:
            _failed.add(cid)
            raise RuntimeError(f"injected failure on {cid} after training round {round_num}")
    return update


def main() -> None:
    import nvflare.client as flare

    if any(kind == "die" for _, kind, *_ in _FAULTS):
        flare.init()
        if any(who == flare.get_site_name() and kind == "die" for who, kind, *_ in _FAULTS):
            raise RuntimeError(f"injected death of {flare.get_site_name()}'s client script")
    client_session.handle_round = _handle_round
    client_session.run_client_round = _run_client_round
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    client_main(args.config, app_dir=str(Path(__file__).resolve().parent.parent), workdir=args.workdir)


if __name__ == "__main__":
    main()
