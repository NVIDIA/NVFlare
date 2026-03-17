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

"""Subprocess training script for large-model memory integration test.

Simulates the root cause scenario (large model exchanged between server, CJ, and
subprocess) at a scale that runs locally without a large machine:

- Model: ~4 MB total with each key > 2 MB → each key goes through the
  ViaDownloaderDecomposer streaming path (same code as the 5 GiB case).
- PASS_THROUGH: CJ receives LazyDownloadRef placeholders; this subprocess
  downloads each tensor directly from the FL server — the fix (Fixes 7/8)
  ensures CJ never materialises the tensors in its own memory.
- memory_gc_rounds=1: gc.collect() fires on every send (Fix 11/12).

Memory diagnostics are printed so test output shows the RSS profile.
"""

import os

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType


def _rss_mib():
    """Return RSS in MiB (Linux /proc) or None on other platforms."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None


def _log_rss(stage: str):
    rss = _rss_mib()
    if rss is not None:
        print(f"[MEM] stage={stage} pid={os.getpid()} rss_mib={rss:.1f}", flush=True)


def main():
    print(f"[train_script] subprocess started, pid={os.getpid()}", flush=True)

    flare.init()

    site = flare.get_site_name()
    job_id = flare.get_job_id()
    print(f"[train_script] site={site} job_id={job_id}", flush=True)

    while flare.is_running():
        _log_rss("before_receive")
        input_model = flare.receive()
        if input_model is None:
            print("[train_script] received None — exiting", flush=True)
            break

        np_data = input_model.params
        n_keys = len(np_data) if isinstance(np_data, dict) else 0
        size_mb = sum(v.size * v.itemsize for v in np_data.values()) / 1_000_000 if np_data else 0
        print(
            f"[train_script] round={input_model.current_round} n_keys={n_keys} size={size_mb:.1f} MB",
            flush=True,
        )
        _log_rss("after_receive")

        # Dummy "training": no-op, just send the model back unchanged.
        # This isolates the memory lifecycle from training compute.
        output_model = FLModel(
            params=np_data,
            params_type=ParamsType.FULL,
            metrics={"accuracy": 0.5 + 0.01 * (input_model.current_round or 0)},
            meta={"NUM_STEPS_CURRENT_ROUND": 1},
        )

        flare.send(output_model)
        _log_rss("after_send")

    print("[train_script] training loop done", flush=True)


if __name__ == "__main__":
    main()
